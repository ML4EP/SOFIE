"""TICL track-linking GNN in two sizes: scores every edge of a graph (PyTorch TorchScript exporter)."""

import numpy as np

from _helpers import FLOAT, INT32, INT64, _model, _rand, _tensor, _vi, helper

NODE_FEATURES, EDGE_FEATURES = 29, 5


def make_gnn(hidden, iterations, biases):
    """MLP encoders for nodes and edges, attention-weighted message passing layers, then an MLP scoring each edge."""
    rng = np.random.RandomState(7)
    H, K = hidden, iterations

    # Linear layers in the order of their weights in the file: name, inputs, outputs.
    layers = [("inputnetwork.0", NODE_FEATURES, H), ("inputnetwork.3", H, H),
              ("edge_inputnetwork.0", EDGE_FEATURES, H), ("edge_inputnetwork.3", H, H),
              ("attention_direct.0", H, H), ("attention_direct.3", H, 1),
              ("attention_reverse.0", H, H), ("attention_reverse.3", H, 1)]
    for i in range(K):
        layers += [(f"graphconvs.{i}.conv_0", 2 * H, 2 * H), (f"graphconvs.{i}.convs.0", 2 * H, H),
                   (f"graphconvs.{i}.sc", H, H)]
    layers += [("edgenetwork.0", 3 * H + EDGE_FEATURES, H), ("edgenetwork.3", H, H),
               ("outputnetwork.0", H, H), ("outputnetwork.2", H, 1)]

    inits = [_tensor("node_scaler", FLOAT, [NODE_FEATURES], [1.0] * NODE_FEATURES),
             _tensor("edge_scaler", FLOAT, [EDGE_FEATURES], [1.0] * EDGE_FEATURES)]
    for name, n_in, n_out in layers:
        inits.append(_rand(rng, name + ".weight", [n_out, n_in]))
        # The one-element biases keep their exported value, the others are random.
        inits.append(_rand(rng, name + ".bias", [n_out]) if n_out > 1
                     else _tensor(name + ".bias", FLOAT, [1], [biases[name]]))
    inits.append(_tensor("sofie_patched_squeeze_axes", INT64, [1], [-1]))

    nodes, used = [], {}

    def op(scope, op_type, inputs, out=None, **attrs):
        """Adds a node named as the exporter does: scope/OpType, with _1, _2, ... for repeats in the scope."""
        n = used.get((scope, op_type), 0)
        used[scope, op_type] = n + 1
        name = f"{scope}/{op_type}" + (f"_{n}" if n else "")
        out = out or name + "_output_0"
        nodes.append(helper.make_node(op_type, inputs, [out], name=name, **attrs))
        return out

    def const(scope, vals, dtype=INT64):
        """Constant node holding a scalar, or a 1-D tensor when vals is a list."""
        dims, vals = ([len(vals)], vals) if isinstance(vals, list) else ([], [vals])
        return op(scope, "Constant", [], value=_tensor("", dtype, dims, vals))

    def filled(scope, shape, dtype, val):
        return op(scope, "ConstantOfShape", [shape], value=_tensor("", dtype, [1], [val]))

    def dim(scope, x, axis):
        """Size of x along axis, as a scalar."""
        return op(scope, "Gather", [op(scope, "Shape", [x]), const(scope, axis)], axis=0)

    def unsqueeze(scope, x, skew):
        """Unsqueeze on axis 0 to put a size into a shape.

        The axes come from an unscoped Constant named Constant_<c> with output onnx::Unsqueeze_<v>.
        Both numbers follow the node position. The offsets and skew are fitted to the two exports,
        (32, 2) and (64, 4), and are not checked for other sizes.
        """
        axes = f"onnx::Unsqueeze_{len(nodes) + 6 * K + 30 + skew}"
        nodes.append(helper.make_node("Constant", [], [axes], name=f"Constant_{len(nodes) + 10 * K + 7}",
                                      value=_tensor("", INT64, [1], [0])))
        return op(scope, "Unsqueeze", [x, axes])

    def linear(scope, param, x):
        return op(scope, "Gemm", [x, param + ".weight", param + ".bias"], alpha=1.0, beta=1.0, transB=1)

    def layer(name, k, x, act="LeakyRelu"):
        """Linear name.k with the activation name.(k + 1)."""
        x = linear(f"/{name}/{name}.{k}", f"{name}.{k}", x)
        return op(f"/{name}/{name}.{k + 1}", act, [x], **({"alpha": 0.01} if act == "LeakyRelu" else {}))

    def mlp(name, x, act="LeakyRelu"):
        """Layers 0 and 3 of name. Index 2 has no node in the exports."""
        return layer(name, 3, layer(name, 0, x), act)

    shared = {}  # made by the first graph conv layer, reused by the later ones

    def graph_conv(i, x):
        """A message per edge from an MLP on (x_i, x_j - x_i), weighted and summed into node i, plus a skip Linear."""
        s = f"/graphconvs.{i}"
        skew = 2 * i + 1

        def fc(child, x):
            return linear(f"{s}/{child}", f"graphconvs.{i}.{child}", x)

        x_i = op(s, "Gather", [x, receivers], axis=0)
        x_j = op(s, "Gather", [x, senders], axis=0)
        pair = op(s, "Concat", [x_i, op(s, "Sub", [x_j, x_i])], axis=1)
        n_nodes = dim(s, x, 0)
        ones = filled(s, op(s, "Concat", [unsqueeze(s, n_nodes, skew)], axis=0), FLOAT, 1.0)
        msg = op(f"{s}/act_0", "Relu", [fc("conv_0", pair)])
        msg = op(f"{s}/acts.0", "Relu", [fc("convs.0", msg)])

        # Message weights: 1 for the first n_nodes messages, the attention for the edges.
        if i == 0:
            shared["attention"] = op(s, "Squeeze", [attention, "sofie_patched_squeeze_axes"])
        weight = op(s, "Concat", [ones, shared["attention"]], axis=0)
        msg = op(s, "Transpose", [op(s, "Mul", [weight, op(s, "Transpose", [msg], perm=[1, 0])])], perm=[1, 0])

        # Scatter-add of the messages into a zero [n_nodes, channels] tensor. The receiver index is
        # repeated over the channels by Expand to a shape of ones followed by Tile. The repeats
        # [1, channels] are built twice: once to count them for Expand, once for Tile.
        channels = dim(s, msg, 1)
        total = op(s, "Concat", [unsqueeze(s, n_nodes, skew), unsqueeze(s, channels, skew)], axis=0)
        total = filled(s, total, FLOAT, 0.0)
        if i == 0:
            shared["index"] = op(s, "Unsqueeze", [receivers, const(s, [1])])
        counted = op(s, "Concat", [const(s, [1]), unsqueeze(s, channels, skew + 1)], axis=0)
        repeats = op(s, "Concat", [const(s, [1]), unsqueeze(s, channels, skew + 2)], axis=0)
        index = op(s, "Expand", [shared["index"], filled(s, op(s, "Shape", [counted]), INT64, 1)])
        index = op(s, "Tile", [index, repeats])
        size = op(s, "Shape", [index])
        msg = op(s, "Slice", [msg, const(s, [0, 0]), size])
        total = op(s, "ScatterElements", [total, index, msg], axis=0, reduction="add")
        return op(f"{s}/sc_act", "Relu", [op(s, "Add", [fc("sc", x), total])])

    edge_in = op("", "Div", [op("", "Add", ["edge_features", const("", 1e-4, FLOAT)]), "edge_scaler"])
    node_in = op("", "Div", ["node_features", "node_scaler"])
    edge = mlp("edge_inputnetwork", edge_in)
    # One attention weight per edge and direction.
    attention = [mlp("attention_direct", edge, "Sigmoid"), mlp("attention_reverse", edge, "Sigmoid")]
    attention = op("", "Concat", attention, axis=0)
    x = mlp("inputnetwork", node_in)

    # Receiving and sending node of every message: n_nodes zeros, then each edge in both directions.
    zeros = filled("", op("", "Concat", [unsqueeze("", dim("", node_in, 0), 0)], axis=0), INT32, 0)
    ends = ["/Split_output_0", "/Split_output_1"]
    nodes.append(helper.make_node("Split", ["edge_index", const("", [1, 1])], ends, name="/Split", axis=1))
    src = op("", "Squeeze", [ends[0], const("", [1])])
    dst = op("", "Squeeze", [ends[1], const("", [1])])
    receivers = op("", "Concat", [op("", "Cast", [zeros], to=INT64), src, dst], axis=0)
    senders = op("", "Concat", [op("", "Cast", [zeros], to=INT64), dst, src], axis=0)

    for i in range(K):
        x = graph_conv(i, x)

    # Edge classifier on both end nodes, the edge embedding and the scaled edge features.
    pair = [op("", "Gather", [x, src], axis=0), op("", "Gather", [x, dst], axis=0)]
    edge = mlp("edgenetwork", op("", "Concat", pair + [edge, edge_in], axis=-1))
    out = linear("/outputnetwork/outputnetwork.2", "outputnetwork.2", layer("outputnetwork", 0, edge))
    op("/scale", "Sigmoid", [out], out="edge_scores")

    graph = helper.make_graph(
        nodes, "main_graph",
        inputs=[_vi("node_features", FLOAT, ["n_nodes", NODE_FEATURES]),
                _vi("edge_features", FLOAT, ["n_edges", EDGE_FEATURES]),
                _vi("edge_index", INT64, ["n_edges", 2])],
        outputs=[_vi("edge_scores", FLOAT, ["n_edges", 1])],
        initializer=inits)
    return _model(graph, opset=17, ir_version=8, producer_name="pytorch", producer_version="2.6.0")


MODELS = {
    "gnn_h32_k2": lambda: make_gnn(32, 2, {"attention_direct.3": -0.04276712238788605,
                                           "attention_reverse.3": -0.11728253960609436,
                                           "outputnetwork.2": -0.05991296470165253}),
    "gnn_h64_k4": lambda: make_gnn(64, 4, {"attention_direct.3": 0.020984411239624023,
                                           "attention_reverse.3": 0.03993186354637146,
                                           "outputnetwork.2": 0.05626416206359863}),
}
