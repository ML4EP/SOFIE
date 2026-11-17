import numpy as np

from _helpers import FLOAT, INT64, helper, _model, _rand, _tensor, _vi

# Bias of the last layer, the only one with a single output.
OUT_BIAS = -0.022584186866879463
# Order of the weights in the file. The nodes use them in a different order.
MODULES = ["node_encoder", "edge_encoder", "edge_network", "node_network", "edge_decoder", "edge_output_transform"]


def make():
    """GNN: node and edge encoders, 8 edge updates with a node update between them, edge decoder."""
    hidden, n_iter = 128, 8  # n_iter is not free: see late_const()
    rng = np.random.RandomState(7)
    nodes, inits, used = [], [], {}

    def node(op, inputs, scope="", **attrs):
        """Adds a node named the way the exporter does: scope/Op, scope/Op_1, scope/Op_2, ..."""
        k = used[scope, op] = used.get((scope, op), -1) + 1
        name = f"{scope}/{op}_{k}" if k else f"{scope}/{op}"
        nodes.append(helper.make_node(op, inputs, [f"{name}_output_0"], name=name, **attrs))
        return f"{name}_output_0"

    def const(value):
        """Integer constant: a scalar for a number, 1-d for a list."""
        return node("Constant", [], value=_tensor("", INT64, np.shape(value), np.ravel(value).tolist()))

    def late_const():
        """Constant [0] that the exporter names by its position in the graph, not by scope."""
        # In the original the number in the node name is the node index + 198 and the one in the
        # output name the node index + 121, for all 28 of them. Checked for this graph only.
        k = len(nodes)
        out = f"onnx::Unsqueeze_{k + 121}"
        nodes.append(helper.make_node("Constant", [], [out], name=f"Constant_{k + 198}",
                                      value=_tensor("", INT64, [1], [0])))
        return out

    def mlp(x, name, sizes, last_relu=True):
        """nn.Sequential `name`: Linear layers with LayerNorm and Relu between them, plus a final Relu if last_relu."""
        for i, (n_in, n_out) in enumerate(zip(sizes, sizes[1:])):
            k = 3 * i  # position of this Linear in the Sequential
            layer, scope = f"{name}.{k}", f"/{name}/{name}."
            inits.append(_rand(rng, f"{layer}.weight", [n_out, n_in]))
            inits.append(_tensor(f"{layer}.bias", FLOAT, [1], [OUT_BIAS]) if n_out == 1 else
                         _rand(rng, f"{layer}.bias", [n_out]))
            x = node("Gemm", [x, f"{layer}.weight", f"{layer}.bias"], f"{scope}{k}", alpha=1.0, beta=1.0, transB=1)
            if i < len(sizes) - 2:
                # Scale 1 and bias 0 of the LayerNorm are Constant nodes, not initializers.
                norm = [node("Constant", [], f"{scope}{k + 1}", value=_tensor("", FLOAT, [n_out], [v] * n_out))
                        for v in (1.0, 0.0)]
                x = node("LayerNormalization", [x] + norm, f"{scope}{k + 1}", axis=-1, epsilon=1e-5)
                x = node("Relu", [x], f"{scope}{k + 2}")
            elif last_relu:
                x = node("Relu", [x], f"{scope}{k + 1}")
        return x

    def dim(x, axis):
        """Size of x along axis."""
        return node("Gather", [node("Shape", [x]), const(axis)], axis=0)

    def stack(sizes, late=False):
        """Shape tensor from scalar sizes."""
        return node("Concat", [node("Unsqueeze", [s, late_const() if late else const([0])]) for s in sizes], axis=0)

    def expand(column, shape):
        """Repeats an index column to shape. The exporter first replaces any -1 in shape by 1."""
        shape = node("Reshape", [shape, const([-1])], allowzero=0)
        ones = node("ConstantOfShape", [node("Shape", [shape])], value=_tensor("", INT64, [1], [1]))
        is_minus_one = node("Equal", [shape, node("Mul", [ones, const(-1)])])
        return node("Expand", [column, node("Where", [is_minus_one, ones, shape])])

    def scatter_add(shape, rows, values):
        """Zeros of the given shape with values summed into the rows that the index names."""
        zeros = node("ConstantOfShape", [shape], value=_tensor("", FLOAT, [1], [0.0]))
        size = node("Shape", [rows])
        values = node("Slice", [values, const([0, 0]), size])
        return node("ScatterElements", [zeros, rows, values], axis=0, reduction="add")

    src, dst = (node("Gather", ["edge_index", const(i)], axis=0) for i in range(2))
    h = mlp("x", "node_encoder", [12] + 3 * [hidden])
    e = mlp("edge_attr", "edge_encoder", [6] + 3 * [hidden])

    for i in range(n_iter):
        # Edge update from the edge and its two end nodes.
        ends = [node("Gather", [h, end], axis=0) for end in (src, dst)]
        e = mlp(node("Concat", [e] + ends, axis=-1), f"edge_network.{i}", [3 * hidden] + 3 * [hidden])
        if i == n_iter - 1:
            break

        # Node update from the node and the sums of its incoming and outgoing edges.
        # The index columns are made in the first iteration and reused afterwards.
        n_nodes = dim(h, 0)
        if i == 0:
            dst_column = node("Unsqueeze", [dst, const([-1])])
        n_edges, width = dim(e, 0), dim(e, 1)
        dst_rows = expand(dst_column, stack([n_edges, width]))
        shapes = [stack([n_nodes, width], late=True) for _ in range(2)]
        incoming = scatter_add(shapes[0], dst_rows, e)
        if i == 0:
            src_column = node("Unsqueeze", [src, const([-1])])
        src_rows = expand(src_column, stack([n_edges, width]))
        outgoing = scatter_add(shapes[1], src_rows, e)
        h = mlp(node("Concat", [incoming, outgoing, h], axis=-1), f"node_network.{i}", [3 * hidden] + 3 * [hidden])

    e = mlp(e, "edge_decoder", 4 * [hidden])
    e = mlp(e, "edge_output_transform", [hidden, hidden, 1], last_relu=False)
    nodes.append(helper.make_node("Cast", [node("Squeeze", [e, const([1])])], ["output"], name="/Cast", to=FLOAT))

    inits.sort(key=lambda t: MODULES.index(t.name.split(".")[0]))
    graph = helper.make_graph(
        nodes, "main_graph",
        [_vi("x", FLOAT, ["n_nodes", 12]), _vi("edge_index", INT64, [2, "n_edges"]),
         _vi("edge_attr", FLOAT, ["n_edges", 6])],
        [_vi("output", FLOAT, ["Castoutput_dim_0"])], initializer=inits)
    return _model(graph, 17, 8, producer_name="pytorch", producer_version="2.4.0")
