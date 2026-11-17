"""PUNet: scores the edges of a graph from its node and edge features.

The graphs follow the torch export node for node, so the helpers below also
reproduce the names and running numbers the exporter left behind.
"""

import numpy as np

from _helpers import FLOAT, INT32, INT64, _model, _rand, _tensor, _vi, helper

HEADS, LAYERS = 4, 2  # of the transformer encoder

# Encoder parameters that are all zero (biases) or all one (norm weights).
# The exports keep the first of each kind and make the others Identity copies of it.
SHARED = [f"encoder_layers.{layer}.{p}" for layer in range(LAYERS) for p in (
    "self_attn.W_q.bias", "self_attn.W_k.bias", "self_attn.W_v.bias", "self_attn.W_o.bias",
    "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "final_norm.weight", "final_norm.bias")]
ZEROS, ONES = SHARED[0], SHARED[4]  # the first bias and the first norm weight

# Order of the parameter groups in the initializer list. The nodes come in another order.
DECLARED = ("node_scaler", "edge_scaler", "inputnetwork", "encoder_layers", "edge_inputnetwork",
            "attention_direct", "attention_reverse", "graphconvs", "edgenetwork", "outputnetwork")


# The steps in the two id tables below were read off the two exports (LAYERS = 2, k = 2 and 4).
def _axes_ids(first, k):
    """Value ids of the Unsqueeze axes the exporter adds, in graph order."""
    for layer in range(LAYERS):
        late = 91 if layer else 92  # one higher in the first layer
        # two per head split (q, k, v), then three for the merge
        yield from (first + 124 * layer + s for s in (0, 2, 20, 22, 40, 42, late, late + 2, late + 4))
    yield first + 249  # shape of the per-node index entries
    for conv in range(k):
        # the first graph convolution also makes the two tensors that the later ones reuse
        steps = (3, 18, 20, 26, 31) if conv else (0, 16, 18, 26, 31)
        yield from (first + 271 + 49 * conv + s for s in steps)


def _weight_ids(first, k):
    """Value ids of the encoder MatMul weights. They come after the ids of the k graph convolutions."""
    for layer in range(LAYERS):
        # W_q, W_k, W_v, W_o, fc1, fc2
        yield from (first + 278 + 49 * k + 14 * layer + s for s in (0, 3, 6, 11, 12, 13))


class _Graph:
    """Node and initializer lists, named the way the torch exporter names them."""

    def __init__(self, hidden, k, const_offset, first_id):
        self.hidden = hidden
        self.nodes, self.params, self.weights = [], [], []
        self.rng = np.random.RandomState(7)
        self.used = {}  # how often each scope/Op name was taken
        self.shared = {}  # outputs kept by once()
        self.count = 0  # nodes made so far, those of the If branches included
        self.const_offset = const_offset
        self.axes_ids = _axes_ids(first_id, k)
        self.weight_ids = _weight_ids(first_id, k)

    def _emit(self, op, name, inputs, out, **attrs):
        self.nodes.append(helper.make_node(op, inputs, [out] if isinstance(out, str) else out, name=name, **attrs))
        self.count += 1
        return out

    def node(self, scope, op, inputs, out=None, **attrs):
        """Adds scope/Op (then scope/Op_1, ...) and returns its output name."""
        name = f"{scope}/{op}"
        n = self.used[name] = self.used.get(name, -1) + 1
        if n:
            name += f"_{n}"
        return self._emit(op, name, inputs, out or name + "_output_0", **attrs)

    def own(self, op, inputs, out, **attrs):
        """Node without a scope: named Op_<n>, n the count of nodes made so far plus const_offset."""
        return self._emit(op, f"{op}_{self.count + self.const_offset}", inputs, out, **attrs)

    def const(self, scope, val, dtype=INT64):
        """Constant node: a 1-D tensor if val is a list, else a scalar."""
        dims, vals = ([len(val)], val) if isinstance(val, list) else ([], [val])
        return self.node(scope, "Constant", [], value=_tensor("", dtype, dims, vals))

    def dim(self, scope, x, axis):
        """Run-time size of one axis of x."""
        shape = self.node(scope, "Shape", [x])
        return self.node(scope, "Gather", [shape, self.const(scope, axis)], axis=0)

    def shape(self, scope, *dims):
        """Shape tensor from fixed sizes (int) and run-time sizes (tensor name)."""
        parts = []
        for d in dims:
            if isinstance(d, int):
                parts.append(self.const(scope, [d]))
            else:
                axes = self.own("Constant", [], f"onnx::Unsqueeze_{next(self.axes_ids)}",
                                value=_tensor("", INT64, [1], [0]))
                parts.append(self.node(scope, "Unsqueeze", [d, axes]))
        return self.node(scope, "Concat", parts, axis=0)

    def fill(self, scope, shape, value, dtype=FLOAT):
        """Tensor of a run-time shape with every element set to value."""
        return self.node(scope, "ConstantOfShape", [shape], value=_tensor("", dtype, [1], [value]))

    def once(self, key, make):
        """Output of a node that the first caller emits and the later ones reuse."""
        if key not in self.shared:
            self.shared[key] = make()
        return self.shared[key]

    def branch(self, name, build, out_shape):
        """Subgraph of an If. Its nodes take their names from the same counters as the main graph."""
        main, self.nodes = self.nodes, []
        out = build()
        nodes, self.nodes = self.nodes, main
        return helper.make_graph(nodes, name, [], [_vi(out, FLOAT, out_shape)])

    def param(self, name, dims, vals=None):
        """Named torch parameter: random, or the given values."""
        self.params.append(_rand(self.rng, name, dims) if vals is None else _tensor(name, FLOAT, dims, vals))
        return name

    def gemm(self, scope, param, x, n_in, n_out, bias=None):
        """Linear layer as one Gemm. A single-element bias is given by value."""
        w = self.param(param + ".weight", [n_out, n_in])
        b = self.param(param + ".bias", [n_out], None if bias is None else [bias])
        return self.node(scope, "Gemm", [x, w, b], alpha=1.0, beta=1.0, transB=1)

    def linear(self, scope, bias, x, n_in, n_out):
        """Linear layer as MatMul and Add. Its weight is named by value id and listed after the parameters."""
        w = f"onnx::MatMul_{next(self.weight_ids)}"
        self.weights.append(_rand(self.rng, w, [n_in, n_out]))
        return self.node(scope, "Add", [bias, self.node(scope, "MatMul", [x, w])])


def _mlp(g, name, x, n_in, n_out, last=3, act="LeakyRelu", bias=None):
    """Linear, LeakyRelu, Linear, then act if given. In the names the second Linear is layer `last`."""
    scope, slope = f"/{name}/{name}", dict(alpha=0.01)
    x = g.gemm(f"{scope}.0", f"{name}.0", x, n_in, g.hidden)
    x = g.node(f"{scope}.1", "LeakyRelu", [x], **slope)
    x = g.gemm(f"{scope}.{last}", f"{name}.{last}", x, g.hidden, n_out, bias)
    if act:
        x = g.node(f"{scope}.{last + 1}", act, [x], **(slope if act == "LeakyRelu" else {}))
    return x


def _encoder_layer(g, layer, x):
    """Pre-norm transformer layer: cosine-similarity attention and a feed-forward block, each with a residual."""
    h = g.hidden
    scope, param = f"/encoder_layers.{layer}", f"encoder_layers.{layer}"
    attn, ff = scope + "/self_attn", scope + "/feed_forward"

    # The initializers of the layer, in their order in the exports: attention, feed-forward block, norms.
    if layer == 0:
        g.param(ZEROS, [h], h * [0.0])
    fc1_bias = g.param(f"{param}.feed_forward.fc1.bias", [2 * h])
    fc2_bias = g.param(f"{param}.feed_forward.fc2.bias", [h])
    if layer == 0:
        g.param(ONES, [h], h * [1.0])

    def norm(name, x):
        return g.node(f"{scope}/{name}", "LayerNormalization", [x, f"{param}.{name}.weight", f"{param}.{name}.bias"],
                      axis=-1, epsilon=1e-4)

    def heads(name, x):
        """Projection of x split into heads: (batch, length, h) to (batch, heads, length, h / heads)."""
        x = g.linear(f"{attn}/{name}", f"{param}.self_attn.{name}.bias", x, h, h)
        size = g.shape(attn, g.dim(attn, x, 0), g.dim(attn, x, 1), HEADS, h // HEADS)
        return g.node(attn, "Transpose", [g.node(attn, "Reshape", [x, size], allowzero=0)], perm=[0, 2, 1, 3])

    def unit(x):
        """x scaled to unit length along the last axis."""
        length = g.node(attn, "ReduceL2", [x], axes=[-1], keepdims=1)
        length = g.node(attn, "Clip", [length, g.const(attn, 1e-12, FLOAT), ""])
        return g.node(attn, "Div", [x, g.node(attn, "Expand", [length, g.node(attn, "Shape", [x])])])

    y = norm("norm1", x)
    q, k, v = (heads(name, y) for name in ("W_q", "W_k", "W_v"))
    q, k = (g.node(attn, "Cast", [t], to=FLOAT) for t in (q, k))
    q, k = unit(q), unit(k)
    # softmax of the cosine similarities over a temperature of 0.5
    score = g.node(attn, "MatMul", [q, g.node(attn, "Transpose", [k], perm=[0, 1, 3, 2])])
    score = g.node(attn, "Div", [score, g.const(attn, 0.5, FLOAT)])
    score = g.node(attn, "Sub", [score, g.node(attn, "ReduceMax", [score], axes=[-1], keepdims=1)])
    score = g.node(attn, "Softmax", [score], axis=-1)
    y = g.node(attn, "MatMul", [score, g.node(attn, "Cast", [v], to=FLOAT)])
    y = g.node(attn, "Cast", [y], to=FLOAT)
    # heads merged again: (batch, heads, length, h / heads) to (batch, length, h)
    batch, n_heads, length, width = (g.dim(attn, y, axis) for axis in range(4))
    y = g.node(attn, "Transpose", [y], perm=[0, 2, 1, 3])
    size = g.shape(attn, batch, length, g.node(attn, "Mul", [n_heads, width]))
    y = g.node(attn, "Reshape", [y, size], allowzero=0)
    x = g.node(scope, "Add", [x, g.linear(f"{attn}/W_o", f"{param}.self_attn.W_o.bias", y, h, h)])

    y = g.linear(f"{ff}/fc1", fc1_bias, norm("norm2", x), h, 2 * h)
    y = g.linear(f"{ff}/fc2", fc2_bias, g.node(f"{ff}/relu", "Relu", [y]), 2 * h, h)
    return norm("final_norm", g.node(scope, "Add", [x, y]))


def _graph_conv(g, conv, x, center, neighbour, attention):
    """Edge convolution: an MLP of (x_i, x_j - x_i) per entry, weighted and summed into node i, plus a shortcut."""
    h = g.hidden
    scope, param = f"/graphconvs.{conv}", f"graphconvs.{conv}"
    xi, xj = (g.node(scope, "Gather", [x, index], axis=0) for index in (center, neighbour))
    msg = g.node(scope, "Concat", [xi, g.node(scope, "Sub", [xj, xi])], axis=1)
    n_nodes = g.dim(scope, x, 0)
    ones = g.fill(scope, g.shape(scope, n_nodes), 1.0)
    msg = g.node(f"{scope}/act_0", "Relu", [g.gemm(f"{scope}/conv_0", f"{param}.conv_0", msg, 2 * h, 2 * h)])
    msg = g.node(f"{scope}/acts.0", "Relu", [g.gemm(f"{scope}/convs.0", f"{param}.convs.0", msg, 2 * h, h)])

    # weights: 1 for the per-node entries, the attention for the edges
    edge_weight = g.once("edge_weight", lambda: g.node(scope, "Squeeze", [attention, "sofie_patched_squeeze_axes"]))
    weight = g.node(scope, "Concat", [ones, edge_weight], axis=0)
    msg = g.node(scope, "Mul", [weight, g.node(scope, "Transpose", [msg], perm=[1, 0])])
    msg = g.node(scope, "Transpose", [msg], perm=[1, 0])

    # scatter-add into the nodes, with the node index repeated across the columns
    width = g.dim(scope, msg, 1)
    total = g.fill(scope, g.shape(scope, n_nodes, width), 0.0)
    column = g.once("column", lambda: g.node(scope, "Unsqueeze", [center, g.const(scope, [1])]))
    size, repeats = g.shape(scope, 1, width), g.shape(scope, 1, width)
    index = g.node(scope, "Expand", [column, g.fill(scope, g.node(scope, "Shape", [size]), 1, INT64)])
    index = g.node(scope, "Tile", [index, repeats])
    index_shape = g.node(scope, "Shape", [index])
    msg = g.node(scope, "Slice", [msg, g.const(scope, [0, 0]), index_shape])
    total = g.node(scope, "ScatterElements", [total, index, msg], axis=0, reduction="add")

    x = g.node(scope, "Add", [g.gemm(f"{scope}/sc", f"{param}.sc", x, h, h), total])
    return g.node(f"{scope}/sc_act", "Relu", [x])


def make_punet(hidden, k, const_offset, first_id, last_bias):
    """PUNet: node and edge MLPs, a transformer encoder, k attention-weighted graph convolutions, an edge MLP."""
    h = hidden
    g = _Graph(h, k, const_offset, first_id)
    for name in reversed(SHARED):
        if name not in (ZEROS, ONES):
            g.own("Identity", [ONES if name.endswith("weight") else ZEROS], name)

    # scaled inputs
    node_scaler, edge_scaler = g.param("node_scaler", [29], 29 * [1.0]), g.param("edge_scaler", [5], 5 * [1.0])
    edge_in = g.node("", "Div", [g.node("", "Add", ["edge_features", g.const("", 1e-4, FLOAT)]), edge_scaler])
    node_in = g.node("", "Div", ["node_features", node_scaler])

    # edge embedding, and from it one attention weight per edge and direction
    edge = _mlp(g, "edge_inputnetwork", edge_in, 5, h)
    attention = [_mlp(g, f"attention_{way}", edge, h, 1, act="Sigmoid", bias=b)
                 for way, b in zip(("direct", "reverse"), last_bias)]
    attention = g.node("", "Concat", attention, axis=0)

    # node embedding, passed through the encoder as a batch of one sequence
    x = _mlp(g, "inputnetwork", node_in, 29, h)
    x = g.node("", "Unsqueeze", [x, g.const("", [0])])
    for layer in range(LAYERS):
        x = _encoder_layer(g, layer, x)
    # torch's squeeze(0): the batch axis is dropped only if its size is one
    axis = g.const("", [0])
    batch = g.node("", "Gather", [g.node("", "Shape", [x]), axis], axis=0)
    is_one = g.node("", "Equal", [batch, g.const("", [1])])
    squeezed = g.branch("sub_graph", lambda: g.node("", "Squeeze", [x, g.const("", [0])]),
                        ["Squeeze/Squeeze_output_0_dim_0", h])
    kept = g.branch("sub_graph1", lambda: g.node("", "Identity", [x]),
                    ["Identity/Identity_output_0_dim_0", "Squeeze/Squeeze_output_0_dim_0", h])
    x = g.node("", "If", [is_one], then_branch=squeezed, else_branch=kept)

    # index lists for the convolutions: a zero per node, then every edge in both directions
    per_node = g.fill("", g.shape("", g.dim("", node_in, 0)), 0, INT32)
    ends = g.node("", "Split", ["edge_index", g.const("", [1, 1])], ["/Split_output_0", "/Split_output_1"], axis=1)
    first, second = (g.node("", "Squeeze", [end, g.const("", [1])]) for end in ends)
    center, neighbour = (g.node("", "Concat", [g.node("", "Cast", [per_node], to=INT64), a, b], axis=0)
                         for a, b in ((first, second), (second, first)))
    for conv in range(k):
        x = _graph_conv(g, conv, x, center, neighbour, attention)

    # score of an edge from its two nodes, its embedding and its scaled features
    x = [g.node("", "Gather", [x, end], axis=0) for end in (first, second)]
    x = _mlp(g, "edgenetwork", g.node("", "Concat", x + [edge, edge_in], axis=-1), 3 * h + 5, h)
    x = _mlp(g, "outputnetwork", x, h, 1, last=2, act=None, bias=last_bias[2])
    g.node("/scale", "Sigmoid", [x], "edge_scores")

    params = sorted(g.params, key=lambda t: DECLARED.index(t.name.split(".")[0]))
    squeeze_axes = _tensor("sofie_patched_squeeze_axes", INT64, [1], [-1])
    graph = helper.make_graph(
        g.nodes, "main_graph",
        [_vi("node_features", FLOAT, ["n_nodes", 29]), _vi("edge_features", FLOAT, ["n_edges", 5]),
         _vi("edge_index", INT64, ["n_edges", 2])],
        [_vi("edge_scores", FLOAT, ["n_edges", 1])],
        initializer=params + g.weights + [squeeze_axes])
    return _model(graph, opset=17, ir_version=8, producer_name="pytorch", producer_version="2.6.0")


# Per model: hidden size, number of graph convolutions k, and three things read off its export:
# the offset in the names of the Identity_<n> and Constant_<n> nodes, the value id of the first
# Unsqueeze axes, and the single-element biases of the two attention MLPs and of the output MLP.
# In both exports the offset is 66 + 10 * k and the first id is 107 + 6 * k.
MODELS = {
    "punet_h32_k2_heads4_layers2": lambda: make_punet(32, 2, 86, 119, (0.0686774, 0.0064797616, 0.090150945)),
    "punet_h64_k4_heads4_layers2": lambda: make_punet(64, 4, 106, 131, (-0.042072058, 0.06620267, 0.020407796)),
}
