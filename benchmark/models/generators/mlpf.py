"""MLPF particle-flow transformer, exported with fused or expanded attention.

The graphs follow the torch export node for node, so the helpers below also
reproduce the names and running numbers the exporter left behind.
"""

import numpy as np

from _helpers import BOOL, DOUBLE, FLOAT, INT64, TensorProto, _model, _rand, _tensor, _vi, helper

WIDTH = 256  # embedding size
HEADS = 16  # attention heads
HEAD_SIZE = WIDTH // HEADS
ELEM_TYPES = (1, 4, 5, 6, 8, 9, 10, 11)

# Single-element biases of the per-type pt and energy heads, kept at their original values.
PT_BIAS = (0.0342467576, 0.0460062623, -0.0161146093, -0.00285273092,
           0.0673165247, -0.053600315, -0.263388842, 0.325131595)
ENERGY_BIAS = (-0.00744379358, -0.0217400566, -0.00594738731, -0.00497830613,
               -0.0150533812, -0.00208132155, -0.007161038, -0.0045911083)

# Numbers in the names of the fused (True) and the expanded export: what Constant_<n> adds to
# the node position (in both exports the number of Constant nodes in the graph), id of the
# first MatMul weight, id of the last Unsqueeze axes.
EXPORT_IDS = {True: (123, 1045, 944), False: (327, 1647, 1546)}

# Expanded attention only: value id of the first Unsqueeze axes of each block, and the
# steps from it to the other fifteen. The first ids are 130 apart, plus 1 after conv_id.0
# and plus 2 into the third layer of a stack, which has an extra Cast and Mul.
AXES_FIRST_ID = {"conv_id.0": 532, "conv_id.1": 663, "conv_id.2": 795,
                 "conv_reg.0": 925, "conv_reg.1": 1055, "conv_reg.2": 1187}
AXES_ID_STEPS = (0, 2, 9, 11, 18, 20, 31, 33, 38, 40, 45, 47, 76, 80, 87, 89)


class _Graph:
    """Node and initializer lists, named the way the torch exporter names them."""

    def __init__(self, const_offset, weight_id):
        self.nodes, self.params, self.weights, self.value_info = [], [], [], []
        self.rng = np.random.RandomState(7)
        self.used = {}  # how often each scope/Op name was taken
        self.const_offset = const_offset
        # Running id of the onnx::MatMul_<id> weights. In the exports the ids also skip
        # four at each slice() and one at each fixed reshape size.
        self.weight_id = weight_id
        self.sum_axes = None

    def node(self, scope, op, inputs, out=None, **attrs):
        """Adds scope/Op (then scope/Op_1, ...) and returns its output name."""
        name = f"{scope}/{op}"
        k = self.used[name] = self.used.get(name, -1) + 1
        if k:
            name += f"_{k}"
        out = out or name + "_output_0"
        self.nodes.append(helper.make_node(op, inputs, [out], name=name, **attrs))
        return out

    def const(self, scope, val, dtype=INT64):
        """Constant node: a 1-D tensor if val is a list, else a scalar."""
        dims, vals = ([len(val)], val) if isinstance(val, list) else ([], [val])
        return self.node(scope, "Constant", [], value=_tensor("", dtype, dims, vals))

    def rand_const(self, scope, *dims):
        return self.node(scope, "Constant", [], value=_rand(self.rng, "", dims))

    def axes(self, op, value_id, axis):
        """Axes constant without a scope: named Constant_<position + offset>, output onnx::<op>_<value id>."""
        out = f"onnx::{op}_{value_id}"
        name = f"Constant_{len(self.nodes) + self.const_offset}"
        self.nodes.append(helper.make_node("Constant", [], [out], name=name, value=_tensor("", INT64, [1], [axis])))
        return out

    def slice(self, scope, x, start, end):
        """x[:, :, start:end], with constants for axis, start, end and step."""
        axis, start, end, step = (self.const(scope, [v]) for v in (2, start, end, 1))
        self.weight_id += 4
        return self.node(scope, "Slice", [x, start, end, axis, step])

    def sum_last(self, scope, x, keepdims):
        """ReduceSum over the last axis. The axes constant, value id 424 in both exports, is made once."""
        self.sum_axes = self.sum_axes or self.axes("ReduceSum", 424, -1)
        return self.node(scope, "ReduceSum", [x, self.sum_axes], keepdims=keepdims)

    def param(self, name, size, vals=None):
        """Named torch parameter: random, or the given values."""
        self.params.append(_rand(self.rng, name, [size]) if vals is None else _tensor(name, FLOAT, [size], vals))
        return name

    def linear(self, scope, bias, x, n_in, n_out, out=None):
        w = f"onnx::MatMul_{self.weight_id}"
        self.weight_id += 1
        self.weights.append(_rand(self.rng, w, [n_in, n_out]))
        return self.node(scope, "Add", [bias, self.node(scope, "MatMul", [x, w])], out)

    def layer_norm(self, scope, param, x):
        w, b = self.param(param + ".weight", WIDTH), self.param(param + ".bias", WIDTH)
        return self.node(scope, "LayerNormalization", [x, w, b], axis=-1, epsilon=1e-5)


def _ffn(g, scope, param, x, n_in, n_out, out=None, last_bias=None):
    """Linear, Relu, LayerNorm, Linear. scope and param prefix the node and parameter names."""
    x = g.linear(f"{scope}.0", g.param(f"{param}.0.bias", WIDTH), x, n_in, WIDTH)
    x = g.node(f"{scope}.1", "Relu", [x])
    x = g.layer_norm(f"{scope}.2", f"{param}.2", x)
    return g.linear(f"{scope}.4", g.param(f"{param}.4.bias", n_out, last_bias), x, WIDTH, n_out, out)


def _own_type(g, scope, per_type, onehot, keepdims):
    """Keeps, for each element, the per-type output of its own type."""
    x = g.node(scope, "Mul", [per_type, g.node(scope, "Cast", [onehot], to=FLOAT)])
    return g.sum_last(scope, x, keepdims)


def _attention(g, mha, block, q, kv, fused):
    """Multi-head scaled dot-product attention: one SDPA call if fused, else primitive ops."""
    if not fused:
        batch, length = (g.node(mha, "Gather", [g.node(mha, "Shape", [q]), g.const(mha, i)], axis=0) for i in (0, 1))
    q, k, v = (g.node(mha, "Add", [g.node(mha, "MatMul", [x, g.rand_const(mha, WIDTH, WIDTH)]),
                                   g.rand_const(mha, WIDTH)]) for x in (q, kv, kv))
    if fused:
        out = g.node(mha, "SDPA", [q, k, v], domain="onnx-script", num_heads=HEADS)
        g.value_info.append(_vi(out, FLOAT, [f"SDPA{out}_dim_0", f"SDPA{out}_dim_1", WIDTH]))
        return out

    ids = [AXES_FIRST_ID[block] + step for step in AXES_ID_STEPS]
    if block == "conv_id.0":  # the first block of the model ends one id higher
        ids[-2:] = [i + 1 for i in ids[-2:]]
    ids = iter(ids)

    def shape(*dims):
        """Shape tensor from fixed sizes (int) and run-time sizes (tensor name)."""
        parts = []
        for d in dims:
            if isinstance(d, int):
                parts.append(g.const(mha, [d]))
                g.weight_id += 1
            else:
                parts.append(g.node(mha, "Unsqueeze", [d, g.axes("Unsqueeze", next(ids), 0)]))
        return g.node(mha, "Concat", parts, axis=0)

    def reshape(x, shape):
        return g.node(mha, "Reshape", [x, shape], allowzero=0)

    def swap(x):
        return g.node(mha, "Transpose", [x], perm=[0, 2, 1, 3])

    # (batch, length, WIDTH) to (batch * HEADS, length, HEAD_SIZE). The export builds the three
    # shapes of a kind before it uses the first; another order changes the numbered node names.
    split = [shape(batch, length, HEADS, HEAD_SIZE) for _ in range(3)]
    q = swap(reshape(q, split[0]))
    batch_heads = g.node(mha, "Mul", [batch, g.const(mha, HEADS)])
    merged = [shape(batch_heads, length, HEAD_SIZE) for _ in range(3)]
    q = reshape(q, merged[0])
    k = reshape(swap(reshape(k, split[1])), merged[1])
    v = reshape(swap(reshape(v, split[2])), merged[2])

    # q and k are each scaled by sqrt(1 / sqrt(head size))
    size = g.node(mha, "Slice", [g.node(mha, "Shape", [q]), g.const(mha, [-1]), g.const(mha, [2**63 - 1])])
    scale = g.node(mha, "Sqrt", [g.node(mha, "Cast", [size], to=FLOAT)])
    scale = g.node(mha, "Cast", [g.node(mha, "Div", [g.const(mha, [1.0], FLOAT), scale])], to=FLOAT)
    k = g.node(mha, "Transpose", [k], perm=[0, 2, 1])
    q = g.node(mha, "Mul", [q, g.node(mha, "Sqrt", [scale])])
    k = g.node(mha, "Mul", [k, g.node(mha, "Sqrt", [scale])])
    out = g.node(mha, "MatMul", [g.node(mha, "Softmax", [g.node(mha, "MatMul", [q, k])], axis=-1), v])

    out = swap(reshape(out, shape(batch, HEADS, length, HEAD_SIZE)))
    return reshape(out, shape(batch, length, WIDTH))


def _layer(g, block, x, mask, query, fused):
    """Pre-norm transformer layer: attention and a two-layer ffn, each with a residual, then masking."""
    scope = "/" + block
    out_bias = g.param(f"{block}.mha.out_proj.bias", WIDTH)  # listed before the norms in the original
    h = g.layer_norm(f"{scope}/norm0", f"{block}.norm0", x)
    # the last layer of a stack uses the masked input embedding as query
    q = g.node(scope, "Mul", [query, g.node(scope, "Cast", [mask], to=FLOAT)]) if query else h
    h = _attention(g, f"{scope}/mha", block, q, h, fused)
    x = g.node(scope, "Add", [x, g.linear(f"{scope}/mha/out_proj", out_bias, h, WIDTH, WIDTH)])
    h = g.layer_norm(f"{scope}/norm1", f"{block}.norm1", x)
    for i in (0, 2):
        h = g.linear(f"{scope}/seq/seq.{i}", g.param(f"{block}.seq.{i}.bias", WIDTH), h, WIDTH, WIDTH)
        h = g.node(f"{scope}/seq/seq.{i + 1}", "Relu", [h])
    x = g.node(scope, "Add", [x, h])
    return g.node(scope, "Mul", [x, g.node(scope, "Cast", [mask], to=FLOAT)])


def _scaled_feature(g, name, feature, x):
    """feature * a + b, with a and b the two outputs of an ffn."""
    scope = f"/nn_{name}"
    ab = _ffn(g, f"{scope}/nn/nn", f"nn_{name}.nn", x, WIDTH, 2)
    scaled = g.node(scope, "Mul", [feature, g.slice(scope, ab, 0, 1)])
    return g.node(scope, "Add", [scaled, g.slice(scope, ab, 1, 2)])


def _sdpa_function():
    """Model-local function behind a fused attention: MultiHeadAttention in half precision."""
    nodes = [helper.make_node("Cast", [x], [f"{x}_{i}"], name=f"n{i}", to=TensorProto.FLOAT16)
             for i, x in enumerate(("query", "key", "value"))]
    nodes.append(helper.make_node("MultiHeadAttention", ["query_0", "key_1", "value_2"], ["output", "_", "__3"],
                                  name="n3", domain="com.microsoft", num_heads=HEADS))
    nodes.append(helper.make_node("Cast", ["output"], ["output_4"], name="n4", to=FLOAT))
    opsets = [helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)]
    return helper.make_function("onnx-script", "SDPA", ["query", "key", "value"], ["output_4"], nodes,
                                opset_imports=opsets)


def make_mlpf(fused):
    """MLPF: per-type input ffns, a 3-layer transformer stack each for id and regression, ffn heads."""
    const_offset, weight_id, last_axes_id = EXPORT_IDS[fused]
    g = _Graph(const_offset, weight_id)
    feats = "Xfeat_normed"
    mask = g.node("", "Cast", ["mask"], to=BOOL)

    def embed(name):
        """One ffn per element type, stacked on a new last axis."""
        outs = [_ffn(g, f"/{name}.{i}/{name}.{i}", f"{name}.{i}", feats, 55, WIDTH) for i in range(len(ELEM_TYPES))]
        return g.node("", "Concat", [g.node("", "Unsqueeze", [o, g.const("", [-1])]) for o in outs], axis=-1)

    def encode(name, emb):
        x = emb
        for i in range(3):
            x = _layer(g, f"{name}.{i}", x, mask3, emb if i == 2 else None, fused)
        return x

    def per_type_head(name, biases):
        """One single-output ffn per element type."""
        outs = [_ffn(g, f"/{name}/nn.{i}/nn.{i}", f"{name}.nn.{i}", reg, WIDTH, 1, last_bias=[b])
                for i, b in enumerate(biases)]
        return _own_type(g, f"/{name}", g.node(f"/{name}", "Concat", outs, axis=-1), onehot, 1)

    # input embeddings; feature 0 is the element type
    emb_id = embed("nn0_id")
    elem_type = g.slice("", feats, 0, 1)
    is_type = [g.node("", "Equal", [elem_type, g.const("", float(t), FLOAT)]) for t in ELEM_TYPES]
    onehot = g.node("", "Concat", is_type, axis=-1)
    onehot4 = g.node("", "Unsqueeze", [onehot, g.const("", [-2])])
    emb_id = _own_type(g, "", emb_id, onehot4, 0)
    emb_reg = _own_type(g, "", embed("nn0_reg"), onehot4, 0)

    mask3 = g.node("/conv_id.0", "Unsqueeze", [mask, g.const("/conv_id.0", [-1])])
    enc_id, enc_reg = encode("conv_id", emb_id), encode("conv_reg", emb_reg)

    # classification heads
    cls = g.layer_norm("/final_norm_id", "final_norm_id", g.node("", "Concat", [enc_id], axis=-1))
    for name, n_out, out in (("nn_binary_particle", 2, "bid"), ("nn_pid", 9, "id"), ("nn_pu", 2, "pu")):
        _ffn(g, f"/{name}/{name}", name, cls, WIDTH, n_out, out)

    # momentum heads: pt per type, then eta, sin phi, cos phi as corrections to input features 2, 3, 4
    reg = g.layer_norm("/final_norm_reg", "final_norm_reg", g.node("", "Concat", [enc_reg], axis=-1))
    pt = per_type_head("nn_pt", PT_BIAS)
    eta = _scaled_feature(g, "eta", g.slice("", feats, 2, 3), reg)
    sin_phi = _scaled_feature(g, "sin_phi", g.slice("", feats, 3, 4), reg)
    cos_phi = _scaled_feature(g, "cos_phi", g.slice("", feats, 4, 5), reg)

    # energy from pt and eta, sqrt(pt^2 + (pt sinh eta)^2), relative to input feature 5
    pt_abs = g.node("", "Mul", [g.node("", "Exp", [pt]), g.slice("", feats, 1, 2)])
    two_sinh = g.node("", "Sub", [g.node("", "Exp", [eta]), g.node("", "Exp", [g.node("", "Neg", [eta])])])
    pz = g.node("", "Div", [g.node("", "Mul", [pt_abs, two_sinh]), g.const("", 2.0, FLOAT)])
    pt2 = g.node("", "Pow", [pt_abs, g.const("", 2.0, FLOAT)])
    pz2 = g.node("", "Pow", [pz, g.const("", 2.0, FLOAT)])
    energy = g.node("", "Div", [g.node("", "Sqrt", [g.node("", "Add", [pt2, pz2])]), g.slice("", feats, 5, 6)])

    # log(energy) with the padded elements set to zero, as torch writes x[~mask] = 0
    padded = g.node("", "Not", [mask3])
    n_padded = g.node("", "ReduceSum", [g.node("", "Cast", [g.node("", "Not", [mask])], to=INT64)], keepdims=0)
    n_zeros = [g.const("", [0]), g.node("", "Unsqueeze", [n_padded, g.axes("Unsqueeze", last_axes_id, 0)])]
    zeros = g.node("", "ConstantOfShape", [g.node("", "Concat", n_zeros, axis=0)],
                   value=_tensor("", FLOAT, [1], [0.0]))
    zeros = g.node("", "Cast", [zeros], to=FLOAT)
    energy = g.node("", "Log", [energy])
    index = g.node("", "Expand", [padded, g.node("", "Shape", [energy])])
    index = g.node("", "Transpose", [g.node("", "NonZero", [index])], perm=[1, 0])
    zeros = g.node("", "Reshape", [zeros, g.const("", [-1])], allowzero=0)
    count = g.node("", "Gather", [g.node("", "Shape", [index]), g.const("", 0)], axis=0)
    axis, start = g.const("", [0]), g.const("", [0])
    g.weight_id += 1  # at this slice the ids skip one, not four
    zeros = g.node("", "Slice", [zeros, start, g.node("", "Unsqueeze", [count, g.const("", [0])]), axis])
    energy = g.node("", "ScatterND", [energy, index, zeros])

    # nan_to_num, then the per-type energy correction
    is_inf = g.node("", "Cast", [g.node("", "IsInf", [g.node("", "Cast", [energy], to=DOUBLE)])], to=BOOL)
    energy = g.node("", "Where", [is_inf, g.const("", 0.0, FLOAT), energy])
    is_nan = g.node("", "Cast", [g.node("", "IsNaN", [energy])], to=BOOL)
    energy = g.node("", "Where", [is_nan, g.const("", 0.0, FLOAT), energy])
    energy = g.node("", "Add", [energy, g.node("", "Relu", [per_type_head("nn_energy", ENERGY_BIAS)])])
    g.node("", "Concat", [pt, eta, sin_phi, cos_phi, energy], "momentum", axis=-1)

    dims = ["num_batch", "num_elements"]
    # the original lists the two final norms after all other named parameters
    params = sorted(g.params, key=lambda t: t.name.startswith("final_norm"))
    graph = helper.make_graph(
        g.nodes, "main_graph",
        [_vi(feats, FLOAT, dims + [55]), _vi("mask", FLOAT, dims)],
        [_vi(name, FLOAT, dims + [n]) for name, n in (("bid", 2), ("id", 9), ("momentum", 5), ("pu", 2))],
        initializer=params + g.weights, value_info=g.value_info)
    model = _model(graph, 20, 9, producer_name="pytorch", producer_version="2.5.1",
                   functions=[_sdpa_function()] if fused else [])
    if fused:
        model.opset_import.append(helper.make_opsetid("onnx-script", 1))
    return model


MODELS = {"mlpf_fp32_fused": lambda: make_mlpf(True), "mlpf_fp32_unfused": lambda: make_mlpf(False)}
