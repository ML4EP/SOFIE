"""Static-shape exports of one small transformer encoder classifier (PyTorch 2.9.1)."""

import numpy as np

from _helpers import BOOL, FLOAT, INT64, _model, _rand, _tensor, _vi, helper, inf

GEMM = dict(alpha=1.0, beta=1.0, transA=0, transB=1)


def make_transformer(seq, batch):
    """Linear embedding, one pre-norm encoder layer with 8-head self attention, masked mean pooling, classifier."""
    rng = np.random.RandomState(7)
    L, B = seq, batch
    F, D, H, FF, C = 16, 64, 8, 256, 2  # input features, model width, heads, feed-forward width, classes
    d = D // H  # width of one head
    # In the exports all three LayerNormalization nodes take the norm1 weight and bias.
    norm = ["encoder_layer.norm1.weight", "encoder_layer.norm1.bias"]
    select = ["select", "select_1", "select_2"]

    # The batch 100 exports order and number the mask and the query/value reshapes differently from the
    # batch 1 exports. Only the shapes in MODELS have an original; other batch sizes get the batch 100 names.
    if B == 1:
        qv_shape, mask, mask_shape, mask_expand = "val_41", "view_10", "val_59", "val_65"
        flat, unflat = "view_13", "view_14"
    else:
        mask, mask_shape, mask_expand, qv_shape = "view_5", "val_41", "val_47", "val_64"
        flat, unflat = "view_10", "view_11"

    def ints(name, *vals):
        return _tensor(name, INT64, [len(vals)], vals)

    qv_shapes = [ints(qv_shape, B, H, L, d)]
    mask_shapes = [ints(mask_shape, B, 1, 1, L), ints(mask_expand, 1, H, 1, 1)]
    inits = [
        _rand(rng, "embedding.bias", [D]),
        _tensor("encoder_layer.self_attn.in_proj_bias", FLOAT, [3 * D], [0.0] * (3 * D)),
        _rand(rng, "encoder_layer.linear1.bias", [FF]),
        _rand(rng, "encoder_layer.linear2.bias", [D]),
        _tensor(norm[0], FLOAT, [D], [1.0] * D),
        _tensor(norm[1], FLOAT, [D], [0.0] * D),
        _rand(rng, "classifier.1.weight", [C, D]),
        _rand(rng, "classifier.1.bias", [C]),
        _rand(rng, "encoder_layer.self_attn.out_proj.weight", [D, D]),
        _rand(rng, "val_0", [F, D]),
        _tensor("zeros_like", FLOAT, [B, L], [0.0] * (B * L)),
        _tensor("val_6", FLOAT, [], [-inf]),
        _rand(rng, "val_9", [D, 3 * D]),
        ints("val_16", L, B, 3, D),
        ints("val_25", L, B * H, d),
        *(qv_shapes + mask_shapes if B == 1 else mask_shapes + qv_shapes),
        ints("val_97", B, H, d, L),
        _tensor("val_99", FLOAT, [1], [d ** -0.25]),  # scales query and key, d ** -0.5 on their product
        ints("val_109", L * B, D),
        ints("val_114", L, B, D),
        _rand(rng, "val_117", [D, FF]),
        _rand(rng, "val_119", [FF, D]),
        ints("val_124", 1),
        _tensor("val_129", FLOAT, [], [1.0]),
        _tensor("val_2", INT64, [], [0]),
        ints("val_17", 0),
        ints("val_18", -2),
        _tensor("val_19", INT64, [], [1]),
        _tensor("val_20", INT64, [], [2]),
        ints("val_88", -1),
    ]
    nodes, outs = [], []

    def op(op_type, inputs, out, shape, name=None, dtype=FLOAT, **attrs):
        """Add a node and the value_info of its output; name is passed where the export does not use node_<out>."""
        nodes.append(helper.make_node(op_type, inputs, [out], name=name or "node_" + out, **attrs))
        outs.append(_vi(out, dtype, shape))

    def linear(x, w, bias, out, shape, matmul):
        """Linear layer: node_MatMul_<matmul> with weight val_<w> into val_<w + 1>, then Add of the bias."""
        op("MatMul", [x, f"val_{w}"], f"val_{w + 1}", shape, name=f"node_MatMul_{matmul}")
        op("Add", [f"val_{w + 1}", bias], out, shape)

    def layer_norm(x, out, shape):
        op("LayerNormalization", [x] + norm, out, shape, axis=-1, epsilon=1e-5, stash_type=1)

    def mask_bias():
        op("Reshape", ["masked_fill", mask_shape], mask, [B, 1, 1, L], allowzero=1)
        op("Expand", [mask, mask_expand], "expand_1", [B, H, 1, L])

    def query_value_heads():
        op("Reshape", ["transpose_2", qv_shape], "view_7", [B, H, L, d], allowzero=1)
        op("Reshape", ["transpose_4", qv_shape], "view_9", [B, H, L, d], allowzero=1)

    linear("x_particles", 0, "embedding.bias", "linear", [B, L, D], 1)
    op("Not", ["padding_mask"], "bitwise_not", [B, L], dtype=BOOL)
    op("Where", ["bitwise_not", "val_6", "zeros_like"], "masked_fill", [B, L])

    # self attention: one packed projection, split into query, key and value
    layer_norm("linear", "layer_norm", [B, L, D])
    op("Transpose", ["layer_norm"], "transpose", [L, B, D], perm=[1, 0, 2])
    linear("transpose", 9, "encoder_layer.self_attn.in_proj_bias", "linear_1", [L, B, 3 * D], 8)
    op("Reshape", ["linear_1", "val_16"], "view", [L, B, 3, D], allowzero=1)
    op("Unsqueeze", ["view", "val_17"], "unsqueeze", [1, L, B, 3, D])
    op("Transpose", ["unsqueeze"], "transpose_1", [3, L, B, 1, D], perm=[3, 1, 2, 0, 4])
    op("Squeeze", ["transpose_1", "val_18"], "squeeze", [3, L, B, D])
    for i, index in enumerate(["val_2", "val_19", "val_20"]):
        op("Gather", ["squeeze", index], select[i], [L, B, D], axis=0)
    for i in range(3):
        op("Reshape", [select[i], "val_25"], f"view_{i + 1}", [L, B * H, d], allowzero=1)
        op("Transpose", [f"view_{i + 1}"], f"transpose_{i + 2}", [B * H, L, d], perm=[1, 0, 2])
    for block in (query_value_heads, mask_bias) if B == 1 else (mask_bias, query_value_heads):
        block()
    op("Transpose", ["transpose_3"], "val_96", [B * H, d, L], name="node_Transpose_94", perm=[0, 2, 1])
    op("Reshape", ["val_96", "val_97"], "val_98", [B, H, d, L], name="node_Reshape_96", allowzero=0)
    op("Mul", ["view_7", "val_99"], "val_100", [B, H, L, d], name="node_Mul_98")
    op("Mul", ["val_98", "val_99"], "val_102", [B, H, d, L], name="node_Mul_100")
    op("MatMul", ["val_100", "val_102"], "val_103", [B, H, L, L], name="node_MatMul_101")
    op("Add", ["val_103", "expand_1"], "val_104", [B, H, L, L], name="node_Add_102")
    op("Softmax", ["val_104"], "val_105", [B, H, L, L], name="node_Softmax_103", axis=-1)
    op("MatMul", ["val_105", "view_9"], "scaled_dot_product_attention", [B, H, L, d])
    op("Transpose", ["scaled_dot_product_attention"], "permute", [L, B, H, d], perm=[2, 0, 1, 3])
    op("Reshape", ["permute", "val_109"], flat, [L * B, D], allowzero=1)
    op("Gemm", [flat, "encoder_layer.self_attn.out_proj.weight"], "linear_2", [L * B, D], name="node_Gemm_14", **GEMM)
    op("Reshape", ["linear_2", "val_114"], unflat, [L, B, D], allowzero=1)
    op("Transpose", [unflat], "transpose_5", [B, L, D], perm=[1, 0, 2])
    op("Add", ["linear", "transpose_5"], "add", [B, L, D])

    # feed-forward
    layer_norm("add", "layer_norm_1", [B, L, D])
    linear("layer_norm_1", 117, "encoder_layer.linear1.bias", "linear_3", [B, L, FF], 114)
    op("Relu", ["linear_3"], "relu", [B, L, FF])
    linear("relu", 119, "encoder_layer.linear2.bias", "linear_4", [B, L, D], 116)
    op("Add", ["add", "linear_4"], "add_1", [B, L, D])

    # mean over the particles kept by padding_mask, then the classifier
    op("Unsqueeze", ["padding_mask", "val_88"], "unsqueeze_1", [B, L, 1], dtype=BOOL)
    op("Cast", ["unsqueeze_1"], "_to_copy", [B, L, 1], to=FLOAT)
    op("Mul", ["add_1", "_to_copy"], "mul", [B, L, D])
    op("ReduceSum", ["mul", "val_124"], "sum_1", [B, D], keepdims=0, noop_with_empty_axes=0)
    op("ReduceSum", ["_to_copy", "val_124"], "sum_2", [B, 1], keepdims=0, noop_with_empty_axes=0)
    op("Clip", ["sum_2", "val_129"], "clamp_min", [B, 1])
    op("Div", ["sum_1", "clamp_min"], "div", [B, D])
    layer_norm("div", "layer_norm_2", [B, D])
    classifier = ["classifier.1.weight", "classifier.1.bias"]
    op("Gemm", ["layer_norm_2"] + classifier, "logits", [B, C], name="node_linear_5", **GEMM)

    logits = outs.pop()
    graph = helper.make_graph(
        nodes,
        "main_graph",
        inputs=[_vi("x_particles", FLOAT, [B, L, F]), _vi("padding_mask", BOOL, [B, L])],
        outputs=[logits],
        initializer=inits,
        value_info=[_vi(t.name, t.data_type, list(t.dims)) for t in inits] + outs,
    )
    return _model(graph, opset=17, ir_version=10, producer_name="pytorch", producer_version="2.9.1")


MODELS = {
    "simple_transformer": lambda: make_transformer(30, 1),
    "simple_transformer_300": lambda: make_transformer(300, 1),
    "transformer_L1000_B1": lambda: make_transformer(1000, 1),
    "transformer_L1000_B100": lambda: make_transformer(1000, 100),
    # the original file of this name also has sequence length 1000
    "transformer_L100_B100": lambda: make_transformer(1000, 100),
    "transformer_L8000_B1": lambda: make_transformer(8000, 1),
}
