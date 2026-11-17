"""Transformer decoder with dynamic sequence lengths (PyTorch TorchScript exporter)."""

from collections import Counter

import numpy as np

from _helpers import BOOL, FLOAT, INT64, _model, _rand, _tensor, _vi, helper, inf

D, H, LAYERS, FF = 32, 2, 6, 32  # model width, heads, decoder layers, feed-forward width
VOCAB, MAX_LEN = 132, 60  # target vocabulary size, positional encoding length


def _const(value):
    """Unnamed tensor from a scalar or a flat list: float32 for floats, int64 for ints."""
    vals = value if isinstance(value, list) else [value]
    dtype = FLOAT if isinstance(vals[0], float) else INT64
    return _tensor("", dtype, [len(vals)] if isinstance(value, list) else [], vals)


def make():
    """Embedded src as memory, six post-norm decoder layers with cosine attention, token classifier."""
    rng = np.random.RandomState(7)
    nodes, weights, count = [], [], Counter()
    zeros, ones = "decoder_layers.0.self_attn.W_q.bias", "decoder_layers.0.norm1.weight"

    def op(scope, op_type, inputs, out=None, **attrs):
        """Adds a node named <scope>/<Op>, then <Op>_1, <Op>_2 and so on within the scope, as the exporter does."""
        k = count[scope, op_type]
        count[scope, op_type] += 1
        name = f"{scope}/{op_type}" + (f"_{k}" if k else "")
        out = out or name + "_output_0"
        nodes.append(helper.make_node(op_type, inputs, [out], name=name, **attrs))
        return out

    def unscoped(op_type, inputs, out, **attrs):
        """Adds a node the exporter left without a scope; it named those <Op>_<450 + position in the graph>."""
        nodes.append(helper.make_node(op_type, inputs, [out], name=f"{op_type}_{450 + len(nodes)}", **attrs))
        return out

    def const(scope, value):
        return op(scope, "Constant", [], value=_const(value))

    def dim(scope, x, i):
        """Size of x along axis i, as a scalar."""
        return op(scope, "Gather", [op(scope, "Shape", [x]), const(scope, i)], axis=0)

    def unsqueeze0(scope, x, uid):
        """Scalar to 1-D. The axes constant is unscoped and its output carries the exporter's value id."""
        return op(scope, "Unsqueeze", [x, unscoped("Constant", [], f"onnx::Unsqueeze_{uid}", value=_const([0]))])

    def linear(scope, x, wid, n_in, n_out, bias=None, out=None):
        """Linear layer as exported: MatMul with weight onnx::MatMul_<wid>, then Add of the bias."""
        weights.append(_rand(rng, f"onnx::MatMul_{wid}", [n_in, n_out]))
        y = op(scope, "MatMul", [x, f"onnx::MatMul_{wid}"])
        return op(scope, "Add", [bias or scope[1:].replace("/", ".") + ".bias", y], out)

    def normalize(scope, x):
        """x / max(norm(x), 1e-12) along the last axis."""
        norm = op(scope, "ReduceL2", [x], axes=[-1], keepdims=1)
        norm = op(scope, "Clip", [norm, const(scope, 1e-12), ""])
        return op(scope, "Div", [x, op(scope, "Expand", [norm, op(scope, "Shape", [x])])])

    def attention(scope, query, memory, mask, wid, uid, merge_uid):
        """Multi-head attention on L2-normalised queries and keys, with its output projection."""
        # wid is the id of the query weight, uid and merge_uid are the first Unsqueeze ids of the head split
        # and the head merge. The other ids of the block sit at fixed offsets from them.
        heads = []
        for j, (p, x) in enumerate(zip("qkv", (query, memory, memory))):
            x = linear(f"{scope}/W_{p}", x, wid + 3 * j, D, D)
            # view(batch, length, H, D // H).transpose(1, 2)
            shape = [dim(scope, x, 0), dim(scope, x, 1)]
            shape = [unsqueeze0(scope, n, uid + 20 * j + 2 * k) for k, n in enumerate(shape)]
            shape += [const(scope, [H]), const(scope, [D // H])]
            x = op(scope, "Reshape", [x, op(scope, "Concat", shape, axis=0)], allowzero=0)
            heads.append(op(scope, "Transpose", [x], perm=[0, 2, 1, 3]))

        # cosine similarity of queries and keys, divided by 0.5
        q, k = [op(scope, "Cast", [x], to=FLOAT) for x in heads[:2]]
        q, k = normalize(scope, q), normalize(scope, k)
        scores = op(scope, "MatMul", [q, op(scope, "Transpose", [k], perm=[0, 1, 3, 2])])
        scores = op(scope, "Div", [scores, const(scope, 0.5)])

        # mask.expand(-1, heads, -1, -1); the graph replaces the -1 entries of the size by 1
        n = dim(scope, scores, 1)
        size = [const(scope, [-1]), op(scope, "Unsqueeze", [n, const(scope, [0])])]
        size += [const(scope, [-1]), const(scope, [-1])]
        size = op(scope, "Reshape", [op(scope, "Concat", size, axis=0), const(scope, [-1])], allowzero=0)
        one = op(scope, "ConstantOfShape", [op(scope, "Shape", [size])], value=_const([1]))
        free = op(scope, "Equal", [size, op(scope, "Mul", [one, const(scope, -1)])])
        mask = op(scope, "Expand", [mask, op(scope, "Where", [free, one, size])])
        masked = op(scope, "Cast", [op(scope, "Not", [mask])], to=BOOL)
        scores = op(scope, "Where", [masked, const(scope, -inf), scores])
        scores = op(scope, "Sub", [scores, op(scope, "ReduceMax", [scores], axes=[-1], keepdims=1)])
        x = op(scope, "MatMul", [op(scope, "Softmax", [scores], axis=-1), op(scope, "Cast", [heads[2]], to=FLOAT)])
        x = op(scope, "Cast", [x], to=FLOAT)

        # transpose(1, 2).view(batch, length, D)
        batch, n_heads, length, width = [dim(scope, x, i) for i in range(4)]
        x = op(scope, "Transpose", [x], perm=[0, 2, 1, 3])
        shape = [batch, length, op(scope, "Mul", [n_heads, width])]
        shape = [unsqueeze0(scope, n, merge_uid + 2 * k) for k, n in enumerate(shape)]
        x = op(scope, "Reshape", [x, op(scope, "Concat", shape, axis=0)], allowzero=0)
        return linear(f"{scope}/W_o", x, wid + 14, D, D)

    def decoder_layer(i, x, memory, tgt_mask, src_mask):
        """Self attention, cross attention and feed-forward, each followed by a residual Add and LayerNorm."""
        name = f"decoder_layers.{i}"
        scope = "/" + name
        # From one layer to the next the weight ids advance by 32 and the Unsqueeze ids by 285. Only the
        # head split of the first self attention sits one id lower.
        wid, uid = 2055 + 32 * i, 345 + 285 * i

        def add_norm(x, y, k):
            y = op(scope, "Add", [x, y])
            norm = [f"{name}.norm{k}.weight", f"{name}.norm{k}.bias"]
            return op(f"{scope}/norm{k}", "LayerNormalization", [y] + norm, axis=-1, epsilon=1e-5)

        x = add_norm(x, attention(f"{scope}/self_attn", x, x, tgt_mask, wid, uid - (i == 0), uid + 116), 1)
        x = add_norm(x, attention(f"{scope}/cross_attn", x, memory, src_mask, wid + 15, uid + 138, uid + 254), 2)
        y = linear(f"{scope}/feed_forward/fc1", x, wid + 30, D, FF)
        y = op(f"{scope}/feed_forward/relu", "Relu", [y])
        return add_norm(x, linear(f"{scope}/feed_forward/fc2", y, wid + 31, FF, D), 3)

    inits = [
        _rand(rng, "encoder_embedding.0.bias", [D]),
        _rand(rng, "encoder_embedding.3.bias", [D]),
        _rand(rng, "decoder_embedding.weight", [VOCAB, D]),
        _rand(rng, "tgt_positional_encoding.pe", [1, MAX_LEN, D]),
        _tensor(zeros, FLOAT, [D], [0.0] * D),
        _rand(rng, "decoder_layers.0.feed_forward.fc1.bias", [FF]),
        _rand(rng, "decoder_layers.0.feed_forward.fc2.bias", [D]),
        _tensor(ones, FLOAT, [D], [1.0] * D),
    ]
    for i in range(1, LAYERS):
        inits.append(_rand(rng, f"decoder_layers.{i}.feed_forward.fc1.bias", [FF]))
        inits.append(_rand(rng, f"decoder_layers.{i}.feed_forward.fc2.bias", [D]))
    inits.append(_rand(rng, "fc.bias", [VOCAB]))

    # The exporter keeps one all-zero bias and one all-one norm weight and aliases every other one to them.
    aliases = [f"norm{k}.{p}" for k in (3, 2, 1) for p in ("bias", "weight")]
    aliases += [f"{a}_attn.W_{p}.bias" for a in ("cross", "self") for p in "ovkq"]
    for i in reversed(range(LAYERS)):
        for alias in aliases:
            alias = f"decoder_layers.{i}.{alias}"
            if alias not in (zeros, ones):
                unscoped("Identity", [ones if alias.endswith("weight") else zeros], alias)

    # The numbers 275, 299, 301, 2050 and 2051 below are ids from the exporter's names, e.g. onnx::Unsqueeze_299.

    # padding masks: source rows that sum to zero, target tokens equal to 0
    axis = unscoped("Constant", [], "onnx::ReduceSum_275", value=_const([2]))
    keep = op("", "Not", [op("", "Equal", [op("", "ReduceSum", ["src", axis], keepdims=0), const("", 0.0)])])
    diagonal = const("", 1)  # diagonal for Trilu below; the exporter put the Constant here
    keep = op("", "Unsqueeze", [keep, const("", [1])])
    src_mask = op("", "Unsqueeze", [keep, const("", [2])])
    keep = op("", "Not", [op("", "Equal", ["tgt", const("", 0)])])
    keep = op("", "Unsqueeze", [keep, const("", [1])])
    tgt_mask = op("", "Unsqueeze", [keep, const("", [3])])

    # causal mask: ones minus the strict upper triangle of a (1, seq, seq) matrix of ones
    seq = dim("", "tgt", 1)
    size = [const("", [1]), unsqueeze0("", seq, 299), unsqueeze0("", seq, 301)]
    full = op("", "ConstantOfShape", [op("", "Concat", size, axis=0)], value=_const([1.0]))
    upper = op("", "Trilu", [full, diagonal], upper=1)
    causal = op("", "Sub", [const("", 1.0), upper])
    causal = op("", "Cast", [op("", "Cast", [causal], to=BOOL)], to=BOOL)
    tgt_mask = op("", "And", [tgt_mask, causal])

    # encoder: only a two-layer embedding of src
    scope = "/encoder_embedding/encoder_embedding"
    x = linear(f"{scope}.0", "src", 2050, 3, D, bias="encoder_embedding.0.bias")
    x = op(f"{scope}.1", "LeakyRelu", [x], alpha=0.01)
    x = linear(f"{scope}.3", x, 2051, D, D, bias="encoder_embedding.3.bias")
    memory = op(f"{scope}.4", "LeakyRelu", [x], alpha=0.01)

    # decoder: token embedding plus the first seq rows of the positional encoding
    x = op("/decoder_embedding", "Gather", ["decoder_embedding.weight", "tgt"])
    scope = "/tgt_positional_encoding"
    seq = dim(scope, x, 1)
    axis, start = const(scope, [1]), const(scope, [0])
    end = op(scope, "Unsqueeze", [seq, const(scope, [0])])
    pe = op(scope, "Slice", ["tgt_positional_encoding.pe", start, end, axis, const(scope, [1])])
    x = op(scope, "Add", [x, pe])
    for i in range(LAYERS):
        x = decoder_layer(i, x, memory, tgt_mask, src_mask)
    linear("/fc", x, 2055 + 32 * LAYERS, D, VOCAB, out="logits")

    graph = helper.make_graph(
        nodes,
        "main_graph",
        inputs=[_vi("src", FLOAT, [1, "n_nodes", 3]), _vi("tgt", INT64, [1, "seq_length"])],
        outputs=[_vi("logits", FLOAT, ["Addlogits_dim_0", "seq_length", VOCAB])],
        initializer=inits + weights,
    )
    return _model(graph, opset=17, ir_version=8, producer_name="pytorch", producer_version="2.6.0")
