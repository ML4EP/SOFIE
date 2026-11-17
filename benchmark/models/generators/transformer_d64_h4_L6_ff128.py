"""Transformer decoder with dynamic sequence lengths (PyTorch dynamo exporter)."""

from collections import Counter

import numpy as np

from _helpers import BOOL, FLOAT, INT64, _model, _rand, _tensor, _vi, helper, inf

# The value ids and name numbers in this file are read off the original, so only LAYERS = 6 gives its names.
D, H, LAYERS, FF = 64, 4, 6, 128  # model width, heads, decoder layers, feed-forward width
VOCAB, MAX_LEN = 132, 2048  # target tokens, positional encoding length
N, S = "n_nodes", "seq_length"  # source and target length
# The original has one all-zero bias and one all-one norm weight, used in every layer.
ZEROS, ONES = "decoder_layers.0.self_attn.W_q.bias", "decoder_layers.0.norm1.weight"
# Value ids of the six shape tensors of the original, in the order make() first needs them.
SHAPES = (15, 38, 65, 88, 108, 134)
# Value ids of the ten constants that come last in the initializer list of the original.
LATE = (2, 4, 5, 769, 770, 23, 36, 37, 75, 87)


def make():
    """Point embedding as memory, six post-norm decoder layers with cosine attention, token classifier."""
    rng = np.random.RandomState(7)
    nodes, count, vals, shapes, inverted = [], Counter(), {}, {}, {}
    info = {"src": (FLOAT, [1, N, 3]), "tgt": (INT64, [1, S])}  # dtype and shape of every value

    def const(vid, value):
        """Initializer val_<vid> from a scalar or a flat list: float32 for floats, int64 for ints."""
        flat = value if isinstance(value, list) else [value]
        dtype = FLOAT if isinstance(flat[0], float) else INT64
        vals[vid] = _tensor(f"val_{vid}", dtype, [len(flat)] if isinstance(value, list) else [], flat)
        return f"val_{vid}"

    def nth(kind):
        """kind, kind_1, kind_2 and so on, the way the original numbers its value names."""
        count[kind] += 1
        return kind + (f"_{count[kind] - 1}" if count[kind] > 1 else "")

    def op(op_type, inputs, out, shape=None, dtype=None, name=None, **attrs):
        """Adds a node. Its output takes dtype and shape from the first input unless they are given."""
        nodes.append(helper.make_node(op_type, inputs, [out], name=name or "node_" + out, **attrs))
        info[out] = (dtype or info[inputs[0]][0], shape or info[inputs[0]][1])
        return out

    def val(op_type, inputs, vid, shape=None, dtype=None, **attrs):
        """Adds a node with output val_<vid>. Its name carries vid minus 2 per LayerNormalization before it."""
        name = f"node_{op_type}_{vid - 2 * count['layer_norm']}"
        return op(op_type, inputs, f"val_{vid}", shape, dtype, name, **attrs)

    # constants and sequence lengths used by the repeated blocks
    one, last, eps = const(29, [1]), const(57, [-1]), const(59, 1e-12)
    n_heads, head_dim, width = const(36, [H]), const(37, [D // H]), const(87, [D])
    half, mask_size, minus_inf = const(75, 0.5), const(81, [1, H, 1, 1]), const(83, -inf)
    size = {N: val("Shape", ["src"], 0, [1], INT64, start=1, end=2),
            S: val("Shape", ["tgt"], 1, [1], INT64, start=1, end=2)}

    def shape(*dims):
        """Shape tensor from 1-D pieces. The original has one Concat per distinct shape, see SHAPES."""
        if dims not in shapes:
            shapes[dims] = val("Concat", list(dims), SHAPES[len(shapes)], [len(dims)], INT64, axis=0)
        return shapes[dims]

    def linear(x, wid, n_out, bias=ZEROS, out=None):
        """Linear layer as exported: MatMul with weight val_<wid> into val_<wid + 1>, then Add of the bias."""
        *lead, n_in = info[x][1]
        vals[wid] = _rand(rng, f"val_{wid}", [n_in, n_out])
        y = val("MatMul", [x, f"val_{wid}"], wid + 1, lead + [n_out])
        name = nth("linear")
        return op("Add", [y, bias], out or name, name="node_" + name)

    def normalize(x):
        """x / max(norm(x), 1e-12) along the last axis, for x of shape (1, H, length, D // H)."""
        length = info[x][1][2]
        norm = op("ReduceL2", [x, last], nth("linalg_vector_norm"), [1, H, length, 1], keepdims=1,
                  noop_with_empty_axes=0)
        norm = op("Clip", [norm, eps], nth("clamp_min"))
        norm = op("Expand", [norm, shape(one, n_heads, size[length], head_dim)], nth("expand"), info[x][1])
        return op("Div", [x, norm], nth("div"))

    def attention(x, memory, mask, wid, out_wid, sub):
        """Multi-head attention on L2-normalised queries and keys, with its output projection."""
        # wid and out_wid are the value ids of the query and the output weight; the Sub output is sub_<sub>.
        M = info[memory][1][1]
        heads = []
        for j, y in enumerate((x, memory, memory)):
            length = info[y][1][1]
            y = linear(y, wid + 8 * j, D)
            # view(1, length, H, D // H).transpose(1, 2)
            split = shape(one, size[length], n_heads, head_dim)
            y = op("Reshape", [y, split], nth("view"), [1, length, H, D // H], allowzero=1)
            heads.append(op("Transpose", [y], nth("transpose"), [1, H, length, D // H], perm=[0, 2, 1, 3]))

        # cosine similarity of queries and keys, divided by 0.5
        q, k = normalize(heads[0]), normalize(heads[1])
        k = op("Transpose", [k], nth("transpose"), [1, H, D // H, M], perm=[0, 1, 3, 2])
        scores = op("MatMul", [q, k], nth("matmul"), [1, H, S, M])
        scores = op("Div", [scores, half], nth("div"))

        # masked_fill(~mask.expand(-1, H, -1, -1), -inf). The original expands and inverts each mask once,
        # in layer 0; every later block skips one expand_<k> name instead.
        expanded = nth("expand")
        if mask not in inverted:
            y = op("Expand", [mask, mask_size], expanded, [1, H] + info[mask][1][2:])
            inverted[mask] = op("Not", [y], nth("bitwise_not"))
        scores = op("Where", [inverted[mask], minus_inf, scores], nth("masked_fill"), [1, H, S, M], FLOAT)
        # softmax(scores - max(scores)). The ReduceMax nodes are named n0, n0_2, n0_3 and so on.
        name = f"n0_{count['amax'] + 1}" if count["amax"] else "n0"
        top = op("ReduceMax", [scores, last], nth("amax"), [1, H, S, 1], name=name, keepdims=1)
        scores = op("Sub", [scores, top], f"sub_{sub}")
        scores = op("Softmax", [scores], nth("softmax"), axis=-1)
        y = op("MatMul", [scores, heads[2]], nth("matmul"), [1, H, S, D // H])

        # transpose(1, 2).view(1, seq, D)
        y = op("Transpose", [y], nth("transpose"), [1, S, H, D // H], perm=[0, 2, 1, 3])
        y = op("Reshape", [y, shape(one, size[S], width)], nth("view"), [1, S, D], allowzero=1)
        return linear(y, out_wid, D)

    def add_norm(x, y, add):
        y = op("Add", [x, y], f"add_{add}")
        return op("LayerNormalization", [y, ONES, ZEROS], nth("layer_norm"), axis=-1, epsilon=1e-5, stash_type=1)

    def decoder_layer(i, x, memory, tgt_mask, src_mask):
        """Self attention, cross attention and feed-forward, each followed by a residual Add and LayerNorm."""
        # Numbers read off the original. From one layer to the next the value ids step by 122, the sub_<n>
        # names by 92 and the add_<n> names by 268. Only the query, key and value weights of the first
        # attention block of layer 0 sit 4 value ids lower.
        wid, sub, add = 35 + 122 * i, 56 + 92 * i, 176 + 268 * i
        x = add_norm(x, attention(x, x, tgt_mask, wid - 4 * (i == 0), wid + 54, sub), add)
        x = add_norm(x, attention(x, memory, src_mask, wid + 58, wid + 112, sub + 42), add + 124)
        y = linear(x, wid + 116, FF, f"decoder_layers.{i}.feed_forward.fc1.bias")
        y = op("Relu", [y], nth("relu"))
        return add_norm(x, linear(y, wid + 118, D, f"decoder_layers.{i}.feed_forward.fc2.bias"), add + 143)

    inits = [
        _rand(rng, "encoder_embedding.0.bias", [D]),
        _rand(rng, "encoder_embedding.3.bias", [D]),
        _rand(rng, "decoder_embedding.weight", [VOCAB, D]),
        _tensor(ZEROS, FLOAT, [D], [0.0] * D),
        _rand(rng, "decoder_layers.0.feed_forward.fc1.bias", [FF]),
        _rand(rng, "decoder_layers.0.feed_forward.fc2.bias", [D]),
        _tensor(ONES, FLOAT, [D], [1.0] * D),
    ]
    for i in range(1, LAYERS):
        inits.append(_rand(rng, f"decoder_layers.{i}.feed_forward.fc1.bias", [FF]))
        inits.append(_rand(rng, f"decoder_layers.{i}.feed_forward.fc2.bias", [D]))
    inits.append(_rand(rng, "fc.bias", [VOCAB]))
    inits.append(_rand(rng, "tgt_positional_encoding.pe", [1, MAX_LEN, D]))
    inits.append(_tensor("scalar_tensor_default", FLOAT, [], [0.0]))

    # padding masks: source points whose coordinates sum to 0, target tokens equal to 0
    keep = op("ReduceSum", ["src", const(2, [2])], "sum_1", [1, N], keepdims=0, noop_with_empty_axes=0)
    keep = op("Not", [val("Equal", [keep, "scalar_tensor_default"], 6, dtype=BOOL)], nth("ne"))
    src_mask = op("Unsqueeze", [keep, const(769, [1, 2])], "unsqueeze_1", [1, 1, 1, N], name="node_Unsqueeze_774")
    keep = op("Not", [val("Equal", ["tgt", const(5, 0)], 9, dtype=BOOL)], nth("ne"))
    tgt_mask = op("Unsqueeze", [keep, const(770, [1, 3])], "unsqueeze_3", [1, 1, S, 1], name="node_Unsqueeze_776")

    # causal mask: ones minus the strict upper triangle of a (1, seq, seq) matrix of ones
    full = op("Expand", [const(18, 1.0), shape(one, size[S], size[S])], "ones", [1, S, S], FLOAT)
    upper = op("Trilu", [full, const(4, 1)], "triu", upper=1)
    causal = op("Sub", ["val_18", upper], "sub_11", [1, S, S], FLOAT)
    causal = op("Cast", [causal], "_to_copy", dtype=BOOL, to=BOOL)
    tgt_mask = op("And", [tgt_mask, causal], "bitwise_and", [1, 1, S, S])

    # encoder: only a two-layer embedding of the source points
    x = linear("src", 19, D, "encoder_embedding.0.bias")
    x = op("LeakyRelu", [x], nth("leaky_relu"), alpha=0.01)
    x = linear(x, 21, D, "encoder_embedding.3.bias")
    memory = op("LeakyRelu", [x], nth("leaky_relu"), alpha=0.01)

    # decoder: token embedding plus the first seq rows of the positional encoding
    x = op("Gather", ["decoder_embedding.weight", "tgt"], "embedding", [1, S, D], FLOAT, axis=0)
    rows = [1, f"Min({MAX_LEN}, {S})", D]
    pe = op("Slice", ["tgt_positional_encoding.pe", const(23, [0]), size[S], one, one], "slice_1", rows, FLOAT)
    x = op("Add", [x, pe], "add_51")
    for i in range(LAYERS):
        x = decoder_layer(i, x, memory, tgt_mask, src_mask)
    linear(x, 35 + 122 * LAYERS, VOCAB, "fc.bias", "logits")

    # Initializer order of the original: parameters, then weights and constants by value id, then LATE.
    # Its value_info lists every initializer and every node output; its metadata_props are not reproduced.
    inits += [vals[k] for k in sorted(set(vals) - set(LATE))] + [vals[k] for k in LATE]
    vi = [_vi(t.name, t.data_type, list(t.dims)) for t in inits]
    vi += [_vi(name, *info[name]) for node in nodes for name in node.output]
    graph = helper.make_graph(
        nodes,
        "main_graph",
        inputs=[_vi(name, *info[name]) for name in ("src", "tgt")],
        outputs=vi[-1:],
        initializer=inits,
        value_info=vi[:-1],
    )
    return _model(graph, opset=18, ir_version=10, producer_name="pytorch", producer_version="2.13.0+cu130")
