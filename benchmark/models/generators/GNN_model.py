import numpy as np

from _helpers import FLOAT, INT32, INT64, helper, _model, _rand, _tensor, _vi

# Divisors that scale the 29 node features and the 5 edge features.
NODE_SCALER = [
    255.7259979248047, 256.3272705078125, 513.258056640625, 3.135737895965576, 3.1415927410125732,
    0.9999950528144836, 0.9999852180480957, 1.000000238418579, 2086.6826171875, 107.41090393066406,
    24.52458953857422, 44.965248107910156, 36.10573959350586, 32.707763671875, 206.0, 2210.0,
    601.1112670898438, 577.4508666992188, 0.9998738765716553, 2.2861906927573727e-06,
    1.454487573937513e-06, 2.1318462586350506e-06, 1.0, 3.517772938721464e-06, 513.2760009765625,
    513.2760009765625, 0.7304964661598206, 16.90070915222168, 99.0,
]
EDGE_SCALER = [600.1822509765625, 190.99005126953125, 484.8155822753906, 2.9532217979431152, 135.07208251953125]
# Biases of the three layers with a single output.
OUT_BIAS = {"attention_direct.3": -0.24479809403419495, "attention_reverse.3": 0.11029389500617981,
            "edgenetwork.3": 0.0633968710899353}
# Output name and attributes of each activation.
ACT = {"LeakyRelu": ("leaky_relu", {"alpha": 0.01}), "Relu": ("relu", {}), "Sigmoid": ("sigmoid", {})}


def make():
    """Edge scoring GNN: node and edge MLPs, two attention weighted graph convolutions, edge MLP."""
    n_nodes, n_edges, hidden = 3370, 24126, 16
    n_msg = n_nodes + 2 * n_edges  # length of the message list built below
    rng = np.random.RandomState(7)
    nodes, inits, used = [], [], {}
    # dtype and shape of every tensor: graph inputs first, graph output last.
    info = {"x": (FLOAT, [n_nodes, 29]), "edge_features": (FLOAT, [n_edges, 5]), "edge_index": (INT64, [n_edges, 2])}

    def const(name, dtype, dims, vals):
        inits.append(_tensor(name, dtype, dims, vals))
        info[name] = (dtype, dims)

    def linear_weights(layer, n_in, n_out):
        for name, dims in ((f"{layer}.weight", [n_out, n_in]), (f"{layer}.bias", [n_out])):
            if dims == [1]:
                const(name, FLOAT, dims, [OUT_BIAS[layer]])
            else:
                inits.append(_rand(rng, name, dims))
                info[name] = (FLOAT, dims)

    def mlp_weights(name, n_in, n_out):
        linear_weights(f"{name}.0", n_in, hidden)
        linear_weights(f"{name}.3", hidden, n_out)

    def node(op, number, inputs, out, shape=None, dtype=None, **attrs):
        """Adds node_<op>_<number>; number is the one in the original, whose numbering has gaps.
        The output takes dtype and shape from the first input unless they are given."""
        k = used[out] = used.get(out, -1) + 1
        out = f"{out}_{k}" if k else out  # as in the original: cat, cat_1, cat_2, ...
        nodes.append(helper.make_node(op, inputs, [out], name=f"node_{op}_{number}", **attrs))
        info[out] = (dtype or info[inputs[0]][0], shape or info[inputs[0]][1])
        return out

    def linear(x, layer, number):
        shape = [info[x][1][0], info[f"{layer}.bias"][1][0]]
        return node("Gemm", number, [x, f"{layer}.weight", f"{layer}.bias"], "linear", shape,
                    alpha=1.0, beta=1.0, transA=0, transB=1)

    def dense(x, layer, number, act):
        out, attrs = ACT[act]
        return node(act, number + 1, [linear(x, layer, number)], out, **attrs)

    def mlp(x, name, number, last):
        """Linear, LeakyRelu, Linear, last. The original skips one node number between the two layers."""
        return dense(dense(x, f"{name}.0", number, "LeakyRelu"), f"{name}.3", number + 3, last)

    # Initializers in the order of the original: the Linear layers, then the constants.
    mlp_weights("inputnetwork", 29, hidden)
    mlp_weights("edge_inputnetwork", 5, hidden)
    mlp_weights("attention_direct", hidden, 1)
    mlp_weights("attention_reverse", hidden, 1)
    for k in range(2):
        linear_weights(f"graphconvs.{k}.conv_0", 2 * hidden, 2 * hidden)
        linear_weights(f"graphconvs.{k}.convs.0", 2 * hidden, hidden)
        linear_weights(f"graphconvs.{k}.sc", hidden, hidden)
    mlp_weights("edgenetwork", 3 * hidden + 5, 1)
    const("node_scaler", FLOAT, [29], NODE_SCALER)
    const("edge_scaler", FLOAT, [5], EDGE_SCALER)
    const("val_0", FLOAT, [], [1e-4])
    const("zeros", INT32, [n_nodes], [0] * n_nodes)
    const("unbind_axis", INT64, [1], [1])
    const("val_22", INT64, [1], [-1])
    const("ones", FLOAT, [n_nodes], [1.0] * n_nodes)
    const("zeros_1", FLOAT, [n_nodes, hidden], [0.0] * n_nodes * hidden)
    const("val_51", INT64, [2], [1, 1])
    const("val_53", INT64, [2], [1, hidden])
    const("ones_1", FLOAT, [n_nodes], [1.0] * n_nodes)
    const("zeros_2", FLOAT, [n_nodes, hidden], [0.0] * n_nodes * hidden)

    # Scale the inputs, embed edges and nodes, and get one attention weight per edge direction.
    edge = node("Add", 1, ["edge_features", "val_0"], "add")
    edge = node("Div", 2, [edge, "edge_scaler"], "div")
    x = node("Div", 3, ["x", "node_scaler"], "div")
    edge_h = mlp(edge, "edge_inputnetwork", 4, "LeakyRelu")
    att = [mlp(edge_h, f"attention_{way}", number, "Sigmoid") for way, number in (("direct", 9), ("reverse", 14))]
    att = node("Concat", 19, att, "cat", [2 * n_edges, 1], axis=0)
    h = mlp(x, "inputnetwork", 20, "LeakyRelu")

    # Message list: n_nodes entries of index 0, then every edge in both directions.
    nodes.append(helper.make_node("Split", ["edge_index"], ["unbind_split_0", "unbind_split_1"],
                                  name="node_Split_193", axis=1, num_outputs=2))
    info["unbind_split_0"] = info["unbind_split_1"] = (INT64, [n_edges, 1])
    src, dst = (node("Squeeze", 195 + i, [f"unbind_split_{i}", "unbind_axis"], f"unbind_split_{i}_squeeze", [n_edges])
                for i in range(2))
    head = node("Cast", 35, ["zeros"], "convert_element_type_default", dtype=INT64, to=INT64)
    centre = node("Concat", 36, [head, src, dst], "cat", [n_msg], axis=0)
    neighbour = node("Concat", 38, [head, dst, src], "cat", [n_msg], axis=0)

    for k in range(2):
        # Layer 1 repeats layer 0 with the node numbers 58 to 94 raised by 69 (a), 106 to 109 by 65 (b)
        # and the val_N names by 45. The k == 0 blocks exist once; layer 1 reuses their outputs.
        a, b = 69 * k, 65 * k
        if k == 0:
            centre_idx = node("Unsqueeze", 56, [centre, "val_22"], "val_23", [n_msg, 1])
        h_centre = node("GatherND", 58 + a, [h, centre_idx], f"val_{25 + 45 * k}", [n_msg, hidden], batch_dims=0)
        if k == 0:
            neighbour_idx = node("Unsqueeze", 75, [neighbour, "val_22"], "val_40", [n_msg, 1])
        h_neighbour = node("GatherND", 77 + a, [h, neighbour_idx], f"val_{42 + 45 * k}", [n_msg, hidden],
                           batch_dims=0)

        # Message: two Linear + Relu layers on the centre features joined with neighbour minus centre.
        msg = node("Sub", 79 + a, [h_neighbour, h_centre], "sub")
        msg = node("Concat", 80 + a, [h_centre, msg], "cat", [n_msg, 2 * hidden], axis=1)
        msg = dense(msg, f"graphconvs.{k}.conv_0", 85 + a, "Relu")
        msg = dense(msg, f"graphconvs.{k}.convs.0", 88 + a, "Relu")

        # Weight the messages: 1 for the leading entries, the attention for edges.
        if k == 0:
            att = node("Squeeze", 90, [att], "squeeze", [2 * n_edges])
        weight = node("Concat", 91 + a, ["ones_1" if k else "ones", att], "cat", [n_msg], axis=0)
        msg = node("Transpose", 92 + a, [msg], "transpose", [hidden, n_msg], perm=[1, 0])
        msg = node("Mul", 93 + a, [weight, msg], "mul", [hidden, n_msg])
        msg = node("Transpose", 94 + a, [msg], "transpose", [n_msg, hidden], perm=[1, 0])

        # Sum the messages per centre node, add the shortcut.
        if k == 0:
            rows = node("Unsqueeze", 101, [centre, "unbind_axis"], "unsqueeze", [n_msg, 1])
            rows = node("Expand", 103, [rows, "val_51"], "val_52")
            rows = node("Tile", 105, [rows, "val_53"], "repeat", [n_msg, hidden])
        msg = node("ScatterElements", 106 + b, [f"zeros_{k + 1}", rows, msg], "scatter_add", axis=0, reduction="add")
        h = node("Add", 108 + b, [linear(h, f"graphconvs.{k}.sc", 107 + b), msg], "add")
        h = node("Relu", 109 + b, [h], "relu")

    # Score every edge from its two end nodes, its embedding and its scaled features.
    ends = []
    for i, end in enumerate((src, dst)):
        index = node("Reshape", 176 + 4 * i, [end, "val_22"], f"val_{96 + 3 * i}", allowzero=0)
        ends.append(node("Gather", 178 + 4 * i, [h, index], "index_select", [n_edges, hidden], axis=0))
    out = node("Concat", 183, ends + [edge_h, edge], "cat", [n_edges, 3 * hidden + 5], axis=-1)
    mlp(out, "edgenetwork", 184, "Sigmoid")

    vi = [_vi(name, *info[name]) for name in info]
    graph = helper.make_graph(nodes, "main_graph", vi[:3], vi[-1:], initializer=inits, value_info=vi[3:-1])
    return _model(graph, opset=18, ir_version=10, producer_name="pytorch", producer_version="2.7.1+cu126")
