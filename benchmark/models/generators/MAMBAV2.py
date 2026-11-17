from itertools import count

import numpy as np

from _helpers import FLOAT, INT64, helper, _model, _rand, _tensor, _vi


def make():
    """Mamba sequence model: 4 layers of RMSNorm, Mamba mixer, RMSNorm, SwiGLU, then a two-layer head."""
    n_layers, seq = 4, 512
    d_model, d_inner, d_ff, d_out = 128, 256, 344, 256
    d_state, dt_rank, d_conv = 16, 8, 4
    rng = np.random.RandomState(7)
    nodes, inits = [], []
    # The rms_*, shape_* and silu_sig_* names are numbered across the whole graph, not per layer.
    n_rms, n_shape, n_silu = count(), count(), count()

    def node(op, inputs, outputs, **attrs):
        nodes.append(helper.make_node(op, inputs, [outputs] if isinstance(outputs, str) else outputs, **attrs))
        return outputs

    def const(name, dtype, dims, vals):
        inits.append(_tensor(name, dtype, dims, vals))
        return name

    def weight(name, *dims):
        inits.append(_rand(rng, name, dims))
        return name

    def reshape(x, dims, out):
        return node("Reshape", [x, const(f"shape_{next(n_shape)}", INT64, [len(dims)], dims)], out)

    def transpose(x, out):
        return node("Transpose", [x], out, perm=[0, 2, 1])

    def gemm(x, w, b, out):
        # Without a bias the original keeps an empty third input and sets beta to 0.
        return node("Gemm", [x, w, b], out, alpha=1.0, beta=1.0 if b else 0.0, transB=1)

    def silu(x, out):
        return node("Mul", [x, node("Sigmoid", [x], f"silu_sig_{next(n_silu)}")], out)

    def rms_norm(x, w, out):
        k = next(n_rms)
        const(w, FLOAT, [d_model], [1.0] * d_model)
        h = node("Pow", [x, const(f"rms_exp2_{k}", FLOAT, [], [2.0])], f"rms_sq_{k}")
        eps = const(f"rms_eps_{k}", FLOAT, [], [1e-6])
        h = node("ReduceMean", [h, const(f"rms_axes_{k}", INT64, [1], [-1])], f"rms_mn_{k}", keepdims=1)
        h = node("Sqrt", [node("Add", [h, eps], f"rms_ad_{k}")], f"rms_rt_{k}")
        return node("Mul", [node("Div", [x, h], f"rms_dv_{k}"), w], out)

    x = "features"
    for i in range(n_layers):
        blk = f"blocks.{i}."

        # Mamba mixer: project up and split into the conv branch and the gate z.
        h = rms_norm(x, f"l{i}_rn1w", f"l{i}_n1_0")
        h = reshape(h, [seq, d_model], f"mb_rsh_in_{i}")
        h = gemm(h, weight(blk + "ip_w", 2 * d_inner, d_model), "", f"mb_xz_flat_{i}")
        h = reshape(h, [1, seq, 2 * d_inner], f"mb_xz_{i}")
        h, z = node("Split", [h, const(f"mb_split_xz_{i}", INT64, [2], [d_inner] * 2)],
                    [f"mb_x_{i}", f"mb_z_{i}"], axis=-1)

        # Causal depthwise conv along the sequence, then SiLU.
        h = transpose(h, f"mb_xt_{i}")
        h = node("Conv", [h, weight(blk + "conv_w", d_inner, 1, d_conv), weight(blk + "conv_b", d_inner)],
                 f"mb_xconv_{i}", group=d_inner, pads=[d_conv - 1, 0])
        u = silu(transpose(h, f"mb_xconv_t_{i}"), f"mb_xact_{i}")

        # Per-token dt, B and C inputs of the scan.
        h = reshape(u, [seq, d_inner], f"mb_xaf_{i}")
        h = gemm(h, weight(blk + "xp_w", dt_rank + 2 * d_state, d_inner), "", f"mb_dbc_{i}")
        h = reshape(h, [1, seq, dt_rank + 2 * d_state], f"mb_dbc3_{i}")
        dt, B, C = node("Split", [h, const(f"mb_split_dbc_{i}", INT64, [3], [dt_rank, d_state, d_state])],
                        [f"mb_dt_raw_{i}", f"mb_Bssm_{i}", f"mb_Cssm_{i}"], axis=-1)
        dt = reshape(dt, [seq, dt_rank], f"mb_dtfl_{i}")
        dt = gemm(dt, weight(blk + "dp_w", d_inner, dt_rank), weight(blk + "dp_b", d_inner), f"mb_dtpf_{i}")
        dt = reshape(dt, [1, seq, d_inner], f"mb_dt3d_{i}")
        # softplus as log(exp(dt) + 1)
        dt = node("Exp", [dt], f"sp_exp_{i}")
        dt = node("Add", [dt, const(f"sp_one_{i}", FLOAT, [], [1.0])], f"sp_add_{i}")
        dt = node("Log", [dt], f"mb_dtsp_{i}")

        # MambaScan, gated by SiLU(z), projected back to the model width.
        dt = transpose(dt, f"mb_dt_bel_{i}")
        B = transpose(B, f"mb_Bbnl_{i}")
        C = transpose(C, f"mb_Cbnl_{i}")
        u = transpose(u, f"mb_u_{i}")
        # Not random: the original A is -(1..d_state) per channel up to float rounding, the same in every layer.
        A = const(blk + "mamba_A", FLOAT, [d_inner, d_state], [-float(s) for s in range(1, d_state + 1)] * d_inner)
        D = const(blk + "mamba_D", FLOAT, [d_inner], [1.0] * d_inner)
        y = node("MambaScan", [u, dt, A, B, C, D], f"mb_scany_{i}", domain="com.sofie")
        z = silu(z, f"mb_zact_{i}")
        h = node("Mul", [transpose(y, f"mb_scany_t_{i}"), z], f"mb_gated_{i}")
        h = reshape(h, [seq, d_inner], f"mb_gf_{i}")
        h = gemm(h, weight(blk + "op_w", d_model, d_inner), "", f"mb_outf_{i}")
        x = node("Add", [x, reshape(h, [1, seq, d_model], f"l{i}_mb_0")], f"l{i}_r1_0")

        # SwiGLU feed-forward. Its weights come before its shapes in the initializer list.
        h = rms_norm(x, f"l{i}_rn2w", f"l{i}_n2_0")
        wg = weight(f"l{i}_sg_wg", d_ff, d_model)
        wu = weight(f"l{i}_sg_wu", d_ff, d_model)
        wd = weight(f"l{i}_sg_wd", d_model, d_ff)
        h = reshape(h, [seq, d_model], f"l{i}_n2f_0")
        gate = gemm(h, wg, "", f"swiglu_gp_{i}")
        up = gemm(h, wu, "", f"swiglu_up_{i}")
        h = node("Mul", [silu(gate, f"swiglu_ga_{i}"), up], f"swiglu_gated_{i}")
        h = gemm(h, wd, "", f"l{i}_ffo_0")
        x = node("Add", [x, reshape(h, [1, seq, d_model], f"l{i}_ff_0")], f"l{i}_r2_0")

    # Head: final norm, Linear, SiLU, Linear.
    h = rms_norm(reshape(x, [seq, d_model], "cur_flat_0"), "fn_norm_w", "fn_normed_0")
    h = gemm(h, weight("head0_w", d_model, d_model), const("head0_b", FLOAT, [d_model], [0.0] * d_model), "fn_h0_0")
    h = silu(h, "fn_h0a_0")
    h = gemm(h, weight("head2_w", d_out, d_model), const("head2_b", FLOAT, [d_out], [0.0] * d_out), "logits_flat_0")
    reshape(h, [1, seq, d_out], "logits")

    graph = helper.make_graph(nodes, "MambaModel", [_vi("features", FLOAT, [1, seq, d_model])],
                              [_vi("logits", FLOAT, [1, seq, d_out])], initializer=inits)
    model = _model(graph, 18, 8)
    model.opset_import.append(helper.make_opsetid("com.sofie", 1))  # MambaScan lives in this domain
    return model
