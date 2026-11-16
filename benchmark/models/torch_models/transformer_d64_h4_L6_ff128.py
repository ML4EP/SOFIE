"""
torch.nn.Module for transformer_d64_h4_L6_ff128 (Transformer, d_model=64,
num_heads=4, num_layers=6, d_ff=128), with weights
loaded straight from the .onnx file and verified bit-identical.
"""

from pathlib import Path

import numpy as np
import torch

from _common import random_inputs, verify_against_onnxruntime
from _ticl_common import (
    fix_anonymous_attention_weights_by_position,
    fix_anonymous_linear_weights_by_bias_topology,
    import_transformer,
    load_onnx_initializers,
    load_weights_strict,
)

MODEL_DIR = Path(__file__).resolve().parent.parent
ONNX_PATH = MODEL_DIR / "transformer_d64_h4_L6_ff128.onnx"

_initializers = load_onnx_initializers(ONNX_PATH)
_Transformer = import_transformer()
_model_uninit = _Transformer(
    tgt_vocab_size=132, d_model=64, num_heads=4, num_layers=6, d_ff=128,
    feature_count=3, max_nodes=4096, max_seq_length=2048, dropout=0.0,
)
_original_initializers = dict(_initializers)
fix_anonymous_linear_weights_by_bias_topology(_model_uninit, _initializers, ONNX_PATH)
fix_anonymous_attention_weights_by_position(
    _model_uninit, _initializers, _original_initializers, ONNX_PATH, num_decoder_layers=6
)
_IDENTITY_AT_INIT = [
    f"decoder_layers.{i}.{name}"
    for i in range(6)
    for name in (
        "self_attn.W_q.bias", "self_attn.W_k.bias", "self_attn.W_v.bias", "self_attn.W_o.bias",
        "cross_attn.W_q.bias", "cross_attn.W_k.bias", "cross_attn.W_v.bias", "cross_attn.W_o.bias",
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight", "norm3.bias",
    )
]
_model = load_weights_strict(
    _model_uninit, _initializers,
    allow_missing_prefixes=("encoder_layers", "src_positional_encoding"),
    allow_missing_exact=_IDENTITY_AT_INIT,
    ignore_unexpected=True,  # leftover anonymous graph constants unrelated to any weight — see _ticl_common.py
)
_model.eval()
verify_against_onnxruntime(
    _model,
    ONNX_PATH,
    random_inputs([
        ("src", (1, 11, 3), np.float32),
        ("tgt", (1, 13), np.int64),
    ]),
)


def build(shape_overrides):
    """shape_overrides: {"n_nodes": int (source seq len), "seq_length": int (target seq len)}."""
    n_nodes = shape_overrides["n_nodes"]
    seq_length = shape_overrides["seq_length"]
    inputs = random_inputs([
        ("src", (1, n_nodes, 3), np.float32),
        ("tgt", (1, seq_length), np.int64),
    ])
    example_inputs = tuple(torch.from_numpy(inputs[n]) for n in ("src", "tgt"))
    return _model, example_inputs


DYNAMIC_DIMS = {
    "src": {1: "n_nodes"},
    "tgt": {1: "seq_length"},
}
