"""
torch.nn.Module for punet_h32_k2_heads4_layers2 (PUNet, hidden_dim=32,
niters=2, num_heads=4, num_layers=2), with weights
loaded straight from the .onnx file and verified bit-identical.
"""

from pathlib import Path

import numpy as np
import torch

from _common import random_inputs, verify_against_onnxruntime
from _ticl_common import (
    fix_anonymous_linear_weights,
    import_punet,
    load_onnx_initializers,
    load_weights_strict,
)

MODEL_DIR = Path(__file__).resolve().parent.parent
ONNX_PATH = MODEL_DIR / "punet_h32_k2_heads4_layers2.onnx"

_initializers = load_onnx_initializers(ONNX_PATH)

# "/graphconvs.0/Squeeze" used to squeeze all size-1 dims implicitly,
# which TensorRT's ONNX parser can't do for a dynamic shape
_initializers.pop("sofie_patched_squeeze_axes", None)
_PUNet = import_punet()
_model_uninit = _PUNet(input_dim=29, hidden_dim=32, edge_feature_dim=5, edge_hidden_dim=32,
                        niters=2, num_heads=4, num_layers=2)
fix_anonymous_linear_weights(_model_uninit, _initializers, ONNX_PATH)
# The self_attn.{W_k,W_v,W_o}.bias / norm*.{weight,bias} names below are
# explicitly zero- (bias) or default- (LayerNorm weight=1/bias=0)
_IDENTITY_AT_INIT = [
    f"encoder_layers.{i}.{name}"
    for i in range(2)
    for name in (
        "self_attn.W_q.bias", "self_attn.W_k.bias", "self_attn.W_v.bias", "self_attn.W_o.bias",
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
        "final_norm.weight", "final_norm.bias",
    )
]
_model = load_weights_strict(
    _model_uninit, _initializers,
    allow_missing_prefixes=("pu_network",),
    allow_missing_exact=_IDENTITY_AT_INIT,
)
_model.eval()
print(f"  [ticl] {ONNX_PATH}: weights loaded and verified bit-identical")
verify_against_onnxruntime(
    _model,
    ONNX_PATH,
    random_inputs([
        ("node_features", (37, 29), np.float32),
        ("edge_features", (53, 5), np.float32),
        ("edge_index", (53, 2), np.int64),
    ]),
)


def build(shape_overrides):
    """shape_overrides: {"n_nodes": int, "n_edges": int}."""
    n_nodes = shape_overrides["n_nodes"]
    n_edges = shape_overrides["n_edges"]
    inputs = random_inputs([
        ("node_features", (n_nodes, 29), np.float32),
        ("edge_features", (n_edges, 5), np.float32),
        ("edge_index", (n_edges, 2), np.int64),
    ])
    example_inputs = tuple(torch.from_numpy(inputs[n]) for n in
                            ("node_features", "edge_features", "edge_index"))
    return _model, example_inputs


DYNAMIC_DIMS = {
    "node_features": {0: "n_nodes"},
    "edge_features": {0: "n_edges"},
    "edge_index": {0: "n_edges"},
}
