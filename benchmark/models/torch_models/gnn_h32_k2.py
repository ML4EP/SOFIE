"""
Real torch.nn.Module for gnn_h32_k2 (GNN_TrackLinkingNet, hidden_dim=32,
niters=2 — see _ticl_common.py), with weights loaded straight from the
.onnx file and verified bit-identical.
"""

from pathlib import Path

import numpy as np
import torch

from _common import random_inputs, verify_against_onnxruntime
from _ticl_common import import_gnn_track_linking_net, load_onnx_initializers, load_weights_strict

MODEL_DIR = Path(__file__).resolve().parent.parent
ONNX_PATH = MODEL_DIR / "gnn_h32_k2.onnx"

_initializers = load_onnx_initializers(ONNX_PATH)

# "/graphconvs.0/Squeeze" used to squeeze all size-1 dims implicitly,
# which TensorRT's ONNX parser can't do for a dynamic shape.
_GNN_TrackLinkingNet = import_gnn_track_linking_net()
_model = load_weights_strict(
    _GNN_TrackLinkingNet(input_dim=29, hidden_dim=32, edge_feature_dim=5, edge_hidden_dim=32, niters=2),
    _initializers,
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
