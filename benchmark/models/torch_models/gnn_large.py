"""
torch.nn.Module for gnn_large (GNNLarge, hidden_dim=128,
num_iters=8 — see _ticl_common.py), with weights loaded straight from the
.onnx file and verified bit-identical.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from _common import random_inputs, verify_against_onnxruntime
from _ticl_common import load_onnx_initializers

MODEL_DIR = Path(__file__).resolve().parent.parent
ONNX_PATH = MODEL_DIR / "gnn_large.onnx"

_HIDDEN_DIM = 128
_NUM_ITERS = 8


def _make_mlp(sizes, final_activation):
    layers = []
    n = len(sizes) - 1
    for i in range(n):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == n - 1
        if not is_last:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU())
        elif final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GNNLarge(nn.Module):
    def __init__(self, node_dim=12, edge_dim=6, hidden_dim=_HIDDEN_DIM, num_iters=_NUM_ITERS):
        super().__init__()
        self.num_iters = num_iters
        self.node_encoder = _make_mlp([node_dim, hidden_dim, hidden_dim, hidden_dim], final_activation=True)
        self.edge_encoder = _make_mlp([edge_dim, hidden_dim, hidden_dim, hidden_dim], final_activation=True)
        self.edge_network = nn.ModuleList(
            _make_mlp([3 * hidden_dim, hidden_dim, hidden_dim, hidden_dim], final_activation=True)
            for _ in range(num_iters)
        )
        self.node_network = nn.ModuleList(
            _make_mlp([3 * hidden_dim, hidden_dim, hidden_dim, hidden_dim], final_activation=True)
            for _ in range(num_iters - 1)
        )
        self.edge_decoder = _make_mlp([hidden_dim, hidden_dim, hidden_dim, hidden_dim], final_activation=True)
        self.edge_output_transform = _make_mlp([hidden_dim, hidden_dim, 1], final_activation=False)

    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        node_emb = self.node_encoder(x)
        edge_emb = self.edge_encoder(edge_attr)

        for i in range(self.num_iters):
            edge_input = torch.cat([edge_emb, node_emb[src], node_emb[dst]], dim=-1)
            edge_emb = self.edge_network[i](edge_input)

            if i < self.num_iters - 1:
                agg_dst = torch.zeros(num_nodes, edge_emb.shape[-1], dtype=edge_emb.dtype, device=edge_emb.device)
                agg_dst.scatter_add_(0, dst.unsqueeze(-1).expand(-1, edge_emb.shape[-1]), edge_emb)
                agg_src = torch.zeros(num_nodes, edge_emb.shape[-1], dtype=edge_emb.dtype, device=edge_emb.device)
                agg_src.scatter_add_(0, src.unsqueeze(-1).expand(-1, edge_emb.shape[-1]), edge_emb)
                node_input = torch.cat([agg_dst, agg_src, node_emb], dim=-1)
                node_emb = self.node_network[i](node_input)

        out = self.edge_output_transform(self.edge_decoder(edge_emb))
        return out.squeeze(-1)


_model = GNNLarge()

_initializers = load_onnx_initializers(ONNX_PATH)
_missing, _unexpected = _model.load_state_dict(_initializers, strict=False)
_layernorm_param_names = {
    f"{mod_name}.{param_name}"
    for mod_name, mod in _model.named_modules()
    if isinstance(mod, nn.LayerNorm)
    for param_name in ("weight", "bias")
}
_unexplained_missing = [m for m in _missing if m not in _layernorm_param_names]
if _unexplained_missing or _unexpected:
    raise RuntimeError(
        f"gnn_large weight loading mismatch: missing={_unexplained_missing}, unexpected={_unexpected}. "
        "The hand-authored architecture doesn't match gnn_large.onnx — check against the .onnx graph directly."
    )
for _name, _tensor in _initializers.items():
    _model_tensor = _model.state_dict()[_name]
    if not torch.equal(_model_tensor, _tensor.to(dtype=_model_tensor.dtype)):
        raise RuntimeError(f"gnn_large: value mismatch after loading {_name} — weights did not load correctly.")
print(
    f"  [gnn_large] hand-authored GNNLarge: weights loaded and verified bit-identical "
    f"({len(_layernorm_param_names)} LayerNorm params left at default identity init, "
    f"confirmed absent from the .onnx file's own initializers)"
)
_model.eval()

verify_against_onnxruntime(
    _model,
    ONNX_PATH,
    random_inputs([
        ("x", (37, 12), np.float32),
        ("edge_index", (2, 53), np.int64),
        ("edge_attr", (53, 6), np.float32),
    ]),
)


def build(shape_overrides):
    """shape_overrides: {"n_nodes": int, "n_edges": int}."""
    n_nodes = shape_overrides["n_nodes"]
    n_edges = shape_overrides["n_edges"]
    inputs = random_inputs([
        ("x", (n_nodes, 12), np.float32),
        ("edge_index", (2, n_edges), np.int64),
        ("edge_attr", (n_edges, 6), np.float32),
    ])
    example_inputs = tuple(torch.from_numpy(inputs[n]) for n in ("x", "edge_index", "edge_attr"))
    return _model, example_inputs


DYNAMIC_DIMS = {
    "x": {0: "n_nodes"},
    "edge_index": {1: "n_edges"},
    "edge_attr": {0: "n_edges"},
}
