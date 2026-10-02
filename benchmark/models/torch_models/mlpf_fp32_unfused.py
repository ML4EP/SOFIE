"""
torch.nn.Module for mlpf_fp32_unfused: MLPF from particleflow
(https://github.com/jpata/particleflow), the ONNX exporter
emits the decomposed MatMul/Softmax graph directly instead 
of a single custom SDPA op, which is what "unfused" means 
here and in the .onnx file itself.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from _common import random_inputs

MODEL_DIR = Path(__file__).resolve().parent.parent
ONNX_PATH = MODEL_DIR / "mlpf_fp32_unfused.onnx"
PARTICLEFLOW_REPO = Path(os.environ.get(
    "SOFIE_PARTICLEFLOW_REPO", "~/.cache/sofie_benchmark/particleflow_src")).expanduser()
CHECKPOINT_DIR = PARTICLEFLOW_REPO / "checkpoints"

if not PARTICLEFLOW_REPO.exists():
    raise RuntimeError(
        f"particleflow not found at {PARTICLEFLOW_REPO}. Configure the "
        f"benchmark with -DSOFIE_BENCHMARK_AOT=ON once to clone it "
        f"automatically, or set SOFIE_PARTICLEFLOW_REPO to point at an "
        f"existing checkout."
    )
sys.path.insert(0, str(PARTICLEFLOW_REPO))
from mlpf.model.mlpf import MLPF  # noqa: E402

_KWARGS_PATH = CHECKPOINT_DIR / "mlpf_model_kwargs.pkl"
_CKPT_PATH = CHECKPOINT_DIR / "mlpf_checkpoint-10-3.812332.pth"
if not _KWARGS_PATH.exists() or not _CKPT_PATH.exists():
    raise FileNotFoundError(
        f"MLPF training checkpoint not found at {CHECKPOINT_DIR} "
        f"(expected {_KWARGS_PATH.name} and {_CKPT_PATH.name}). This checkpoint "
        f"is a large training artifact, not part of the particleflow git repo, "
        f"and isn't auto-downloaded — place both files there by hand, or set "
        f"SOFIE_PARTICLEFLOW_REPO to point at a checkout that already has them "
        f"under checkpoints/."
    )

with open(_KWARGS_PATH, "rb") as f:
    _kwargs = dict(pickle.load(f))
_kwargs["attention_type"] = "math"

_model = MLPF(**_kwargs)
_ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
_model.load_state_dict(_ckpt["model_state_dict"], strict=True)
_model.eval()
print(f"  [particleflow] loaded checkpoint into MLPF('math' attention) "
      f"— {_CKPT_PATH}")

_INPUT_DIM = _kwargs["input_dim"]


def _verify():
    inputs_np = random_inputs([
        ("Xfeat_normed", (1, 37, _INPUT_DIM), np.float32),
        ("mask", (1, 37), np.float32),
    ])
    inputs_np["mask"] = np.ones_like(inputs_np["mask"])

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, inputs_np)

    with torch.no_grad():
        torch_out = _model(
            torch.from_numpy(inputs_np["Xfeat_normed"]),
            torch.from_numpy(inputs_np["mask"]).bool(),
        )
    if isinstance(torch_out, torch.Tensor):
        torch_out = (torch_out,)

    mismatches = []
    for i, (expected, actual) in enumerate(zip(onnx_out, torch_out)):
        actual_np = actual.detach().cpu().numpy()
        if not np.allclose(expected, actual_np, rtol=1e-3, atol=1e-4, equal_nan=True):
            max_diff = np.abs(expected.astype(np.float64) - actual_np.astype(np.float64)).max()
            mismatches.append((i, max_diff))

    if not mismatches:
        print(f"  [verify] {ONNX_PATH}: torch module output matches onnxruntime "
              f"({len(onnx_out)} output(s) checked)")
        return

    for i, max_diff in mismatches:
        print(f"  [WARNING] {ONNX_PATH}, output {i}: torch module output does NOT match "
              f"onnxruntime (max abs diff {max_diff:.4g}). Treat this "
              f"model's AOT numbers as unverified until resolved.")


_verify()


def build(shape_overrides):
    """shape_overrides: {"num_batch": int, "num_elements": int}."""
    num_batch = shape_overrides["num_batch"]
    num_elements = shape_overrides["num_elements"]
    inputs = random_inputs([
        ("Xfeat_normed", (num_batch, num_elements, _INPUT_DIM), np.float32),
    ])
    mask = torch.ones((num_batch, num_elements), dtype=torch.bool)
    return _model, (torch.from_numpy(inputs["Xfeat_normed"]), mask)


DYNAMIC_DIMS = {
    "Xfeat_normed": {0: "num_batch", 1: "num_elements"},
    "mask": {0: "num_batch", 1: "num_elements"},
}
