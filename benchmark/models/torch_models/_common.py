"""
Shared helpers for benchmark/models/torch_models/*.py.
"""

import numpy as np
import onnxruntime as ort
import torch


def verify_against_onnxruntime(torch_model, onnx_path, inputs, rtol=1e-3, atol=1e-4):
    """
    inputs: dict of {input_name: numpy array}, matching the .onnx graph's
    declared input order for the positional torch_model(*tensors) call.

    Runs the same inputs through onnxruntime and the
    constructed torch module, and raises RuntimeError if any output tensor
    doesn't match within tolerance.
    """
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_order = [i.name for i in sess.get_inputs()]
    onnx_outputs = sess.run(None, {name: inputs[name] for name in input_order})

    torch_inputs = [torch.from_numpy(inputs[name]) for name in input_order]
    torch_model.eval()
    with torch.no_grad():
        torch_outputs = torch_model(*torch_inputs)
    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs = (torch_outputs,)
    elif isinstance(torch_outputs, dict):
        torch_outputs = tuple(torch_outputs.values())

    if len(torch_outputs) != len(onnx_outputs):
        raise RuntimeError(
            f"verification failed for {onnx_path}: "
            f"onnxruntime produced {len(onnx_outputs)} output(s), "
            f"the converted module produced {len(torch_outputs)}"
        )

    for i, (expected, actual) in enumerate(zip(onnx_outputs, torch_outputs)):
        actual_np = actual.detach().cpu().numpy()
        if not np.allclose(expected, actual_np, rtol=rtol, atol=atol, equal_nan=True):
            max_diff = np.abs(expected.astype(np.float64) - actual_np.astype(np.float64)).max()
            raise RuntimeError(
                f"mismatch for {onnx_path}, output {i}: "
                f"max abs diff {max_diff} exceeds rtol={rtol}/atol={atol} — "
                f"do not trust this module's AOT numbers until this is resolved."
            )

    print(f"  [verify] {onnx_path}: torch module output matches onnxruntime "
          f"(rtol={rtol}, atol={atol}, {len(onnx_outputs)} output(s) checked)")


def random_inputs(specs, seed=42):
    """
    specs: list of (name, shape: tuple[int], dtype: np.dtype). For an int64
    "index"-like tensor (edge_index, tgt tokens), fills with zeros.
    Returns {name: np.ndarray}.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape, dtype in specs:
        if np.issubdtype(dtype, np.integer):
            out[name] = np.zeros(shape, dtype=dtype)
        else:
            out[name] = rng.uniform(-1.0, 1.0, size=shape).astype(dtype)
    return out
