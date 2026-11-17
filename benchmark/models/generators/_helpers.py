"""Shared helpers for the benchmark model builders in this directory.

A builder reproduces the graph of the original export node for node; weight
tensors are seeded random.
"""

import numpy as np
from onnx import TensorProto, helper

inf = float("inf")

BOOL = TensorProto.BOOL
DOUBLE = TensorProto.DOUBLE
FLOAT = TensorProto.FLOAT
INT32 = TensorProto.INT32
INT64 = TensorProto.INT64


def _vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def _tensor(name, dtype, dims, vals):
    return helper.make_tensor(name, dtype, dims, vals)


def _rand(rng, name, dims):
    """Float weight tensor drawn from rng, a np.random.RandomState."""
    k = 1.0 / np.sqrt(dims[-1])
    vals = rng.uniform(-k, k, int(np.prod(dims))).astype(np.float32)
    return helper.make_tensor(name, TensorProto.FLOAT, dims, vals)


def _model(graph, opset, ir_version, **kwargs):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], **kwargs)
    model.ir_version = ir_version
    return model
