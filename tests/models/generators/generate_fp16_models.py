import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def save_case(directory, name, nodes, initializers, input_shape, output_shape, values):
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, output_shape)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, directory / f"{name}.onnx")
    evaluator = ReferenceEvaluator(model)
    for batch in (1, 3, 4):
        sample = values[:batch].astype(np.float16)
        result = evaluator.run(None, {"input": sample})[0]
        sample.astype("<f2").tofile(directory / f"{name}_{batch}.input")
        result.astype("<f2").tofile(directory / f"{name}_{batch}.expected")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    directory = parser.parse_args().outdir
    directory.mkdir(parents=True, exist_ok=True)
    weights = np.array([[1, 2, -1], [0.5, -1, 2], [-2, 0.25, 1]], dtype=np.float16)
    bias = np.array([0.25, -0.5, 1], dtype=np.float16)
    nodes = [
        helper.make_node("Gemm", ["input", "weights", "bias"], ["hidden"],
                         transB=1, alpha=0.5, beta=2.0),
        helper.make_node("Relu", ["hidden"], ["activated"]),
        helper.make_node("Softmax", ["activated"], ["output"], axis=-1),
    ]
    values = np.arange(-6, 6, dtype=np.float16).reshape(4, 3) / 4
    for encoding in ("raw", "typed", "external"):
        if encoding == "typed":
            initializers = [helper.make_tensor("weights", TensorProto.FLOAT16, [3, 3], weights.ravel()),
                            helper.make_tensor("bias", TensorProto.FLOAT16, [3], bias)]
        else:
            initializers = [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")]
        model = save_case(directory, f"dense_{encoding}", nodes, initializers,
                          ["batch", 3], ["batch", 3], values)
        if encoding == "external":
            onnx.save_model(model, directory / "dense_external.onnx", save_as_external_data=True,
                            all_tensors_to_one_file=True, location="dense_external.data", size_threshold=0)
        if encoding == "typed":
            invalid = onnx.ModelProto()
            invalid.CopyFrom(model)
            invalid.graph.initializer[0].int32_data[0] = 65536
            onnx.save(invalid, directory / "invalid_bits.onnx")
    attention = [
        helper.make_node("Transpose", ["input"], ["keys"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["input", "keys"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["probabilities"], axis=-1),
        helper.make_node("MatMul", ["probabilities", "input"], ["attended"]),
        helper.make_node("Cast", ["attended"], ["wide"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["wide"], ["narrow"], to=TensorProto.FLOAT16),
        helper.make_node("ReduceSum", ["narrow", "axes"], ["sum"], keepdims=1),
        helper.make_node("Abs", ["sum"], ["absolute"]),
        helper.make_node("Add", ["absolute", "one"], ["positive"]),
        helper.make_node("Sqrt", ["positive"], ["scale"]),
        helper.make_node("Div", ["narrow", "scale"], ["output"]),
    ]
    attention.insert(0, helper.make_node("Constant", [], ["one"],
                                       value=numpy_helper.from_array(np.array(1, dtype=np.float16))))
    save_case(directory, "attention", attention,
              [numpy_helper.from_array(np.array([-1], dtype=np.int64), "axes")],
              ["batch", 2, 3], ["batch", 2, 3], np.arange(-12, 12, dtype=np.float16).reshape(4, 2, 3) / 12)
    selection = [
        helper.make_node("Abs", ["input"], ["absolute"]),
        helper.make_node("TopK", ["absolute", "count"], ["largest", "indices"], axis=1),
        helper.make_node("GatherElements", ["input", "indices"], ["selected"], axis=1),
        helper.make_node("Greater", ["selected", "zero"], ["positive"]),
        helper.make_node("Where", ["positive", "selected", "zero"], ["clipped"]),
        helper.make_node("Concat", ["largest", "clipped"], ["joined"], axis=1),
        helper.make_node("Clip", ["joined", "zero", "maximum"], ["output"]),
    ]
    save_case(directory, "selection", selection,
              [numpy_helper.from_array(np.array([2], dtype=np.int64), "count"),
               numpy_helper.from_array(np.array(0, dtype=np.float16), "zero"),
               numpy_helper.from_array(np.array(2, dtype=np.float16), "maximum")],
              ["batch", 3], ["batch", 4], values)


if __name__ == "__main__":
    main()
