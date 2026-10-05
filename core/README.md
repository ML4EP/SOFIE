
# SOFIE

SOFIE (___System for Optimized Fast Inference code Emit___) generates C++ functions easily invokable for the fast inference of trained neural network models. It takes ONNX model files as inputs and produces C++ header files that can be included and utilized in a “plug-and-go” style.

This is currently in an early experimental stage.


## Prerequisite
- BLAS or Eigen (for execution of the generated code for inference)

## Installation

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . -j$(nproc)
```

## Usage
SOFIE works in a parser-generator working architecture. With SOFIE, the user gets an ONNX parser for translating models into SOFIE's internal representation.

In a C++ program, we can proceed with an ONNX model:

```c++
SOFIE::RModelParser_ONNX parser;
SOFIE::RModel model = parser.Parse(“./example_model.onnx”);
model.Generate();
model.OutputGenerated(“./example_output.hxx”);
```

And an C++ header file and a `.dat` file containing the model weights will be generated. You can also use

```c++
model.PrintRequiredInputTensors();
```

to check the required size and type of input tensor for that particular model, and use

```c++
model.PrintInitializedTensors();
```

to check the tensors (weights) already included in the model.

To use the generated inference code:

```c++
#include "example_output.hxx"
float input[INPUT_SIZE];
std::vector<float> out = SOFIE_example_model::infer(input);

// Generated header file shall contain a Session class which requires initialization to load the corresponding weights.
SOFIE_example_model::Session s("example_model.dat")

// Once instantiated the session object's infer method can be used
std::vector<float> out = s.infer(input);
```

With the default settings, the weights are contained in a separate binary file, but if the user instead wants them to be in the generated header file itself, they can use approproiate generation options.

```c++
model.Generate(Options::kNoWeightFile);
```

Other such options includes `Options::kNoSession` (for not generating the Session class, and instead keeping the infer function independent).

By default the separate weight file uses a simple text format (`*.dat`). A
binary alternative is the [safetensors](https://huggingface.co/docs/safetensors)
format (`*.safetensors`), which stores the weights as raw little-endian data
behind a small JSON header. It loads faster, round-trips the values bit-exactly,
and can be inspected with the standard Python and Rust safetensors tooling:

```c++
model.Generate(Options::kSafetensorsWeightFile);
```

### Strided inputs

With `Options::kStridedInput` (CPU `Generate` and `GenerateGPU_ALPAKA`) the generated `Session` constructor takes an
extra last argument: one stride array (in elements, not bytes) per input tensor, in the order of the model inputs.
The operators reading the inputs access them through these strides, without first copying them into contiguous memory,
so a transposed, padded or sliced view can be passed directly to `infer`. An empty argument (the default) means
contiguous inputs, and so does an empty array for a single input.

```c++
model.Generate(Options::kStridedInput);
// ...
SOFIE_Model::Session session("model.dat", /*inputStrides=*/{{8, 1}}); // 2D input with leading dimension 8
auto y = session.infer(x_ptr);
```

When the shape is dynamic, the strides are used as given for every call, while contiguous (default) strides follow
the shape of each call. An operator reading a graph input must implement `ROperator::SupportsStridedInput()`, otherwise
generation fails with an error naming it. All the operators support it, except `Custom` (user code) and `SubGraph`
(subgraphs are not supported with strided inputs). Operators reading a strided input are not fused into GPU fusion
kernels. How the inputs are read, by family:

- **Elementwise, binary, comparison, logic, Where, N-ary, Cast, Identity, IsNaN/IsInf, Clip...**: the offset of each
  element is computed from its multi-index; the inputs of binary operators are broadcast with their own strides.
- **Data movement and indexing** (Transpose, Slice, Gather, GatherND, Concat, Split, Pad, Tile, Expand, ScatterND,
  ScatterElements, Trilu, NonZero, TopK): the strides replace the contiguous ones in the index computation. The data
  copied into the output (Reshape/Flatten of a strided input, ScatterND/ScatterElements) is a contiguous copy made
  through the strides: the output of a strided input cannot alias it.
- **Reductions and normalizations** (Reduce*, Softmax, LayerNorm, RMSNorm, BatchNorm, GroupNorm, InstanceNorm,
  L2Normalization, CumSum, Pool): the code computes the contiguous index of an element, which is converted into the
  offset of the strided input.
- **Attention, scans and recurrent operators** (SDPA, Mamba, RWKV, Griffin RG-LRU, RNN, LSTM, GRU): same conversion;
  RNN, LSTM and GRU gather the strided input in their internal input buffer.
- **Conv**: a strided im2col reads the input (all the layouts are supported). **ConvTranspose** (CPU) passes the input
  to BLAS, so its spatial dimensions must be contiguous and either the channel or the last dimension must have a unit
  stride (e.g. NCHW with padded channels, or NHWC); other layouts throw a `std::runtime_error` at inference time.
- **Gemm** (operands A and B, plain 2D): BLAS describes a matrix by a leading dimension and a transpose flag, so a strided
  operand must have a unit stride: `{ld, 1}` (row-major, e.g. padded rows) or `{1, ld}` (column-major). A view with no unit
  stride throws a `std::runtime_error` at inference time (a strided fallback is planned in sofieBLAS). On the GPU the
  strided Gemm goes through the cuBLAS strided-batched entry point (batch of 1), since the sofieBLAS `gemm`/`matmul` calls
  only use dense leading dimensions. Batched MatMul (rank > 2), low rank factorization and a strided bias are not
  supported and fail at generation. (Einsum reads strided inputs with its generic loops instead of BLAS.)

Not every operator has a GPU implementation (e.g. Gelu, Erf, Swish, HardSigmoid, HardSwish, Einsum, ConvTranspose,
InstanceNorm, RNN, LSTM, GRU): the strided input is then supported on the CPU only, like the operator.

SOFIE also supports generating inference code with RDataFrame as inputs, refer to the tutorials below for examples.

## Supported ONNX operators

Here is the updated list of supported ONNX operators. You can obtain this list by doing
```cpp
SOFIE::RModelParser_ONNX parser;
std::vector<std::string> supportedOperators = parser.GetRegisteredOperators();
```

- [x] Abs
- [x] Add
- [x] AveragePool
- [x] BatchNormalization
- [x] Cast
- [x] Concat
- [x] Constant
- [x] ConstantOfShape
- [x] Conv
- [x] ConvTranspose
- [x] Cos
- [x] Div
- [x] Einsum
- [x] Elu
- [x] Equal
- [x] Erf
- [x] Exp
- [x] Expand
- [x] EyeLike
- [x] Flatten
- [x] GRU
- [x] Gather
- [x] Gemm
- [x] GlobalAveragePool
- [x] Greater
- [x] GreaterOrEqual
- [x] GRU
- [x] Identity
- [x] If
- [x] LSTM
- [x] LayerNormalization
- [x] LeakyRelu
- [x] Less
- [x] LessOrEqual
- [x] Log
- [x] MatMul
- [x] Max
- [x] MaxPool
- [x] Mean
- [x] Min
- [x] Mul
- [x] Neg
- [x] Pad
- [x] Pow
- [x] RNN
- [x] Range
- [x] Reciprocal
- [x] ReduceMean
- [x] ReduceProd
- [x] ReduceSum
- [x] ReduceSumSquare
- [x] Relu
- [x] Reshape
- [x] ScatterElements
- [x] Selu
- [x] Shape
- [x] Sigmoid
- [x] Sin
- [x] Slice
- [x] Softmax
- [x] Split
- [x] Sqrt
- [x] Squeeze
- [x] Sub
- [x] Sum
- [x] Tanh
- [x] Tile
- [x] TopK
- [x] Transpose
- [x] Unsqueeze
- [x] Where

The above operators are supported for tensors of the following types:

- [x] float
- [x] double
- [x] int32
- [x] int64
- [x] bool (for comparison operators)

You can also check your model whether all operators are implemented by doing the following:
```c++
SOFIE::RModelParser_ONNX parser;
parser.CheckModel("example_model.ONNX");
```
