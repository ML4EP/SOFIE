
# SOFIE

## External GPU workspace and runtime integration

On `alice-custom`, call `model.SetExternalWorkspace()` before
`model.GenerateGPU_ALPAKA(...)` to generate a session that borrows its
intermediate tensor storage. Construct the generated session with its maximum
dynamic dimensions, query `GetWorkspaceSize()`, then call
`SetWorkspace(devicePointer, bytes)`. The pointer must be 256-byte aligned and
the allocation must remain valid until queued GPU work completes. Rebind after
the caller recycles its arena; binding neither owns nor frees that memory.
Inference without a required workspace, insufficient size, misalignment, and
oversized input dimensions throw exceptions.

The lifetime planner shares storage only between tensors with disjoint lifetimes.
Graph outputs stay live through the end of inference. Dynamic output capacity
is reserved at construction, avoiding output reallocation on partial batches.
Weights, persistent state, shape metadata and backend BLAS resources remain
session-owned. Data-dependent shapes and operators that allocate temporary
device buffers during inference are rejected in external-workspace mode.
Default generation continues to allocate its own intermediate pool.

`SOFIE/RGPUModel.hxx` provides the POSIX runtime adapter used by O2. Construct
`SOFIE::RGPUModel` from the ordinary ONNX parser's `RModel`, call `Compile`
with CUDA/HIP, a compiler executable and target architecture, and create
sessions with `CreateSession(nativeStream, device, maxBatch)`. The adapter uses
the framework's Alpaka generator and operators. It accepts one FP32 or FP16 input and
one computed output of the same precision with matching leading batch dimensions and fixed
feature dimensions. Fixed-batch graphs require that exact batch on every call.

The native stream is borrowed. The calling thread must have the session's GPU
device active. Inference enqueues kernels and the copy to caller-owned output
without synchronizing; callers must complete queued work before reusing memory
or destroying sessions. Compiled code is shared across sessions; each session
has independent weights, scratch views and BLAS state. Sessions are not thread-safe.

At runtime, `SOFIE_ROOT`, `ALPAKA_ROOT`, and `SOFIEBLAS_ROOT` select installed
header locations. Configure-time include paths are fallbacks. A C++20-capable
`nvcc` or `hipcc` and cuBLAS/cuBLASLt or hipBLAS/hipBLASLt must be available.
Compiler arguments are passed directly without a shell; compilation errors
include the compiler log. Temporary generated source and libraries are removed
after loading or failure.

With `SOFIE_BUILD_TESTS=ON`, `ctest -R '^SOFIEWorkspace$'` checks lifetime
reuse and overflow. To enable the GPU regression test, also configure
`SOFIE_RUNTIME_TEST_BACKEND=CUDA` and `SOFIE_RUNTIME_TEST_ARCH=sm_80`, or
`SOFIE_RUNTIME_TEST_BACKEND=HIP` and `SOFIE_RUNTIME_TEST_ARCH=gfx906`, choosing
the actual device architecture. After installing SOFIE and loading its package
environment, run `ctest -R '^SOFIERuntimeWorkspace$' --output-on-failure`.
This compiles a Gemm/Relu model at runtime and checks arena rebinding, partial
batches, output bounds, guard bytes, invalid workspaces and session reuse in both precisions.

### FP16 Alpaka models

FLOAT16 is represented by `SOFIE::Float16`: a trivially-copyable, two-byte IEEE
binary16 value. Raw ONNX, external tensor data, and ONNX `int32_data` bit-pattern
initializers are supported. Generated constants preserve the original bits,
including signed zero, subnormals and NaNs. GPU scalar conversions use native
HIP/CUDA half instructions; host conversions round to nearest, ties to even.

FP16 support covers Gemm/MatMul (including transposes, broadcasted batches,
alpha/beta, bias and fused Relu/LeakyRelu), arithmetic, comparison, Cast, Constant,
Relu/LeakyRelu/Sigmoid/Tanh, the basic unary operators, Softmax/LogSoftmax,
reductions, TopK, Gather/GatherElements, Concat, Where, Clip and the existing
type-generic shape/reorganization operators. It does not imply FP16 support for
every ONNX operator: unsupported operators still fail explicitly. The legacy
non-Alpaka CPU Gemm generator does not emit FP16 BLAS.

Dense Alpaka operators use a stream-bound `NativeBlas` session with
hipBLAS/rocBLAS or cuBLAS GemmEx and strided-batched GemmEx. Inputs, weights,
activations, output and planned intermediate tensors stay FP16; matrix products
accumulate in FP32. Bias/activation epilogues calculate in FP32 and store FP16.
This path works without hipBLASLt device kernels (notably gfx906). It also serves
FP32 dense operators. Convolution and low-rank paths retain sofieBLAS.
The package still supplies and links the Lt libraries for those existing paths.

Softmax and reductions accumulate in FP32 for FP16 tensors. FP16 reductions, and
all external-workspace reductions, use one block per output to avoid allocating
partial-reduction buffers during inference. Register/shared-memory accumulators
and BLAS-owned resources are not caller tensor scratch. Persistent weights remain
session-owned. Mixed Cast intermediates use their declared element widths.

Run `ctest -R '^SOFIE(Float16|Float16Onnx|RuntimeWorkspace|Workspace)$' --output-on-failure`
after building. With Python ONNX and NumPy available, test fixtures cover raw,
typed and external initializers, malformed bits, dense layers, attention,
reductions, selection and preprocessing. The optional GPU tests compare against
ONNX reference outputs at batches 1/3/4 on a borrowed stream and rebound guarded
arenas. They compile generated models at test runtime. No upstream quantization PR
is required or merged: these changes extend ordinary FLOAT16 operators rather
than the PR's INT8/FP8 execution path.

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
