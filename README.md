# SOFIE - System for Optimized Fast Inference code Emit

A tool for Fast ML Inference across heterogeneous architectures through
code generation.

SOFIE implements and evaluates inference on
**heterogeneous architectures** (CUDA GPUs, AMD GPUs via HIP/ROCm, CPUs) using the
[alpaka](https://github.com/alpaka-group/alpaka) portability layer.

---

## Installation

### Prerequisites

- CMake ≥ 3.16
- C++20-capable compiler (GCC ≥ 11, Clang ≥ 14)
- *(Optional for GPU testing/benchmarking)* CUDA Toolkit ≥ 11.8

### 1. Clone and build

```bash
git clone https://github.com/sanjibansg/SOFIE.git
cd SOFIE
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --target install -j$(nproc)
```

Alternatively, use the provided CMake presets (`cmake --list-presets`):

```bash
cmake --preset cpu-tests && cmake --build --preset cpu-tests -j$(nproc) && ctest --preset cpu-tests
```

### 2. Environment

```bash
source scripts/setup.sh   # adds SOFIE_core and SOFIE_parsers to LD_LIBRARY_PATH
```

---

## Testing

Unit and integration tests are enabled with `-DSOFIE_BUILD_TESTS=ON` and require
[GoogleTest](https://github.com/google/googletest).

### CPU / default tests

```bash
cmake -DSOFIE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### GPU tests (alpaka, CUDA or HIP)

alpaka-based GPU tests compile SOFIE-generated inference code for an NVIDIA (CUDA)
or AMD (HIP) GPU and verify correctness against reference outputs.  They require
the CUDA Toolkit or ROCm, and a matching GPU.

```bash
cmake -DSOFIE_BUILD_TESTS=ON \
      -DENABLE_ALPAKA_TESTS=ON \
      -DALPAKA_BACKEND=cuda \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . -j$(nproc)
ctest --output-on-failure
```

| CMake flag | Default | Description |
|---|---|---|
| `-DSOFIE_BUILD_TESTS=ON` | `OFF` | Enable the test suite |
| `-DSOFIE_USE_CCACHE=ON` | `OFF` | Use ccache as compiler launcher |
| `-DENABLE_ALPAKA_TESTS=ON` | `OFF` | Enable alpaka GPU tests |
| `-DALPAKA_BACKEND=<val>` | `cuda` | alpaka backend: `cuda`, `hip`, `cpu`, `sycl` |
| `-DROCM_BASE=<path>` | `/opt/rocm` | ROCm installation prefix (`hip` backend only) |

The test executable is `TestCustomModelsFromONNXForAlpakaCuda` with
`-DALPAKA_BACKEND=cuda` (the default), or `TestCustomModelsFromONNXForAlpakaHip`
with `-DALPAKA_BACKEND=hip`.  ONNX model files
used as test inputs are located in `tests/models/onnx/`.  Models with symbolic
(dynamic) input dimensions are specialised by the emitter before testing.

---

## Benchmarking

The benchmark toolkit (`benchmark/`) measures **inference latency and throughput** for
ONNX models compiled by SOFIE and executed via alpaka.  It supports an optional
side-by-side comparison with **ONNX Runtime GPU**.

### Supported backends

| Backend | CMake value | Status |
|---------|-------------|--------|
| NVIDIA CUDA | `CUDA` (default) | Supported |
| AMD HIP/ROCm | `HIP` | Supported |

### Quick start

```bash
# Place .onnx models in benchmark/models/ first
cmake -B build \
      -DSOFIE_BENCHMARK=ON \
      -DSOFIE_BENCHMARK_BACKEND=CUDA \
      -DSOFIE_BENCHMARK_CUDA_ARCH=86 \   # e.g. 86 for RTX 30xx, 80 for A100
      /path/to/SOFIE
cmake --build build --target sofie_benchmark -j$(nproc)
cd build/benchmark && ./sofie_benchmark
```

For a full reference of benchmark CMake flags, runtime options, the large-input
cluster benchmark, and instructions for adding new backends, see
[benchmark/README.md](benchmark/README.md).

### Profiling

Add `-DSOFIE_BENCHMARK_PROFILE=ON` to enable **per-operator GPU timing** and a
**CPU/GPU memory breakdown** printed after each model's throughput line.

```bash
cmake -B build \
      -DSOFIE_BENCHMARK=ON \
      -DSOFIE_BENCHMARK_PROFILE=ON \
      /path/to/SOFIE
cmake --build build --target sofie_benchmark -j$(nproc)
cd build/benchmark && ./sofie_benchmark
```

> Profiling inserts `alpaka::wait(queue)` after each operator, which serialises
> GPU execution.  Use a non-profile build for peak-throughput numbers.

Profiling can also be enabled on a per-model basis outside the benchmark by
passing `Options::kProfile` at code-generation time (see
[Profiling in user code](#profiling-in-user-code) below).

---

## GPU Architecture Support

SOFIE generates alpaka-based inference code that is portable across GPU
architectures:

- **NVIDIA CUDA** — select the SM architecture with
  `-DSOFIE_BENCHMARK_CUDA_ARCH=<sm>` (e.g. `75` for Turing, `86` for Ampere,
  `90` for Hopper).
- **AMD HIP/ROCm** — the alpaka backend tag (`alpaka::TagGpuHipRt`) and the
  `SOFIE_BACKEND_HIP` compile-time define are already wired in
  `benchmark/src/BenchmarkBackend.hxx`; full build-system integration is in
  progress.
- **CPU** — a serial CPU alpaka backend (`alpaka::TagCpuSerial`) is available as a
  fallback for debugging and portability testing.

---

## Project Structure

```
SOFIE/
├── core/           # Core SOFIE library 
|  ├── functions/   # RFunction implementations for GNN operations
|  ├── operators/   # ROperator implementations for ML operations
|  ├── utils/       # helper methods for generator and to be embedded in the generated code
├── parsers/        # ONNX → RModel parser
├── tests/
│   ├── cpu/        # CPU unit/integration tests (GoogleTest)
│   ├── alpaka/     # alpaka/CUDA GPU tests
│   ├── common/     # Shared test helpers
│   ├── models/     # Test models: generators/, onnx/ (+ references/)
│   ├── templates/  # Emitter sources configured by CMake
│   └── gnn/        # GNN emitter examples
├── benchmark/      # Latency / throughput benchmark toolkit
│   ├── models/     # Place .onnx benchmark models here
│   └── src/        # CMake-configured source templates
├── scripts/        # setup.sh (runtime env), check_style.sh (clang-format/tidy)
└── cmake/          # CMake modules and config templates
```

---

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

// Generated header file shall contain a Session struct which requires initialization to load the corresponding weights.
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

### GPU inference with alpaka

The GPU code is generated leveraging the [alpaka](https://github.com/alpaka-group/alpaka) portability layer: the generated header contains the alpaka kernels of the
operators and a `Session` class which launches them, so the same header runs on NVIDIA GPUs (CUDA), AMD GPUs (HIP), and on CPUs, depending on the
accelerator tag the `Session` is instantiated with. Matrix operations (Gemm, Conv) use [sofieBLAS](https://github.com/ML4EP/sofieBLAS).

**Generation.** Parse the model as for the CPU and call `GenerateGPU_ALPAKA`:

```c++
SOFIE::RModelParser_ONNX parser;
SOFIE::RModel model = parser.Parse("./example_model.onnx");
model.GenerateGPU_ALPAKA();                                   // GenerateGPU_ALPAKA(options, batchSize = -1, verbose = false)
model.OutputGenerated("./example_model_GPU_ALPAKA.hxx");      // header, and the weights file next to it
```

An operator without an alpaka implementation makes the generation fail with an error naming it (see [Supported operators](#supported-operators)).
`batchSize` replaces a symbolic batch dimension by a fixed value. The options can be combined with `|`:

| Option | Effect |
|---|---|
| `Options::kNoWeightFile` | Weights embedded in the header instead of a separate file |
| `Options::kKernelOnly` | Only the kernels, no `Session` |
| `Options::kStridedInput` | Inputs with custom strides, see [Strided inputs](#strided-inputs) |
| `Options::kProfile` | Per-operator timing, see [Profiling in user code](#profiling-in-user-code) |
| `Options::kExternalStream` | The `Session` runs on a stream of the application, see below |
| `Options::kCustomMemory` | The application provides the device memory, see below |
| `Options::kDebug` | Verbose generation and `Session`, memory checks, see below |

**The `Session`.** The header defines `SOFIE_<model>::Session<tagAcc>`, a class template on the alpaka accelerator tag, for example
`alpaka::TagGpuCudaRt`, `alpaka::TagGpuHipRt` or `alpaka::TagCpuSerial`. A session creates the platform, the device number 0 of the accelerator
(`devAcc`), the host device (`hostAcc`) and a non-blocking queue (`queue`), allocates the device buffers, loads the weights and instantiates the kernels, so
the creation is the expensive step and a session is meant to be reused for many inferences. The members are public.

```c++
SOFIE_example_model::Session<alpaka::TagGpuCudaRt> session("example_model_GPU_ALPAKA.dat");
```

The constructor takes the weights file and, for a model with symbolic input dimensions, the maximum value of each one of them (`size_t`, with the value of
the model as default), which sizes the device buffers. A session can also run on a queue of the application, `Session(QueueAcc &queue, filename, ...)`;
the sessions of the sub-graphs of a model (`If`) use the queue of the main one.

**Types.** The tensors are one dimensional alpaka objects of the type of the tensor (`float`, `double`, `int32_t`, `int64_t`, `uint8_t`):

| Type | alpaka type | Ownership |
|---|---|---|
| `BufF1D`, `BufD1D`, `BufI321D`, `BufI641D`, `BufUI81D` | `alpaka::Buf<Acc, T, Dim, Idx>` | owns its device memory |
| `ViewF1D`, ..., `ViewConstF1D`, ... | `alpaka::ViewPlainPtr<DevAcc, T, Dim, Idx>` | view of memory owned by someone else |

with `Idx = std::size_t`, `Dim = alpaka::DimInt<1>` and `Ext1D = alpaka::Vec<Dim, Idx>`.

**Inference.** The inputs are device buffers, and the shape parameters of the model are passed before the first input whose shape uses them:

| Method | Description |
|---|---|
| `infer(params..., inputs...)` | Runs the model, waits for the queue and returns the output buffer (a `std::tuple` for several outputs) |
| `infer_nosync(params..., inputs...)` | The same without waiting: call `alpaka::wait(session.queue)` before reading the results |
| `infer(std::span<ViewConstF1D const> inputs, std::span<ViewF1D> outputs, params...)` | Takes views and copies the results into the output views of the caller, then waits |

The returned buffers belong to the session and are overwritten by the next inference. The values of the shape parameters cannot exceed
the maximum values given to the constructor (otherwise a `std::runtime_error` is thrown).

```c++
using Idx = std::size_t;
using Ext1D = alpaka::Vec<alpaka::DimInt<1>, Idx>;
alpaka::PlatformCpu hostPlatform{};
auto host = alpaka::getDevByIdx(hostPlatform, 0);

auto x_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{12}));       // fill it through alpaka::getPtrNative(x_h)
auto x_d = alpaka::allocBuf<float, Idx>(session.devAcc, Ext1D::all(Idx{12}));
alpaka::memcpy(session.queue, x_d, x_h);

auto y_d = session.infer(x_d);

auto y_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{4}));
alpaka::memcpy(session.queue, y_h, y_d);
alpaka::wait(session.queue);
```

**Other members of the `Session`.**

| Member | Description |
|---|---|
| `resetState(queue)` | Zeroes the state of the recurrent operators and the intermediate buffers |
| `GetIntermediateMemoryPoolSize()`, `GetTotalFreeMemory()`, `GetLargestFreeBlock()`, `GetFragmentation()` | Statistics of the memory pool of the intermediate tensors |
| `threadsPerBlock` | Number of threads per block of the kernels (256 by default) |

**Compilation.** The header includes `alpaka/alpaka.hpp` and `sofieBLAS/sofieBLAS.hpp`, so the alpaka and sofieBLAS include directories are needed, and the code
is compiled for the accelerator: for CUDA as C++20 CUDA code with `-DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_HAS_STD_ATOMIC_REF` and the `nvcc` options
`--extended-lambda --expt-relaxed-constexpr`, linking `cudart`, `cublas` and `cublasLt`. The CMake of `tests/` is a working example.

### External CUDA/HIP stream

With `Options::kExternalStream` (alpaka GPU generation, a `Session` class is required) the generated `Session` never creates a
queue. Its only constructor takes the native `cudaStream_t` / `hipStream_t` of the caller as first argument and wraps it as a
non-owning alpaka queue, which all the kernels, copies and BLAS calls of the session and of its sub-graphs are enqueued on.
The stream must belong to the current device and outlive the session; the session never destroys it.

```c++
model.GenerateGPU_ALPAKA(Options::kExternalStream);
// ...
cudaStream_t stream = /* created by the application */;
SOFIE_Model::Session<alpaka::TagGpuCudaRt> session(stream, "model.dat");
auto y = session.infer(x_d);
cudaStreamSynchronize(stream);
```

Without the option the session creates its own queue, or uses an alpaka queue passed to its constructor.

### Custom memory

With `Options::kCustomMemory` (alpaka GPU generation, a `Session` class is required) the application owns the device memory of the
session and SOFIE does the planning. The generated `Session` constructor takes a `MemoryProvider` as first argument (after the queue or
stream, when those are passed), defined in the model namespace of the header:

```c++
struct MemoryProvider {
   std::function<void *(std::size_t)> alloc;                               // device memory block of the given size
   std::function<void *(std::size_t)> assign;                              // optional: offset in the block -> address of that offset
   std::function<void(void *)> dealloc;                                    // optional: called when the session is destroyed
};
```

The constructor computes the size of everything the session needs on the device, calls `alloc` once with that size, and places
the tensors in the block at offsets computed by SOFIE: every tensor gets its own region (weights, persistent state, shape tensors,
graph outputs, operator scratch memory), and the intermediate tensors are packed by SOFIE's lifetime-aware planner so that tensors with disjoint
lifetimes share memory. All regions are aligned to 256 bytes, so `alloc` has to return a pointer with that alignment.
For each tensor `assign(offset)` is called with the offset SOFIE planned for it, and returns the address where the tensor is laid down
(the memory from there has to hold the tensor, and be aligned to 256 bytes); it must not return a null pointer. This lets the application
map offsets of the block onto memory of its own. Without `assign` the tensor is placed at `block + offset`.
The weights are no exception: their regions are assigned like those of every other tensor, then the constructor copies the weights read from the
weight file into them.

```c++
model.GenerateGPU_ALPAKA(Options::kCustomMemory);
// ...
SOFIE_Model::MemoryProvider memory;
memory.alloc = [](std::size_t bytes) { void *p; cudaMalloc(&p, bytes); return p; };
memory.dealloc = [](void *p) { cudaFree(p); };
SOFIE_Model::Session<alpaka::TagGpuCudaRt> session(memory, "model.dat");
```

The block is released through `dealloc` after the queue has been waited on in the session destructor. With this option the generated
`Session` makes no device allocation with alpaka: generation fails with an error naming the allocation if an operator needs device memory
that cannot be planned in the constructor. A model with sub-graphs calls the provider once for the session of the main graph and once for each sub-graph session.
`GetIntermediateMemoryPoolSize()` reports the size of the block, the free-memory statistics refer to the intermediate tensors.
Tensors with a dynamic shape are placed with the size given by the maximum values of the shape parameters passed to the constructor.
The option can be combined with `Options::kExternalStream`.

### Strided inputs

With `Options::kStridedInput` the generated `Session` constructor takes an
extra argument: one stride array in elements per input tensor, in the order of the model inputs.
The operators reading the inputs access them through these strides, without copying them into contiguous memory for any transformation,
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
generation fails with an error naming it.

### Kernel-only mode

`Options::kKernelOnly` generates only the alpaka kernels of the operators, for applications which manage the device buffers, the queue and the launches on
their own, for example to use the SOFIE kernels in a larger pipeline.

```c++
model.GenerateGPU_ALPAKA(Options::kKernelOnly);
model.OutputGenerated("MyModel_KernelOnly_GPU_ALPAKA.hxx");
```

The header has no `Session`, no weights file and no memory management. In the namespace `SOFIE_<model>` it defines:

- `sofie_workdiv<Dim, Idx>(numElements, blockSize = 256)`, the work division covering a number of elements with blocks of threads;
- one kernel functor per operator (shared by the operators of the same kind) or per group of fused operators.
  A kernel takes the device pointers of its inputs and outputs, the number of elements and the attributes of the operator.

The caller launches the kernels with alpaka, in the order of the operators and with buffers of its own:

```c++
using Acc = alpaka::TagToAcc<alpaka::TagGpuCudaRt, alpaka::DimInt<1>, std::size_t>;
auto workDiv = SOFIE_LeakyReluKernelOnly::sofie_workdiv<alpaka::DimInt<1>, std::size_t>(Ext1D::all(n));
alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, SOFIE_LeakyReluKernelOnly::LeakyReluKernel{},
                                                     alpaka::getPtrNative(input_d), alpaka::getPtrNative(output_d), n, /*alpha=*/0.1f));
```

An operator whose computation is a BLAS call only has its auxiliary kernels in the header.
Models with sub-graphs (`If`) are not supported in this mode.

Combined with `Options::kStridedInput`, the header also contains `sofie_strided_layout` (the logical shape and the strides, in elements, of an input) and `sofie_strided_offset`
(the offset in the input buffer of the element with a given row-major logical index). The kernels reading a graph input
take its `sofie_strided_layout` as an argument, so there is no stride storage: the caller builds the layouts. A broadcast
input has the shape of the output and a zero stride for the broadcast dimensions.

### Debug mode

`Options::kDebug` facilitates useful information to debug the generation process and the generated code; it can be used alone or combined with any of the other modes (`kCustomMemory`, `kExternalStream`,
`kStridedInput`, `kProfile`).

**Verbose generation and execution.**

- The generator prints `[SOFIE generator] <model>: ...` lines for the initialization of every operator (also for the CPU generation), the model summary,
  the fusion groups, every device tensor and pooled tensor (with its lifetime), the kernel generated for every operator (or shared with an operator of the
  same kind), the init code and the launch code of every operator (with its number of kernel launches and BLAS calls), the sub-graphs, the kernel objects and
  the helper functions of the generated code.
- The generated `Session` prints `[SOFIE debug] ...` lines while it runs: the construction, the reading and the upload of the weights, the allocation
  of the device buffers, the pool or the memory block with the placement of every tensor, the kernel objects instantiated with the session, the
  initialization of every operator, and, in each inference, the launch of every operator with its number of kernel launches and BLAS calls.
  The printing can be switched off at run time with `SOFIE_Model::DebugLogEnabled() = false;`.

**Memory diagnostics.** The generated `Session` has these methods, for the memory managed by SOFIE in any mode: the buffers allocated by alpaka and the
pool of the intermediate tensors in the default mode, the regions of the block and the pooled tensors with `Options::kCustomMemory`.

```c++
session.GetMemoryReport();     // table with the origin (alpaka, pool, block), offset, size, reserved size and lifetime of every tensor, and the padding
session.GetMemoryRegions();    // the same information as a vector of MemoryRegionInfo (name, address, offset, sizes, lifetime, origin)
session.GetPaddingBytes();     // bytes reserved for the alignment
session.ValidateMemory();      // throws if a tensor has no memory, is not aligned to 256 bytes (the offset in the pool, the address in the block)
                               // or overlaps with another one alive at the same time
```

`ValidateMemory` is also called at the end of the construction of the session. With `Options::kCustomMemory` it runs before the weights are uploaded.

**Generator checks.** They can be called after the generation of a model with `Options::kDebug`:

```c++
model.GenerateGPU_ALPAKA(Options::kCustomMemory | Options::kDebug, -1, /*verbose=*/true);   // verbose also prints the device tensors
std::string tensors = model.GetDeviceRegionsReport_GPU_ALPAKA();   // size and reserved size of the device tensors known at generation
model.CheckNoDeviceAllocation_GPU_ALPAKA();                        // throws if the generated code allocates device memory with alpaka (kDebug and kCustomMemory only)
```

`CheckNoDeviceAllocation_GPU_ALPAKA` is only available for a model generated with both `Options::kDebug` and `Options::kCustomMemory`: it is a debug check, and it checks the
contract of that option (in the default mode the code allocates through alpaka, by design). It throws an exception for any other model; with the two options
the generator calls it by itself after the generation. `GetDeviceRegionsReport_GPU_ALPAKA` is also only available with `Options::kDebug`, in any memory mode, and throws an exception otherwise.

---

## Profiling in user code

Both the CPU and GPU code generators accept `Options::kProfile` to embed
per-operator timing and memory reporting directly in the generated session struct.

### CPU inference

```cpp
#include "SOFIE/RModel.hxx"
#include "SOFIE/RModelParser_ONNX.hxx"

SOFIE::RModelParser_ONNX parser;
SOFIE::RModel model = parser.Parse("my_model.onnx");

// Generate with profiling enabled
model.Generate(SOFIE::Options::kProfile);
model.OutputGenerated("MyModel.hxx");
```

The generated `Session` struct gains:

| Method | Description |
|--------|-------------|
| `PrintProfilingResults(bool order=true)` | Per-operator mean ± stderr (µs), sorted by avg time |
| `ResetProfilingResults()` | Clear accumulated timing data |
| `GetOpAvgTime()` | `std::map<std::string, double>` of averages |
| `GetOpVariance()` | `std::map<std::string, double>` of variances |

```cpp
#include "MyModel.hxx"
SOFIE_MyModel::Session session("MyModel.dat");

// Warmup
for (int i = 0; i < 10; ++i) session.infer(input);
session.ResetProfilingResults();

// Timed runs
for (int i = 0; i < 100; ++i) session.infer(input);
session.PrintProfilingResults();
```

### GPU inference (alpaka/CUDA)

```cpp
model.GenerateGPU_ALPAKA(SOFIE::Options::kProfile);
model.OutputGenerated("MyModel_GPU_ALPAKA.hxx");
```

The generated GPU `Session` additionally provides:

| Method | Description |
|--------|-------------|
| `PrintProfilingResults(bool order=true)` | Per-operator GPU wall-clock time (µs) with `alpaka::wait` sync |
| `ResetProfilingResults()` | Clear accumulated timing data |
| `GetOpAvgTime()` | `std::map<std::string, double>` of averages |
| `PrintMemoryInfo()` | CPU/GPU memory breakdown (computed at code-gen time) |

```cpp
#include "MyModel_GPU_ALPAKA.hxx"
SOFIE_MyModel::Session<AccTag> session("MyModel_GPU_ALPAKA.dat");

for (int i = 0; i < 10; ++i) session.infer(input_d);  // warmup
session.ResetProfilingResults();

for (int i = 0; i < 100; ++i) session.infer(input_d);  // timed
session.PrintProfilingResults();
session.PrintMemoryInfo();
```

> **Timing accuracy:** `alpaka::wait(queue)` is called after each operator kernel
> so the wall-clock measurement captures actual GPU execution time.  This
> disables kernel pipelining; use a non-profile build for throughput measurement.

---

## Supported operators

The ONNX operators the parser can translate are listed by the parser itself, in alphabetical order:

```cpp
SOFIE::RModelParser_ONNX parser;
std::vector<std::string> supportedOperators = parser.GetRegisteredOperators();
```

The GPU (alpaka) generation supports the operators implemented by classes deriving from `ROperator_ALPAKA`
(`core/inc/SOFIE/operators/ROperator_ALPAKA.hxx`). As `CheckModel` does for the CPU, the operators of a model without a GPU implementation can be
listed before generating any code:

```cpp
SOFIE::RModelParser_ONNX parser;
bool supported = parser.CheckModelGPU("./example_model.onnx");   // false if any operator is missing for the CPU or for the GPU

SOFIE::RModel model = parser.Parse("./example_model.onnx");
supported = model.CheckGPUOperators();   // prints the operators without an alpaka implementation, sub-graphs included
```

`model.GenerateGPU_ALPAKA()` rejects such a model by throwing an exception naming the first of them and its position in the model.
`SOFIE::GetAlpaka(op)` returns the alpaka interface of an operator, or `nullptr` if it has none.

---

## License

[LGPL-2.1](https://github.com/ML4EP/SOFIE/blob/main/LICENSE)
