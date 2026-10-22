# SOFIE

This is an experimental standalone version of **SOFIE** — a tool for Fast ML Inference
within [ROOT](https://root.cern), the scientific data analysis framework.

This standalone is especially developed for implementing and evaluating inference on
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
├── core/           # Core SOFIE library (RModel, operators, RTensor, code generators)
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
lifetimes share memory. All regions are aligned to 256 bytes, so `alloc` has to return a pointer with that alignment (as `cudaMalloc` does).
For each tensor `assign(offset)` is called with the offset SOFIE planned for it, and returns the address where the tensor is laid down
(at least as many bytes as the tensor, rounded up to 256, are accessed from there); it must not return a null pointer. This lets the application
map offsets of the block onto memory of its own, for example an arena it manages. Without `assign` the tensor is placed at `block + offset`.
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

With `Options::kKernelOnly`  combined with `Options::kStridedInput`, the generated header contains the heterogeneous alpaka kernels, `sofie_strided_layout` (the logical shape and the strides, in elements, of an input) and `sofie_strided_offset`
(the offset in the input buffer of the element with a given row-major logical index). The kernels reading a graph input
take its `sofie_strided_layout` as an argument, so there is no stride storage: the caller builds the layouts. A broadcast
input has the shape of the output and a zero stride for the broadcast dimensions.

SOFIE also supports generating inference code with RDataFrame as inputs, refer to the tutorials below for examples.

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

## Inspiration

The standalone version of SOFIE is developed with inspiration from the standalone
version of RooFit developed by Jonas Rembser, which can be found
[here](https://github.com/guitargeek/roofit).
