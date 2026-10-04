# SOFIE

This is an experimental standalone version of **SOFIE** — a tool for Fast ML Inference
within [ROOT](https://root.cern), the scientific data analysis framework.

This standalone is especially developed for implementing and evaluating inference on
**heterogeneous architectures** (CUDA GPUs, AMD GPUs via HIP/ROCm, CPUs) using the
[Alpaka](https://github.com/alpaka-group/alpaka) portability layer.

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

### GPU tests (Alpaka/CUDA)

Alpaka-based GPU tests compile SOFIE-generated inference code as CUDA and verify
correctness against reference outputs.  They require the CUDA Toolkit and a
compatible NVIDIA GPU.

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
| `-DENABLE_ALPAKA_TESTS=ON` | `OFF` | Enable Alpaka GPU tests |
| `-DALPAKA_BACKEND=<val>` | `cuda` | Alpaka backend: `cuda`, `hip`, `cpu`, `sycl` |

The test executable is `TestCustomModelsFromONNXForAlpakaCuda`.  ONNX model files
used as test inputs are located in `tests/models/onnx/`.  Models with symbolic
(dynamic) input dimensions are specialised by the emitter before testing.

---

## Benchmarking

The benchmark toolkit (`benchmark/`) measures **inference latency and throughput** for
ONNX models compiled by SOFIE and executed via Alpaka.  It supports an optional
side-by-side comparison with **ONNX Runtime GPU**.

### Supported backends

| Backend | CMake value | Status |
|---------|-------------|--------|
| NVIDIA CUDA | `CUDA` (default) | Supported |
| AMD HIP/ROCm | `HIP` | Planned |

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

SOFIE generates Alpaka-based inference code that is portable across GPU
architectures:

- **NVIDIA CUDA** — select the SM architecture with
  `-DSOFIE_BENCHMARK_CUDA_ARCH=<sm>` (e.g. `75` for Turing, `86` for Ampere,
  `90` for Hopper).
- **AMD HIP/ROCm** — the Alpaka backend tag (`alpaka::TagGpuHipRt`) and the
  `SOFIE_BACKEND_HIP` compile-time define are already wired in
  `benchmark/src/BenchmarkBackend.hxx`; full build-system integration is in
  progress.
- **CPU** — a serial CPU Alpaka backend (`alpaka::TagCpuSerial`) is available as a
  fallback for debugging and portability testing.

---

## Project Structure

```
SOFIE/
├── core/           # Core SOFIE library (RModel, operators, RTensor, code generators)
├── parsers/        # ONNX → RModel parser
├── tests/
│   ├── cpu/        # CPU unit/integration tests (GoogleTest)
│   ├── alpaka/     # Alpaka/CUDA GPU tests
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

### GPU inference (Alpaka/CUDA)

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

## Inspiration

The standalone version of SOFIE is developed with inspiration from the standalone
version of RooFit developed by Jonas Rembser, which can be found
[here](https://github.com/guitargeek/roofit).
