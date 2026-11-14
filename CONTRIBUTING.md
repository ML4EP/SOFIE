# Contributing to SOFIE

Thank you for your interest in contributing to **SOFIE** (System for Optimized Fast Inference code Emit)! SOFIE is developed within the ML4EP Project of the EP-SFT Group at CERN to generate fast, self-contained inference code for deep learning models across CPUs and heterogeneous GPU architectures via [alpaka](https://github.com/alpaka-group/alpaka).

We welcome contributions of all kinds, including new operator implementations, GPU kernel optimizations, bug fixes, and documentation improvements.

---

## Getting Started

### 1. Prerequisites

- **CMake** ≥ 3.16
- **C++20-capable compiler** (GCC ≥ 11, Clang ≥ 14)
- *(Optional for GPU development)* **CUDA Toolkit** ≥ 11.8 or **ROCm/HIP** ≥ 5.0
- *(Optional)* Python 3 with `onnx`, `torch`, and `numpy` for test model generation

### 2. Fork and Clone

1. Fork the repository on GitHub: [https://github.com/ML4EP/SOFIE](https://github.com/ML4EP/SOFIE)
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/SOFIE.git
   cd SOFIE
   git remote add upstream https://github.com/ML4EP/SOFIE.git
   ```

### 3. Branching Strategy

- Active development targeting heterogeneous GPU / Alpaka features is based on the **`gpu/alpaka`** branch.
- Always branch off the latest `gpu/alpaka`:
   ```bash
   git fetch upstream
   git checkout -b <type>/<descriptive-name> upstream/gpu/alpaka
   ```
- Use descriptive branch prefixes:
  - `feat/` for new features or operator implementations
  - `fix/` for bug fixes
  - `perf/` for performance optimizations
  - `docs/` for documentation updates
  - `refactor/` for code restructuring

---

## Development & Code Standards

### C++ Conventions

- **Standard**: C++20.
- **Namespaces**: All core and parser classes reside under the `SOFIE` namespace.
- **Operator Architecture**:
  - Operators reside in `core/inc/SOFIE/operators/ROperator_<Name>.hxx`.
  - GPU-supported operators must inherit from `ROperator_ALPAKA` (`core/inc/SOFIE/operators/ROperator_ALPAKA.hxx`).
  - Assign an explicit `OperatorKind` enum value in `core/inc/SOFIE/operators/ROperator.hxx` and initialize `fKind` in constructors.
  - Implement `Generate_GPU_Kernel_ALPAKA`, `Generate_GPU_Kernel_Definitions_ALPAKA`, and `Generate_GPU_ALPAKA`.
  - When applicable, support strided input layout using `GenerateStridedUnaryKernel` / `GenerateStridedUnaryLaunch`.
- **Memory Safety**:
  - Memory buffers for GPU execution must declare whether they are temporary intermediates or persistent across session inferences via `GetPersistentTensorNames_GPU_ALPAKA()`.
  - Respect custom memory pool and external stream options (`Options::kCustomMemory`, `Options::kExternalStream`).
- **Formatting**:
  - Follow the ROOT coding style (3-space indentation, consistent bracing).
  - Preserve existing docstrings and comments.

### Commit Messages

We adhere to the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short description in present tense>

[optional body with details and rationale]

[optional footer: Fixes #<issue_number>]
```

Examples:
- `feat(alpaka): add support for dynamic shapes in Concat operator (#42)`
- `fix(core): register missing OperatorKind IDs and prevent deduplication collisions (#46, #47)`
- `perf(conv): move WeightVecKernel and dilation setup from infer to session init (#25)`
- `docs: update README with contribution guidelines (#74)`

### AI-Assisted Contributions Policy

Non-maintainers who use AI or LLM-based coding assistants (such as ChatGPT, Claude, GitHub Copilot, Gemini, etc.) to draft or assist with code contributions must adhere to the following guidelines:

- **Explicit Declaration**: Any AI assistance must be explicitly declared in your pull request description and commit trailers (e.g., `Assisted-by: <tool-name>`). Transparency helps maintainers evaluate review depth, edge cases, and architectural compliance.
- **Author Responsibility**: Contributors are fully responsible for the correctness, maintainability, and safety of every line of code submitted. Do not submit AI-generated code without thoroughly reading, understanding, and testing it locally.
- **Architectural & Style Adherence**: Code must strictly conform to SOFIE's C++20 conventions, Alpaka kernel execution model, memory layout rules, and existing error-handling patterns. Boilerplate or unverified LLM output will be closed.

---

## Testing

All contributions must include appropriate tests and pass the local test suite without errors or warnings.

### Building and Running CPU Tests

```bash
cmake --preset cpu-tests
cmake --build --preset cpu-tests -j$(nproc)
ctest --preset cpu-tests --output-on-failure
```

Or using manual CMake flags:
```bash
cmake -B build -DSOFIE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Running GPU Tests (Alpaka CUDA / HIP)

If hardware is available:
```bash
cmake -B build -DSOFIE_BUILD_TESTS=ON -DENABLE_ALPAKA_TESTS=ON -DALPAKA_BACKEND=cuda
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Writing New Tests

- Unit tests for general operator support and code emission: `tests/cpu/TestAlpakaOperatorSupport.cxx` or `tests/cpu/TestSofieModels.cxx`.
- Alpaka GPU integration tests: `tests/alpaka/`.
- Use GoogleTest (`TEST(...)` / `EXPECT_*` / `ASSERT_*`) macros.

---

## Opening a Pull Request

### Pull Request Limits & Review Policy

The SOFIE project values the time you invest to contribute and welcomes pull requests. However, maintainers need sufficient time to thoroughly review, test, and merge pending work. Please observe the following guidelines:

- **Limit concurrent pull requests**: As a good rule of thumb, please keep at most **2 pull requests open at any time**. Work through the review process collaboratively with one of the maintainers, and wait for one of your open PRs to be merged or closed before submitting another one. Submitting multiple PRs in a short time span may overwhelm maintainer capacity and result in immediate closure.
- **Review pacing**: Please note that repeatedly pinging maintainers will not speed up the review process, and in extreme cases might lead a reviewer to close your PR without further question. Maintainers review submissions in batches alongside ongoing research deliverables.
- **Focused scope**: Keep each PR focused on a single logical change or bugfix. Avoid bundling unrelated operator additions, refactorings, or formatting changes into a single pull request.

### Submitting a PR

1. **Push** your branch to your GitHub fork:
   ```bash
   git push origin <type>/<descriptive-name>
   ```
2. **Open the PR** against upstream `gpu/alpaka` branch at [https://github.com/ML4EP/SOFIE](https://github.com/ML4EP/SOFIE).
3. **PR Description**:
   - Provide a clear summary of changes.
   - Explain why the change is necessary and its impact.
   - Detail the local testing performed.
   - Link any related issues (`Fixes #<issue>`).
4. **Continuous Integration**:
   - Automated CI runs tests across CPU and GPU environments.
   - Upstream maintainers (`sanjibansg`, `lmoneta`) trigger GPU runners using `/runtest` comments.
   - Address any reviewer feedback promptly.
