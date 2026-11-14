# Contributing to SOFIE

Thank you for your interest in contributing to **SOFIE** (System for Optimized Fast Inference code Emit)! SOFIE is developed within the ML4EP Project of the EP-SFT Group at CERN to generate fast, self-contained inference code for deep learning models across CPUs and heterogeneous GPU architectures via [alpaka](https://github.com/alpaka-group/alpaka).

We welcome contributions of all kinds, including new operator implementations, GPU kernel optimizations, bug fixes, and documentation improvements.

---

## Getting Started

### 1. Prerequisites

- **Core & CPU Tests**: **CMake** >= 3.21, a **C++20-capable compiler** (GCC >= 11, Clang >= 14), and GoogleTest (fetched or system-installed).
- **GPU / Portability Tests (`ENABLE_ALPAKA_TESTS=ON`)**: Requires the **alpaka** portability library and the target backend toolkit (**CUDA Toolkit** >= 11.8, **ROCm/HIP** >= 5.0, or OpenMP/TBB).
- **Test Model Generation (Optional)**: Python 3 with `onnx`, `torch`, and `numpy` when generating or updating ONNX test models.

### 2. Branching Strategy

- Active development targeting heterogeneous GPU / alpaka features is based on the **`gpu/alpaka`** branch.
- Branch off the latest `upstream/gpu/alpaka` using a short, descriptive branch name aligned with the [Conventional Commits](https://www.conventionalcommits.org/) categories.

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
  - Keep indentation and bracing consistent with the surrounding codebase and preserve existing docstrings and comments.

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages and PR titles.

### AI-Assisted Contributions Policy

Non-maintainers who use AI or LLM-based coding assistants to draft or assist with code contributions must adhere to the following guidelines:

- **Explicit Declaration**: Any AI assistance must be explicitly declared in your pull request description and commit trailers (e.g., `Assisted-by: <tool-name>`). Transparency helps maintainers evaluate review depth, edge cases, and architectural compliance.
- **Author Responsibility**: Contributors are fully responsible for the correctness, maintainability, and safety of every line of code submitted. Do not submit AI-generated code without thoroughly reading, understanding, and testing it locally.
- **Architectural & Style Adherence**: Code must strictly conform to SOFIE's C++20 conventions, alpaka kernel execution model, memory layout rules, and existing error-handling patterns. Boilerplate or unverified LLM output will be closed.

---

## Testing

Please build and run the relevant local tests before opening a pull request. See the [Testing](README.md#testing) section in `README.md` for the CMake preset and build commands (`cpu-tests`, `cuda-tests`).

### Writing New Tests

- Unit tests for general operator support and code emission: `tests/cpu/TestAlpakaOperatorSupport.cxx` or `tests/cpu/TestSofieModels.cxx`.
- Alpaka GPU integration tests: `tests/alpaka/`.
- Use GoogleTest (`TEST(...)` / `EXPECT_*` / `ASSERT_*`) macros.

---

## Opening a Pull Request

### Pull Request Limits & Review Policy

Inspired by [ROOT's contributor policy](https://github.com/root-project/root/pull/24067), SOFIE values the time you invest to contribute while giving maintainers room to thoroughly review, test, and merge pending work:

- **Limit concurrent pull requests**: Please keep at most **2 pull requests open at any time**. Work through the review process collaboratively with the maintainers, and wait for one of your open PRs to be merged or closed before submitting another one.
- **Review pacing**: Repeatedly pinging maintainers will not speed up the review process. Maintainers review submissions in batches alongside ongoing research deliverables.
- **Focused scope**: Keep each PR focused on a single logical change or bugfix. Avoid bundling unrelated operator additions, refactorings, or formatting changes into one pull request.

### PR Checklist & CI

- Target the `gpu/alpaka` branch and fill out the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
- GPU tests run on self-hosted ML4EP runners (`ml4ep-l40s`, `ml4ep-h100`, `ml4ep-w7900`, `ml4ep-mi300x`) when triggered by maintainers (`sanjibansg`, `lmoneta`) via `/runtest` comments, so please validate CPU tests locally first.
