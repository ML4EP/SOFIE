#ifndef SOFIE_RQUANTIZATION_BLAS_CUDA_QUANT
#define SOFIE_RQUANTIZATION_BLAS_CUDA_QUANT

// Staging copy of the quantized cuBLASLt entries for sofieBLAS's CUDA backend
// (backends/cuda/sofieBLAS_cublas.hpp), written in BLAS convention.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SOFIE_USE_CUBLASLT
#include <cublasLt.h>
#include <cuda_runtime.h>
#endif

namespace SOFIE {

#ifdef SOFIE_USE_CUBLASLT

// Quantized cuBLASLt matrix multiplies: int8 with int32 accumulation, FP8 E4M3 with explicit
// operand scales. One instance caches one problem per precision until its description changes.
class QuantBlasCuda {
public:
   // Row-major A [m, k] times row-major B [n, k] read as its transpose, into D. The int8-D
   // profile stores output codes directly; without an algorithm for it, D is the int32 accumulator.
   struct Int8MatmulDesc {
      std::size_t m = 0;
      std::size_t n = 0;
      std::size_t k = 0;
      std::size_t batchCount = 1;
      std::int64_t strideA = 0;
      std::int64_t strideB = 0;
      std::int64_t strideC = 0;
      // A supplied column-major [k, m] instead of row-major [m, k] (plain profile only).
      bool aColumnMajor = false;
      // Requested epilogue of the int8-D profile: bias vector add and value Relu.
      bool bias = false;
      bool relu = false;
      std::size_t workspaceLimitBytes = 0;

      bool operator==(const Int8MatmulDesc &other) const
      {
         return m == other.m && n == other.n && k == other.k && batchCount == other.batchCount &&
                strideA == other.strideA && strideB == other.strideB && strideC == other.strideC &&
                aColumnMajor == other.aColumnMajor && bias == other.bias && relu == other.relu &&
                workspaceLimitBytes == other.workspaceLimitBytes;
      }
   };

   // Column-major TN FP8 E4M3 matmul with float accumulation, into D of dType. A scale of 1.0f
   // stays unprogrammed; bias is an optional device BF16 vector on the library epilogue.
   struct Fp8MatmulDesc {
      std::size_t m = 0;
      std::size_t n = 0;
      std::size_t k = 0;
      std::size_t batchCount = 1;
      std::int64_t strideA = 0;
      std::int64_t strideB = 0;
      std::int64_t strideC = 0;
      float aScale = 1.0f;
      float bScale = 1.0f;
      float dScale = 1.0f;
      cudaDataType_t dType = CUDA_R_32F;
      const void *bias = nullptr;
      bool relu = false;
      std::size_t workspaceLimitBytes = 0;

      bool operator==(const Fp8MatmulDesc &other) const
      {
         return m == other.m && n == other.n && k == other.k && batchCount == other.batchCount &&
                strideA == other.strideA && strideB == other.strideB && strideC == other.strideC &&
                aScale == other.aScale && bScale == other.bScale && dScale == other.dScale &&
                dType == other.dType && bias == other.bias &&
                relu == other.relu && workspaceLimitBytes == other.workspaceLimitBytes;
      }
   };

   // Facts about the live problem of one precision, for workspace sizing and reporting.
   struct MatmulStats {
      std::size_t workspaceBytes = 0;         // the selected algorithm's requirement
      std::size_t workspaceCapacityBytes = 0; // largest requirement over all candidates
      int heuristicCount = 0;
      int selectedHeuristic = 0;
      bool autotuned = false;
      float autotuneMs = 0.0f;
      int autotunedCandidates = 0;
      float selectedCandidateMs = 0.0f;
   };

   QuantBlasCuda() = default;
   QuantBlasCuda(const QuantBlasCuda &) = delete;
   QuantBlasCuda &operator=(const QuantBlasCuda &) = delete;
   QuantBlasCuda(QuantBlasCuda &&) noexcept = default;
   QuantBlasCuda &operator=(QuantBlasCuda &&) noexcept = default;
   ~QuantBlasCuda() = default;

   // ---- int8, CUBLAS_COMPUTE_32I ------------------------------------------------------

   // Builds descriptor, layouts, and heuristics; a repeated equal description is free.
   // Returns false (or throws) when neither the int8-D nor the int32 profile has an algorithm.
   bool prepareInt8(const Int8MatmulDesc &desc, bool int8Output, bool throwOnNoAlgorithm = true)
   {
      if (int8Prepared && int8Desc == desc && int8OutputRequested == int8Output)
         return true;
      // A requested int8-D profile the heuristic declines falls back to the accumulator, so a
      // shape without an int8-D algorithm still runs through the plain profile.
      if (!(int8Output && configureInt8(desc, true)) && !configureInt8(desc, false)) {
         if (!throwOnNoAlgorithm) {
            resetInt8();
            return false;
         }
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM found no algorithm for the selected int8 shape");
      }
      int8OutputRequested = int8Output;
      return true;
   }

   // The plain profile with a column-major A; false, with the problem reset, when no algorithm
   // exists. The rejection is sticky, so an ineligible layout is probed once per instance.
   bool tryPrepareInt8ColumnMajorA(Int8MatmulDesc desc)
   {
      if (int8ColumnMajorAUnsupported)
         return false;
      desc.aColumnMajor = true;
      if (int8Prepared && int8Desc == desc && !int8OutputRequested)
         return true;
      if (!configureInt8(desc, false)) {
         int8ColumnMajorAUnsupported = true;
         return false;
      }
      int8OutputRequested = false;
      return true;
   }

   bool int8OutputNarrowed() const { return int8Narrowed; }

   // Whether the live int8 problem reads A column-major, and whether the column-major layout
   // was probed and factually rejected by the provider.
   bool int8ColumnMajorA() const { return int8Prepared && int8Desc.aColumnMajor; }
   bool int8ColumnMajorARejected() const { return int8ColumnMajorAUnsupported; }

   // Runs the prepared problem: a [m, k] row-major (or [k, m] under aColumnMajor), b [n, k]
   // row-major, d the accumulator or int8 output. alpha and dBias act on the int8-D profile only.
   void matmulInt8(cudaStream_t stream, const std::int8_t *a, const std::int8_t *b, void *d,
                   float alpha, const float *dBias, void *workspace, bool autotune,
                   int autotuneIterations)
   {
      if (int8Narrowed && int8Desc.bias) {
         if (dBias == nullptr)
            throw std::runtime_error("SOFIE cuBLASLt quantized GEMM narrowed output requires the bias offset vector");
         check(cublasLtMatmulDescSetAttribute(int8Problem.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                              &dBias, sizeof(dBias)),
               "cublasLtMatmulDescSetAttribute(bias pointer)");
      }
      // The int8-D profile runs the transposed problem, so b leads there.
      const void *first = int8Narrowed ? static_cast<const void *>(b) : static_cast<const void *>(a);
      const void *second = int8Narrowed ? static_cast<const void *>(a) : static_cast<const void *>(b);
      const std::int32_t alphaInt = 1;
      const std::int32_t betaInt = 0;
      const float alphaFloat = alpha;
      const float betaFloat = 0.0f;
      const void *alphaPtr = int8Narrowed ? static_cast<const void *>(&alphaFloat)
                                           : static_cast<const void *>(&alphaInt);
      const void *betaPtr = int8Narrowed ? static_cast<const void *>(&betaFloat)
                                          : static_cast<const void *>(&betaInt);
      auto launch = [&](const cublasLtMatmulAlgo_t &algo) {
         return cublasLtMatmul(int8Problem.handle, int8Problem.operation, alphaPtr, first, int8Problem.aLayout,
                               second, int8Problem.bLayout, betaPtr, d, int8Problem.cLayout,
                               d, int8Problem.dLayout, &algo, workspace,
                               int8Problem.workspaceCapacityBytes, stream);
      };
      autotuneWalk(int8Problem, autotune, autotuneIterations, stream, launch);
      check(launch(int8Problem.heuristic.algo), "cublasLtMatmul");
   }

   const MatmulStats &int8Stats() const { return int8Problem.stats; }

   void resetInt8() noexcept
   {
      int8Problem.reset();
      int8Desc = Int8MatmulDesc{};
      int8Prepared = false;
      int8Narrowed = false;
      int8OutputRequested = false;
   }

   // ---- FP8 E4M3 TN, CUBLAS_COMPUTE_32F -----------------------------------------------

   // Builds descriptor, scale buffer, layouts, and heuristics for the described problem; a
   // repeated call with an equal description is free. Throws when no algorithm exists.
   void prepareFp8(const Fp8MatmulDesc &desc)
   {
      if (fp8Prepared && fp8Desc == desc)
         return;
      resetFp8();
      try {
         check(cublasLtCreate(fp8Problem.handle.Receive()), "cublasLtCreate(FP8)");
         check(cublasLtMatmulDescCreate(fp8Problem.operation.Receive(), CUBLAS_COMPUTE_32F, CUDA_R_32F),
               "cublasLtMatmulDescCreate(FP8)");
         const cublasOperation_t transA = CUBLAS_OP_T;
         const cublasOperation_t transB = CUBLAS_OP_N;
         check(cublasLtMatmulDescSetAttribute(fp8Problem.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                              &transA, sizeof(transA)),
               "cublasLtMatmulDescSetAttribute(FP8 transA)");
         check(cublasLtMatmulDescSetAttribute(fp8Problem.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                              &transB, sizeof(transB)),
               "cublasLtMatmulDescSetAttribute(FP8 transB)");

         // Unit scales are left unprogrammed so an unscaled call keeps the exact operand path.
         if (desc.aScale != 1.0f || desc.bScale != 1.0f || desc.dScale != 1.0f) {
            // Every scale pointer must be 16-byte aligned; cuBLASLt 12.9 rejects base+4
            // packing with NOT_SUPPORTED, so each scalar gets its own 16-byte slot.
            constexpr std::size_t kScaleStride = 16u / sizeof(float);
            float scales[3u * kScaleStride] = {};
            scales[0] = desc.aScale;
            scales[kScaleStride] = desc.bScale;
            scales[2u * kScaleStride] = desc.dScale;
            checkCuda(cudaMalloc(fp8Scales.Receive(), sizeof(scales)), "cudaMalloc(FP8 operand scales)");
            checkCuda(cudaMemcpy(fp8Scales.Get(), scales, sizeof(scales), cudaMemcpyHostToDevice),
                      "cudaMemcpy(FP8 operand scales)");
            float *const aScalePointer = fp8Scales.Get();
            float *const bScalePointer = fp8Scales.Get() + kScaleStride;
            check(cublasLtMatmulDescSetAttribute(fp8Problem.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                 &aScalePointer, sizeof(aScalePointer)),
                  "cublasLtMatmulDescSetAttribute(FP8 A scale)");
            check(cublasLtMatmulDescSetAttribute(fp8Problem.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                 &bScalePointer, sizeof(bScalePointer)),
                  "cublasLtMatmulDescSetAttribute(FP8 B scale)");
            if (desc.dScale != 1.0f) {
               float *const dScalePointer = fp8Scales.Get() + 2u * kScaleStride;
               check(cublasLtMatmulDescSetAttribute(fp8Problem.operation, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                    &dScalePointer, sizeof(dScalePointer)),
                     "cublasLtMatmulDescSetAttribute(FP8 D scale)");
            }
         }

         // The epilogue must be programmed before the heuristic query below: an algorithm
         // chosen without it may not support it.
         if (desc.bias != nullptr) {
            const cublasLtEpilogue_t epilogue =
               desc.relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
            check(cublasLtMatmulDescSetAttribute(fp8Problem.operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                 &epilogue, sizeof(epilogue)),
                  "cublasLtMatmulDescSetAttribute(FP8 bias epilogue)");
            check(cublasLtMatmulDescSetAttribute(fp8Problem.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                 &desc.bias, sizeof(desc.bias)),
                  "cublasLtMatmulDescSetAttribute(FP8 bias pointer)");
            // Only BF16 is accepted here; float32 and fp16 are rejected by the FP8 heuristic.
            const cudaDataType_t biasType = CUDA_R_16BF;
            check(cublasLtMatmulDescSetAttribute(fp8Problem.operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                 &biasType, sizeof(biasType)),
                  "cublasLtMatmulDescSetAttribute(FP8 bias data type)");
         }

         check(cublasLtMatrixLayoutCreate(fp8Problem.aLayout.Receive(), CUDA_R_8F_E4M3,
                                          static_cast<std::uint64_t>(desc.k),
                                          static_cast<std::uint64_t>(desc.m),
                                          static_cast<std::int64_t>(desc.k)),
               "cublasLtMatrixLayoutCreate(FP8 A)");
         check(cublasLtMatrixLayoutCreate(fp8Problem.bLayout.Receive(), CUDA_R_8F_E4M3,
                                          static_cast<std::uint64_t>(desc.k),
                                          static_cast<std::uint64_t>(desc.n),
                                          static_cast<std::int64_t>(desc.k)),
               "cublasLtMatrixLayoutCreate(FP8 B)");
         // cuBLASLt requires a BF16 or FP16 C when D is FP8. beta is 0 here, so C is unread.
         const auto cType = (desc.dType == CUDA_R_8F_E4M3 || desc.dType == CUDA_R_8F_E5M2)
                               ? CUDA_R_16BF
                               : desc.dType;
         check(cublasLtMatrixLayoutCreate(fp8Problem.cLayout.Receive(), cType,
                                          static_cast<std::uint64_t>(desc.m),
                                          static_cast<std::uint64_t>(desc.n),
                                          static_cast<std::int64_t>(desc.m)),
               "cublasLtMatrixLayoutCreate(FP8 C)");
         check(cublasLtMatrixLayoutCreate(fp8Problem.dLayout.Receive(), desc.dType,
                                          static_cast<std::uint64_t>(desc.m),
                                          static_cast<std::uint64_t>(desc.n),
                                          static_cast<std::int64_t>(desc.m)),
               "cublasLtMatrixLayoutCreate(FP8 D)");
         setStridedBatch(fp8Problem.aLayout, desc.batchCount, desc.strideA);
         setStridedBatch(fp8Problem.bLayout, desc.batchCount, desc.strideB);
         setStridedBatch(fp8Problem.cLayout, desc.batchCount, desc.strideC);
         setStridedBatch(fp8Problem.dLayout, desc.batchCount, desc.strideC);

         if (!selectHeuristics(fp8Problem, desc.workspaceLimitBytes, false))
            throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear path found no E4M3 TN algorithm for the requested output profile");

         fp8Desc = desc;
         fp8Prepared = true;
      } catch (...) {
         resetFp8();
         throw;
      }
   }

   // Runs the prepared FP8 problem. A heuristic candidate may still reject the full
   // descriptor at launch, so the candidate list is walked until one runs.
   void matmulFp8(cudaStream_t stream, const void *a, const void *b, void *d, float alpha,
                  void *workspace, bool autotune, int autotuneIterations)
   {
      // cuBLASLt reports an already-errored context as a rejection of this call, so the entry
      // error is peeked, not consumed, to tell an earlier failure from this call's own.
      const cudaError_t entryError = cudaPeekAtLastError();
      const float beta = 0.0f;
      auto launch = [&](const cublasLtMatmulAlgo_t &algo) {
         return cublasLtMatmul(fp8Problem.handle, fp8Problem.operation, &alpha, a, fp8Problem.aLayout,
                               b, fp8Problem.bLayout, &beta, d, fp8Problem.cLayout, d, fp8Problem.dLayout,
                               &algo, workspace, fp8Problem.workspaceCapacityBytes, stream);
      };
      autotuneWalk(fp8Problem, autotune, autotuneIterations, stream, launch);
      // Set SOFIE_FP8_DUMP to print every descriptor, layout, pointer, and residency fact
      // cuBLASLt holds at this call.
      if (const char *tag = std::getenv("SOFIE_FP8_DUMP"))
         dumpFp8(tag, a, b, d, workspace);
      auto status = launch(fp8Problem.heuristic.algo);
      for (int i = 0; status != CUBLAS_STATUS_SUCCESS && i < fp8Problem.stats.heuristicCount; ++i) {
         if (i == fp8Problem.stats.selectedHeuristic)
            continue;
         status = launch(fp8Problem.heuristicResults[i].algo);
         if (status == CUBLAS_STATUS_SUCCESS) {
            fp8Problem.stats.selectedHeuristic = i;
            fp8Problem.heuristic = fp8Problem.heuristicResults[i];
         }
      }
      // The geometry travels with the status: a bare code cannot say which call failed.
      if (status != CUBLAS_STATUS_SUCCESS) {
         const std::string where = "cublasLtMatmul(FP8) m=" + std::to_string(fp8Desc.m) +
                                   " n=" + std::to_string(fp8Desc.n) + " k=" + std::to_string(fp8Desc.k) +
                                   " batch=" + std::to_string(fp8Desc.batchCount) +
                                   " aScale=" + std::to_string(fp8Desc.aScale) +
                                   " bScale=" + std::to_string(fp8Desc.bScale) +
                                   " heuristics=" + std::to_string(fp8Problem.stats.heuristicCount) +
                                   " entryError=" + cudaGetErrorName(entryError) +
                                   " workspace=" + std::to_string(fp8Problem.workspaceCapacityBytes) +
                                   " workspaceAlign=" +
                                   std::to_string(reinterpret_cast<std::uintptr_t>(workspace) % 256u);
         check(status, where.c_str());
      }
   }

   const MatmulStats &fp8Stats() const { return fp8Problem.stats; }

   void resetFp8() noexcept
   {
      fp8Problem.reset();
      fp8Scales.Reset();
      fp8Desc = Fp8MatmulDesc{};
      fp8Prepared = false;
   }

private:
   // Move-only owner of one cuBLASLt/CUDA handle: destroys in the destructor, nulls on move.
   template <typename Handle, auto DestroyFn>
   struct Owned {
      Handle value = nullptr;

      Owned() = default;
      Owned(const Owned &) = delete;
      Owned &operator=(const Owned &) = delete;
      Owned(Owned &&other) noexcept : value(other.value) { other.value = nullptr; }
      Owned &operator=(Owned &&other) noexcept
      {
         if (this != &other) {
            Reset();
            value = other.value;
            other.value = nullptr;
         }
         return *this;
      }
      // Adopts a handle created elsewhere, destroying any current one.
      Owned &operator=(Handle adopted) noexcept
      {
         Reset();
         value = adopted;
         return *this;
      }
      ~Owned() { Reset(); }

      void Reset() noexcept
      {
         if (value != nullptr) {
            DestroyFn(value);
            value = nullptr;
         }
      }
      Handle *Receive()
      {
         Reset();
         return &value;
      }
      Handle Get() const { return value; }
      operator Handle() const { return value; }
   };

   // One recurring matmul problem: descriptor, layouts, heuristic candidates, autotuned
   // selection. Declaration order fixes the reverse teardown: preference, layouts, operation, handle.
   struct Problem {
      Owned<cublasLtHandle_t, cublasLtDestroy> handle;
      Owned<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy> operation;
      Owned<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy> aLayout;
      Owned<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy> bLayout;
      Owned<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy> cLayout;
      Owned<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy> dLayout;
      Owned<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy> preference;
      static constexpr int kMaxHeuristicResults = 8;
      cublasLtMatmulHeuristicResult_t heuristicResults[kMaxHeuristicResults]{};
      cublasLtMatmulHeuristicResult_t heuristic{};
      std::size_t workspaceLimitBytes = 0;
      std::size_t workspaceCapacityBytes = 0;
      MatmulStats stats{};

      void reset() noexcept
      {
         preference.Reset();
         dLayout.Reset();
         cLayout.Reset();
         bLayout.Reset();
         aLayout.Reset();
         operation.Reset();
         handle.Reset();
         for (auto &result : heuristicResults)
            result = cublasLtMatmulHeuristicResult_t{};
         heuristic = cublasLtMatmulHeuristicResult_t{};
         workspaceLimitBytes = 0;
         workspaceCapacityBytes = 0;
         stats = MatmulStats{};
      }
   };

   static void check(cublasStatus_t status, const char *where)
   {
      if (status != CUBLAS_STATUS_SUCCESS) {
         throw std::runtime_error(std::string("SOFIE cuBLASLt quantized GEMM failure in ") + where +
                                  ": status " + std::to_string(static_cast<int>(status)));
      }
   }

   static void checkCuda(cudaError_t status, const char *where)
   {
      if (status != cudaSuccess) {
         throw std::runtime_error(std::string("SOFIE CUDA quantized failure in ") + where + ": " +
                                  cudaGetErrorString(status));
      }
   }

   static cublasLtMatrixLayout_t createColumnMajorLayout(cudaDataType_t type, std::uint64_t rows,
                                                         std::uint64_t cols, std::int64_t leadingDimension)
   {
      cublasLtMatrixLayout_t layout = nullptr;
      check(cublasLtMatrixLayoutCreate(&layout, type, rows, cols, leadingDimension),
            "cublasLtMatrixLayoutCreate");
      return layout;
   }

   static cublasLtMatrixLayout_t createRowMajorLayout(cudaDataType_t type, std::uint64_t rows,
                                                      std::uint64_t cols, std::int64_t leadingDimension)
   {
      cublasLtMatrixLayout_t layout = createColumnMajorLayout(type, rows, cols, leadingDimension);
      const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
      check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
            "cublasLtMatrixLayoutSetAttribute(row-major)");
      return layout;
   }

   static void setStridedBatch(cublasLtMatrixLayout_t layout, std::size_t batchCount,
                               std::int64_t batchStride)
   {
      if (batchCount <= 1)
         return;
      if (batchStride <= 0)
         throw std::runtime_error("SOFIE cuBLASLt strided-batch layouts require a positive matrix stride");
      const auto count = static_cast<std::int32_t>(batchCount);
      check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                             &count, sizeof(count)),
            "cublasLtMatrixLayoutSetAttribute(batch-count)");
      check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &batchStride, sizeof(batchStride)),
            "cublasLtMatrixLayoutSetAttribute(batch-stride)");
   }

   // Selects heuristic candidate 0 and sizes the workspace capacity to the largest candidate.
   // tolerateUnsupported returns false instead of throwing, for callers with a fallback profile.
   static bool selectHeuristics(Problem &problem, std::size_t maxWorkspaceBytes, bool tolerateUnsupported)
   {
      check(cublasLtMatmulPreferenceCreate(problem.preference.Receive()),
            "cublasLtMatmulPreferenceCreate");
      problem.workspaceLimitBytes = maxWorkspaceBytes;
      check(cublasLtMatmulPreferenceSetAttribute(problem.preference,
                                                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &problem.workspaceLimitBytes,
                                                 sizeof(problem.workspaceLimitBytes)),
            "cublasLtMatmulPreferenceSetAttribute(workspace)");
      int resultCount = 0;
      const auto heuristicStatus = cublasLtMatmulAlgoGetHeuristic(
         problem.handle, problem.operation, problem.aLayout, problem.bLayout, problem.cLayout,
         problem.dLayout, problem.preference, Problem::kMaxHeuristicResults,
         problem.heuristicResults, &resultCount);
      if (tolerateUnsupported && heuristicStatus == CUBLAS_STATUS_NOT_SUPPORTED)
         return false;
      check(heuristicStatus, "cublasLtMatmulAlgoGetHeuristic");
      problem.stats.heuristicCount = resultCount;
      if (resultCount == 0)
         return false;
      problem.stats.selectedHeuristic = 0;
      problem.heuristic = problem.heuristicResults[0];
      problem.stats.workspaceBytes = problem.heuristic.workspaceSize;
      for (int i = 0; i < resultCount; ++i) {
         if (problem.heuristicResults[i].workspaceSize > problem.workspaceCapacityBytes)
            problem.workspaceCapacityBytes = problem.heuristicResults[i].workspaceSize;
      }
      problem.stats.workspaceCapacityBytes = problem.workspaceCapacityBytes;
      return true;
   }

   // Warms up and times every heuristic candidate through the caller's launch functor, then
   // selects the fastest. Runs once per prepared problem, when enabled.
   template <typename Launch>
   static void autotuneWalk(Problem &problem, bool enable, int iterations, cudaStream_t stream,
                            const Launch &launch)
   {
      if (problem.stats.autotuned || !enable || problem.stats.heuristicCount <= 1) {
         problem.stats.autotuned = true;
         return;
      }

      cudaEvent_t totalStart = nullptr;
      cudaEvent_t totalStop = nullptr;
      cudaEvent_t candidateStart = nullptr;
      cudaEvent_t candidateStop = nullptr;
      checkCuda(cudaEventCreate(&totalStart), "cudaEventCreate(autotuneTotalStart)");
      checkCuda(cudaEventCreate(&totalStop), "cudaEventCreate(autotuneTotalStop)");
      checkCuda(cudaEventCreate(&candidateStart), "cudaEventCreate(autotuneCandidateStart)");
      checkCuda(cudaEventCreate(&candidateStop), "cudaEventCreate(autotuneCandidateStop)");

      const int rounds = iterations > 0 ? iterations : 1;
      float bestMs = 0.0f;
      int bestIndex = problem.stats.selectedHeuristic;
      int measuredCandidates = 0;
      checkCuda(cudaEventRecord(totalStart, stream), "cudaEventRecord(autotuneTotalStart)");
      for (int i = 0; i < problem.stats.heuristicCount; ++i) {
         const auto warmupStatus = launch(problem.heuristicResults[i].algo);
         if (warmupStatus != CUBLAS_STATUS_SUCCESS)
            continue;
         checkCuda(cudaEventRecord(candidateStart, stream), "cudaEventRecord(autotuneCandidateStart)");
         bool candidateOk = true;
         for (int round = 0; round < rounds; ++round) {
            const auto status = launch(problem.heuristicResults[i].algo);
            if (status != CUBLAS_STATUS_SUCCESS) {
               candidateOk = false;
               break;
            }
         }
         if (!candidateOk)
            continue;
         checkCuda(cudaEventRecord(candidateStop, stream), "cudaEventRecord(autotuneCandidateStop)");
         checkCuda(cudaEventSynchronize(candidateStop), "cudaEventSynchronize(autotuneCandidateStop)");
         float candidateMs = 0.0f;
         checkCuda(cudaEventElapsedTime(&candidateMs, candidateStart, candidateStop),
                   "cudaEventElapsedTime(autotuneCandidate)");
         candidateMs /= static_cast<float>(rounds);
         ++measuredCandidates;
         if (measuredCandidates == 1 || candidateMs < bestMs) {
            bestMs = candidateMs;
            bestIndex = i;
         }
      }
      checkCuda(cudaEventRecord(totalStop, stream), "cudaEventRecord(autotuneTotalStop)");
      checkCuda(cudaEventSynchronize(totalStop), "cudaEventSynchronize(autotuneTotalStop)");
      checkCuda(cudaEventElapsedTime(&problem.stats.autotuneMs, totalStart, totalStop),
                "cudaEventElapsedTime(autotuneTotal)");

      cudaEventDestroy(candidateStop);
      cudaEventDestroy(candidateStart);
      cudaEventDestroy(totalStop);
      cudaEventDestroy(totalStart);

      if (measuredCandidates > 0) {
         problem.stats.selectedHeuristic = bestIndex;
         problem.heuristic = problem.heuristicResults[bestIndex];
         problem.stats.workspaceBytes = problem.heuristic.workspaceSize;
         problem.stats.selectedCandidateMs = bestMs;
         problem.stats.autotunedCandidates = measuredCandidates;
      }
      problem.stats.autotuned = true;
   }

   // Builds descriptor, layouts, and heuristics for one int8 profile (int8 D or int32
   // accumulator); returns false with the problem reset when no algorithm exists for it.
   bool configureInt8(const Int8MatmulDesc &desc, bool int8Output)
   {
      resetInt8();
      try {
         check(cublasLtCreate(int8Problem.handle.Receive()), "cublasLtCreate");
         // The int8-D profile scales in float on the way out, so the descriptor carries a
         // float scale type even though the accumulation stays integer.
         check(cublasLtMatmulDescCreate(int8Problem.operation.Receive(), CUBLAS_COMPUTE_32I,
                                        int8Output ? CUDA_R_32F : CUDA_R_32I),
               "cublasLtMatmulDescCreate");

         // The int8-D profile runs the transposed problem (row-major [m, n] D is column-major
         // [n, m] memory): no bias epilogue exists on a row-major int8 D, and this moves no data.
         const cublasOperation_t transA = int8Output ? CUBLAS_OP_T
                                                     : (desc.aColumnMajor ? CUBLAS_OP_T : CUBLAS_OP_N);
         const cublasOperation_t transB = int8Output ? CUBLAS_OP_N : CUBLAS_OP_T;
         check(cublasLtMatmulDescSetAttribute(int8Problem.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                              &transA, sizeof(transA)),
               "cublasLtMatmulDescSetAttribute(transA)");
         check(cublasLtMatmulDescSetAttribute(int8Problem.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                              &transB, sizeof(transB)),
               "cublasLtMatmulDescSetAttribute(transB)");

         if (int8Output) {
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
            if (desc.bias && desc.relu)
               epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
            else if (desc.bias)
               epilogue = CUBLASLT_EPILOGUE_BIAS;
            else if (desc.relu)
               epilogue = CUBLASLT_EPILOGUE_RELU;
            if (epilogue != CUBLASLT_EPILOGUE_DEFAULT)
               check(cublasLtMatmulDescSetAttribute(int8Problem.operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &epilogue, sizeof(epilogue)),
                     "cublasLtMatmulDescSetAttribute(epilogue)");
            if (desc.bias) {
               // Only float32 is accepted alongside an int8 D.
               const cudaDataType_t biasType = CUDA_R_32F;
               check(cublasLtMatmulDescSetAttribute(int8Problem.operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                    &biasType, sizeof(biasType)),
                     "cublasLtMatmulDescSetAttribute(bias data type)");
            }
         }

         const cudaDataType_t outputType = int8Output ? CUDA_R_8I : CUDA_R_32I;
         if (int8Output) {
            // Row-major [n, k] B and [m, k] A read column-major as [k, n] and [k, m]; the
            // first operand is B, so D comes out column-major [n, m].
            int8Problem.aLayout = createColumnMajorLayout(CUDA_R_8I, desc.k, desc.n,
                                                     static_cast<std::int64_t>(desc.k));
            int8Problem.bLayout = createColumnMajorLayout(CUDA_R_8I, desc.k, desc.m,
                                                     static_cast<std::int64_t>(desc.k));
            int8Problem.cLayout = createColumnMajorLayout(outputType, desc.n, desc.m,
                                                     static_cast<std::int64_t>(desc.n));
            int8Problem.dLayout = createColumnMajorLayout(outputType, desc.n, desc.m,
                                                     static_cast<std::int64_t>(desc.n));
            setStridedBatch(int8Problem.aLayout, desc.batchCount, desc.strideB);
            setStridedBatch(int8Problem.bLayout, desc.batchCount, desc.strideA);
         } else {
            // Column-major [m, k] A is described to the row-major convention as its
            // transpose: a [k, m] row-major matrix with leading dimension m.
            int8Problem.aLayout = desc.aColumnMajor
                                ? createRowMajorLayout(CUDA_R_8I, desc.k, desc.m,
                                                       static_cast<std::int64_t>(desc.m))
                                : createRowMajorLayout(CUDA_R_8I, desc.m, desc.k,
                                                       static_cast<std::int64_t>(desc.k));
            int8Problem.bLayout = createRowMajorLayout(CUDA_R_8I, desc.n, desc.k,
                                                  static_cast<std::int64_t>(desc.k));
            // C goes unread at beta 0 and carries D's type so the provider accepts the pair.
            int8Problem.cLayout = createRowMajorLayout(outputType, desc.m, desc.n,
                                                  static_cast<std::int64_t>(desc.n));
            int8Problem.dLayout = createRowMajorLayout(outputType, desc.m, desc.n,
                                                  static_cast<std::int64_t>(desc.n));
            setStridedBatch(int8Problem.aLayout, desc.batchCount, desc.strideA);
            setStridedBatch(int8Problem.bLayout, desc.batchCount, desc.strideB);
         }
         setStridedBatch(int8Problem.cLayout, desc.batchCount, desc.strideC);
         setStridedBatch(int8Problem.dLayout, desc.batchCount, desc.strideC);

         if (!selectHeuristics(int8Problem, desc.workspaceLimitBytes, int8Output)) {
            resetInt8();
            return false;
         }

         int8Desc = desc;
         int8Narrowed = int8Output;
         int8Prepared = true;
      } catch (...) {
         resetInt8();
         throw;
      }
      return true;
   }

   // Prints every descriptor and layout attribute cuBLASLt holds, unconditionally and in a
   // fixed order so two dumps diff as text.
   void dumpFp8(const char *tag, const void *a, const void *b, const void *d, const void *workspace) const
   {
      auto descInt = [&](const char *name, cublasLtMatmulDescAttributes_t attr) {
         std::int32_t value = -1;
         std::size_t written = 0;
         const auto status = cublasLtMatmulDescGetAttribute(fp8Problem.operation, attr, &value, sizeof(value), &written);
         std::printf("  desc.%-22s = %-12d (status %d)\n", name, static_cast<int>(value), static_cast<int>(status));
      };
      auto descPtr = [&](const char *name, cublasLtMatmulDescAttributes_t attr) {
         void *value = nullptr;
         std::size_t written = 0;
         const auto status = cublasLtMatmulDescGetAttribute(fp8Problem.operation, attr, &value, sizeof(value), &written);
         std::printf("  desc.%-22s = %-12p (status %d)\n", name, value, static_cast<int>(status));
      };
      auto layout = [&](const char *name, cublasLtMatrixLayout_t handle) {
         std::int32_t type = -1, order = -1, batch = -1;
         std::uint64_t rows = 0, cols = 0;
         std::int64_t ld = 0, stride = 0;
         std::size_t written = 0;
         cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_TYPE, &type, sizeof(type), &written);
         cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order), &written);
         cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows), &written);
         cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols), &written);
         cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &written);
         cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch), &written);
         cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                                          sizeof(stride), &written);
         std::printf("  layout.%-2s type=%-3d order=%-2d rows=%-6llu cols=%-6llu ld=%-6lld batch=%-4d stride=%lld\n",
                     name, static_cast<int>(type), static_cast<int>(order),
                     static_cast<unsigned long long>(rows), static_cast<unsigned long long>(cols),
                     static_cast<long long>(ld), static_cast<int>(batch), static_cast<long long>(stride));
      };

      std::printf("[FP8 dump %s] m=%zu n=%zu k=%zu batch=%zu\n", tag, fp8Desc.m, fp8Desc.n, fp8Desc.k,
                  fp8Desc.batchCount);
      descInt("TRANSA", CUBLASLT_MATMUL_DESC_TRANSA);
      descInt("TRANSB", CUBLASLT_MATMUL_DESC_TRANSB);
      descInt("COMPUTE_TYPE", CUBLASLT_MATMUL_DESC_COMPUTE_TYPE);
      descInt("SCALE_TYPE", CUBLASLT_MATMUL_DESC_SCALE_TYPE);
      descInt("POINTER_MODE", CUBLASLT_MATMUL_DESC_POINTER_MODE);
      descInt("EPILOGUE", CUBLASLT_MATMUL_DESC_EPILOGUE);
      descInt("FAST_ACCUM", CUBLASLT_MATMUL_DESC_FAST_ACCUM);
      descPtr("A_SCALE_POINTER", CUBLASLT_MATMUL_DESC_A_SCALE_POINTER);
      descPtr("B_SCALE_POINTER", CUBLASLT_MATMUL_DESC_B_SCALE_POINTER);
      descPtr("C_SCALE_POINTER", CUBLASLT_MATMUL_DESC_C_SCALE_POINTER);
      descPtr("D_SCALE_POINTER", CUBLASLT_MATMUL_DESC_D_SCALE_POINTER);
      descPtr("BIAS_POINTER", CUBLASLT_MATMUL_DESC_BIAS_POINTER);
      layout("A", fp8Problem.aLayout);
      layout("B", fp8Problem.bLayout);
      layout("C", fp8Problem.cLayout);
      layout("D", fp8Problem.dLayout);
      std::printf("  ptr.A=%p (mod16 %zu)  ptr.B=%p (mod16 %zu)  ptr.D=%p (mod16 %zu)\n", a,
                  reinterpret_cast<std::uintptr_t>(a) % 16u, b,
                  reinterpret_cast<std::uintptr_t>(b) % 16u, d,
                  reinterpret_cast<std::uintptr_t>(d) % 16u);
      std::printf("  workspace=%p bytes=%zu  heuristics=%d  scaleBuffer=%p\n", workspace,
                  fp8Problem.workspaceCapacityBytes, fp8Problem.stats.heuristicCount,
                  static_cast<const void *>(fp8Scales.Get()));
      // cuBLASLt binds a handle to the device current at creation and rejects operands living
      // on another one; that is invisible in the attributes above, so residency is asked directly.
      int currentDevice = -1;
      cudaGetDevice(&currentDevice);
      auto residency = [](const char *name, const void *pointer) {
         cudaPointerAttributes attributes{};
         const auto status = cudaPointerGetAttributes(&attributes, pointer);
         std::printf("  residency.%-9s device=%-3d type=%-2d (status %d)\n", name, attributes.device,
                     static_cast<int>(attributes.type), static_cast<int>(status));
      };
      std::printf("  currentDevice=%d\n", currentDevice);
      residency("A", a);
      residency("B", b);
      residency("D", d);
      residency("workspace", workspace);
      residency("scales", fp8Scales.Get());
      std::fflush(stdout);
   }

   Problem int8Problem;
   Int8MatmulDesc int8Desc{};
   bool int8Prepared = false;
   bool int8Narrowed = false;
   bool int8OutputRequested = false;
   bool int8ColumnMajorAUnsupported = false;

   Problem fp8Problem;
   Fp8MatmulDesc fp8Desc{};
   bool fp8Prepared = false;
   Owned<float *, cudaFree> fp8Scales;
};

#endif // SOFIE_USE_CUBLASLT

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_BLAS_CUDA_QUANT
