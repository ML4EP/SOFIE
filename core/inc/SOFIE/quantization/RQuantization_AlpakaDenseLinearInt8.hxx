#ifndef SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_INT8
#define SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_INT8

// The int8 dense-linear path: input quantization and padding kernels, the epilogue value/code
// arithmetic and its kernels, and the cuBLASLt state and call that drive them.

#include "SOFIE/quantization/RQuantization_AlpakaDenseLinearCommon.hxx"

namespace SOFIE {

#ifndef SOFIE_USE_CUBLASLT

// Generated code emits BindScratch and WorkspaceSize unconditionally, so the stub
// carries both as no-ops.
struct QuantizedGemmCudaLtState : QuantizedDeferredEpilogueHolder {
   void BindScratch(QuantizedCudaScratchView) {}
   std::size_t WorkspaceSize() const { return 0; }
};

inline void QuantizedGemmCudaLt_Call(QuantizedGemmCudaLtState &, QuantizedGemmCudaStream, void *, const void *,
                                     const void *, const float *, const float *, const QuantizedDenseLinearInvocation &)
{
   throw std::runtime_error("SOFIE cuBLASLt quantized GEMM path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

#else

namespace INTERNAL {

__global__ void QuantizedGemmCudaQuantizeInputKernel(const float *input, std::int8_t *inputQuantized,
                                                     std::size_t elements, double scale, std::int32_t zero,
                                                     std::int32_t qmin, std::int32_t qmax)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;
   inputQuantized[idx] = static_cast<std::int8_t>(QuantizedCudaQuantizeClamp(static_cast<double>(input[idx]), scale,
                                                                              zero, qmin, qmax));
}

__global__ void QuantizedGemmCudaPadInt8MatrixKernel(const std::int8_t *__restrict__ input,
                                                     std::int8_t *__restrict__ padded,
                                                     std::size_t logicalRows, std::size_t logicalCols,
                                                     std::size_t physicalRows, std::size_t physicalCols,
                                                     std::int8_t padValue)
{
   const std::size_t elements = physicalRows * physicalCols;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;
   const std::size_t row = idx / physicalCols;
   const std::size_t col = idx % physicalCols;
   padded[idx] = (row < logicalRows && col < logicalCols) ? input[row * logicalCols + col] : padValue;
}
__global__ void QuantizedGemmCudaBiasOutputOffsetKernel(float *__restrict__ biasOutputOffset,
                                                        const float *__restrict__ bias,
                                                        const float *__restrict__ weightScaleVector,
                                                        QuantizedDenseLinearInvocation params)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= params.n)
      return;

   const float biasScale = (params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel)
                              ? static_cast<float>(params.inputScale * static_cast<double>(weightScaleVector[idx]))
                              : static_cast<float>(params.biasScale);
   int bq = __float2int_rn(bias[idx] / biasScale) + params.biasZeroPoint;
   bq = QuantizedCudaClamp(bq, params.biasQMin, params.biasQMax);
   biasOutputOffset[idx] = static_cast<float>(params.beta) * static_cast<float>(bq - params.biasZeroPoint) *
                           biasScale / static_cast<float>(params.outputScale);
}
// Per-output-channel sums of the int8 weight rows in [N, K] storage. An input at zero
// point zp contributes zp * sum_k(w[n][k]) to every s8xs8 dot product in column n.
__global__ void QuantizedGemmCudaWeightColumnSumKernel(std::int32_t *__restrict__ columnSums,
                                                       const std::int8_t *__restrict__ weight,
                                                       std::size_t outputChannels,
                                                       std::size_t reduceLength)
{
   const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (n >= outputChannels)
      return;
   const std::int8_t *row = weight + n * reduceLength;
   std::int32_t sum = 0;
   for (std::size_t k = 0; k < reduceLength; ++k)
      sum += row[k];
   columnSums[n] = sum;
}

// Accumulator value with the asymmetric-input correction removed per column; a null
// column-sum pointer is the symmetric case and returns the value unchanged.
__host__ __device__ inline double QuantizedGemmCudaCorrectedAccumulator(
   std::int32_t accumulatorValue, std::size_t col, const std::int32_t *inputZeroPointColumnSums,
   const QuantizedDenseLinearInvocation &params)
{
   if (inputZeroPointColumnSums == nullptr)
      return static_cast<double>(accumulatorValue);
   const long long corrected = static_cast<long long>(accumulatorValue) -
                               static_cast<long long>(params.inputZeroPoint) *
                                  static_cast<long long>(inputZeroPointColumnSums[col]);
   return static_cast<double>(corrected);
}

// Quantized-epilogue core shared by the dense-linear and Conv integer epilogues: fused
// multiply-add onto the output grid, Relu on the code, then the range clamp.
template <bool HasRelu>
__device__ inline std::int32_t QuantizedCudaFmaReluClampCode(float accumulatorValue, float scale,
                                                             float offset, std::int32_t outputZeroPoint,
                                                             std::int32_t outputQMin,
                                                             std::int32_t outputQMax)
{
   int code = __float2int_rn(__fmaf_rn(accumulatorValue, scale, offset)) + outputZeroPoint;
   if constexpr (HasRelu) {
      if (code < outputZeroPoint)
         code = outputZeroPoint;
   }
   return QuantizedCudaClamp(code, outputQMin, outputQMax);
}

template <typename OutputT, bool HasBias, bool HasRelu>
__global__ void QuantizedGemmCudaQuantizedEpilogueKernel(OutputT *__restrict__ output,
                                                        const std::int32_t *__restrict__ accumulator,
                                                        const float *__restrict__ biasOutputOffset,
                                                        const float *__restrict__ weightScaleVector,
                                                        const std::int32_t *__restrict__ inputZeroPointColumnSums,
                                                        QuantizedDenseLinearInvocation params)
{
   // Whole batch, not one slice. The accumulator is contiguous [batch][m][n], so idx and
   // the column lookup carry over.
   const std::size_t elements = (params.batchCount == 0 ? 1 : params.batchCount) * params.m * params.n;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;

   const std::size_t col = idx % params.n;
   const float scale = (params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel)
                         ? static_cast<float>((params.alpha * params.inputScale * static_cast<double>(weightScaleVector[col])) / params.outputScale)
                         : static_cast<float>(params.accumulatorToOutputScale);
   const float offset = HasBias ? biasOutputOffset[col] : 0.0f;
   const std::int32_t yq = QuantizedCudaFmaReluClampCode<HasRelu>(
      static_cast<float>(
         QuantizedGemmCudaCorrectedAccumulator(accumulator[idx], col, inputZeroPointColumnSums, params)),
      scale, offset, params.outputZeroPoint, params.outputQMin, params.outputQMax);
   output[idx] = static_cast<OutputT>(yq);
}

// Logical output extent over the whole strided batch: a padded GEMM keeps its physical row
// stride in params.n, so the epilogue slices [logicalM, logicalN] out of the accumulator.
__device__ inline std::size_t QuantizedGemmCudaLogicalElements(const QuantizedDenseLinearInvocation &params)
{
   const std::size_t cols = params.logicalN != 0 ? params.logicalN : params.n;
   const std::size_t rows = params.logicalM != 0 ? params.logicalM : params.m;
   const std::size_t batch = params.batchCount == 0 ? 1 : params.batchCount;
   return batch * rows * cols;
}

// Epilogue shape decisions as runtime fields or compile-time constants. One implementation
// reads both through this policy, so the standalone and inlined forms cannot drift apart.
struct QuantizedEpilogueRuntimeFlags {
   bool hasBias = false;
   bool hasRelu = false;
   bool perChannelScale = false;
   bool correctZeroPoint = false;
   // Never folded on the generic path: only the planner establishes the power-of-two chain.
   static constexpr bool fusedAccumulatorScale = false;
};

template <bool HasBias, bool HasRelu, bool PerChannelScale, bool CorrectZeroPoint,
          bool FusedAccumulatorScale>
struct QuantizedEpilogueStaticFlags {
   static constexpr bool hasBias = HasBias;
   static constexpr bool hasRelu = HasRelu;
   static constexpr bool perChannelScale = PerChannelScale;
   static constexpr bool correctZeroPoint = CorrectZeroPoint;
   static constexpr bool fusedAccumulatorScale = FusedAccumulatorScale;
};

// A null bias folds into hasBias and null column sums into correctZeroPoint, so this is the one
// place either pointer is tested; RequireDeferredEpilogueSpecialization derives them the same way.
__host__ __device__ inline QuantizedEpilogueRuntimeFlags
QuantizedEpilogueFlagsOf(const QuantizedDenseLinearInvocation &params, const float *bias,
                         const std::int32_t *inputZeroPointColumnSums)
{
   QuantizedEpilogueRuntimeFlags flags;
   flags.hasBias = params.hasBias && bias != nullptr;
   flags.hasRelu = params.hasRelu;
   flags.perChannelScale = params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel;
   flags.correctZeroPoint = inputZeroPointColumnSums != nullptr;
   return flags;
}

// Fake-quant output code of one accumulator element: alpha * acc * inputScale * weightScale
// plus the grid-round-tripped bias, rounded onto the output grid with the Relu applied there.
template <typename Flags>
__host__ __device__ inline std::int32_t QuantizedCudaFakeQuantOutputCodeWith(
   const Flags &flags, double accumulatorValue, const float *bias, std::size_t channel,
   double weightScale, const QuantizedDenseLinearInvocation &params)
{
   // The whole scale chain in one host-precomputed multiply, exact only for a power-of-two
   // chain. `if constexpr` so the static_assert skips the arms that never fold.
   if constexpr (Flags::fusedAccumulatorScale) {
      static_assert(!Flags::hasBias && !Flags::perChannelScale,
                    "a fused accumulator scale cannot carry a bias term or a per-column scale");
      auto code = QuantizedCudaClamp(
         static_cast<std::int32_t>(nearbyint(accumulatorValue * params.accumulatorToOutputScale) +
                                   static_cast<double>(params.outputZeroPoint)),
         params.outputQMin, params.outputQMax);
      if (flags.hasRelu && code < params.outputZeroPoint)
         code = params.outputZeroPoint;
      return code;
   } else {
      double real = params.alpha * accumulatorValue * params.inputScale * weightScale;
      if (flags.hasBias) {
         const double biasScale =
            flags.perChannelScale ? params.inputScale * weightScale : params.biasScale;
         const auto bq = QuantizedCudaQuantizeClamp(static_cast<double>(bias[channel]), biasScale,
                                                    params.biasZeroPoint, params.biasQMin, params.biasQMax);
         real += params.beta * static_cast<double>(bq - params.biasZeroPoint) * biasScale;
      }

      // Recip when the host filled it: its residual rides an fma so ties land as the divide.
      auto code = params.outputScaleReciprocal != 0.0
                     ? QuantizedCudaQuantizeClampRecip(real, params.outputScaleReciprocal,
                                                       params.outputScaleReciprocalError,
                                                       params.outputZeroPoint, params.outputQMin,
                                                       params.outputQMax)
                     : QuantizedCudaQuantizeClamp(real, params.outputScale, params.outputZeroPoint,
                                                  params.outputQMin, params.outputQMax);
      if (flags.hasRelu && code < params.outputZeroPoint)
         code = params.outputZeroPoint;
      return code;
   }
}

// Fake-quant value of one logical output element: scale, bias, round onto the output grid with
// Relu, dequantize. __host__ __device__ so a deferring consumer applies the identical function.
template <typename Flags>
__host__ __device__ inline float QuantizedGemmCudaEpilogueValueWith(
   const Flags &flags, std::size_t idx, const std::int32_t *accumulator, const float *bias,
   const float *weightScaleVector, const std::int32_t *inputZeroPointColumnSums,
   const QuantizedDenseLinearInvocation &params)
{
   const std::size_t logicalCols = params.logicalN != 0 ? params.logicalN : params.n;
   const std::size_t col = idx % logicalCols;
   const std::size_t row = idx / logicalCols;
   const std::size_t accIdx = row * params.n + col;
   const double weightScale =
      flags.perChannelScale ? static_cast<double>(weightScaleVector[col]) : params.weightScale;
   // Equivalent to calling the corrector unconditionally: it is a no-op on a null pointer.
   const double corrected =
      flags.correctZeroPoint
         ? QuantizedGemmCudaCorrectedAccumulator(accumulator[accIdx], col, inputZeroPointColumnSums, params)
         : static_cast<double>(accumulator[accIdx]);
   auto yq = QuantizedCudaFakeQuantOutputCodeWith(flags, corrected, bias, col, weightScale, params);
   yq = QuantizedCudaClamp(yq, params.outputQMin, params.outputQMax);
   return static_cast<float>(static_cast<double>(yq - params.outputZeroPoint) * params.outputScale);
}

__host__ __device__ inline std::int32_t QuantizedCudaFakeQuantOutputCode(
   double accumulatorValue, const float *bias, std::size_t channel, double weightScale,
   const QuantizedDenseLinearInvocation &params)
{
   // No column sums reach this entry point; the caller applies the correction before it.
   return QuantizedCudaFakeQuantOutputCodeWith(QuantizedEpilogueFlagsOf(params, bias, nullptr),
                                               accumulatorValue, bias, channel, weightScale, params);
}

__host__ __device__ inline float QuantizedGemmCudaEpilogueValue(
   std::size_t idx, const std::int32_t *accumulator, const float *bias, const float *weightScaleVector,
   const std::int32_t *inputZeroPointColumnSums, const QuantizedDenseLinearInvocation &params)
{
   return QuantizedGemmCudaEpilogueValueWith(
      QuantizedEpilogueFlagsOf(params, bias, inputZeroPointColumnSums), idx, accumulator, bias,
      weightScaleVector, inputZeroPointColumnSums, params);
}

// The same evaluation with its shape decisions as template parameters, so untaken branches are
// dropped. A wrong specialization is silently wrong, so the launch checks it on the host.
template <bool HasBias, bool HasRelu, bool PerChannelScale, bool CorrectZeroPoint,
          bool FusedAccumulatorScale = false>
__host__ __device__ inline std::int32_t QuantizedCudaFakeQuantOutputCodeSpec(
   double accumulatorValue, const float *bias, std::size_t channel, double weightScale,
   const QuantizedDenseLinearInvocation &params)
{
   return QuantizedCudaFakeQuantOutputCodeWith(
      QuantizedEpilogueStaticFlags<HasBias, HasRelu, PerChannelScale, CorrectZeroPoint,
                                   FusedAccumulatorScale>{},
      accumulatorValue, bias, channel, weightScale, params);
}

template <bool HasBias, bool HasRelu, bool PerChannelScale, bool CorrectZeroPoint,
          bool FusedAccumulatorScale = false>
__host__ __device__ inline float QuantizedGemmCudaEpilogueValueSpec(
   std::size_t idx, const std::int32_t *accumulator, const float *bias, const float *weightScaleVector,
   const std::int32_t *inputZeroPointColumnSums, const QuantizedDenseLinearInvocation &params)
{
   return QuantizedGemmCudaEpilogueValueWith(
      QuantizedEpilogueStaticFlags<HasBias, HasRelu, PerChannelScale, CorrectZeroPoint,
                                   FusedAccumulatorScale>{},
      idx, accumulator, bias, weightScaleVector, inputZeroPointColumnSums, params);
}

__global__ void QuantizedGemmCudaEpilogueKernel(float *output, const std::int32_t *accumulator, const float *bias,
                                                const float *weightScaleVector,
                                                const std::int32_t *inputZeroPointColumnSums,
                                                QuantizedDenseLinearInvocation params)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= QuantizedGemmCudaLogicalElements(params))
      return;
   output[idx] =
      QuantizedGemmCudaEpilogueValue(idx, accumulator, bias, weightScaleVector, inputZeroPointColumnSums, params);
}

// Re-quantizes the same fake-quant value onto the consuming region's input grid and stores an
// int8 carrier, collapsing epilogue, Relu and the consumer's input-quantize into one pass.
__global__ void QuantizedGemmCudaFusedRequantizeEpilogueKernel(std::int8_t *output,
                                                              const std::int32_t *accumulator, const float *bias,
                                                              const float *weightScaleVector,
                                                              const std::int32_t *inputZeroPointColumnSums,
                                                              QuantizedDenseLinearInvocation params)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= QuantizedGemmCudaLogicalElements(params))
      return;
   const float value =
      QuantizedGemmCudaEpilogueValue(idx, accumulator, bias, weightScaleVector, inputZeroPointColumnSums, params);
   output[idx] = static_cast<std::int8_t>(QuantizedCudaQuantizeClamp(
      static_cast<double>(value), params.requantizeScale, params.requantizeZeroPoint,
      params.requantizeQMin, params.requantizeQMax));
}

} // namespace INTERNAL

inline void QuantizedGemmCudaPadInt8Matrix(QuantizedGemmCudaStream stream, const std::int8_t *input,
                                           std::int8_t *padded, std::size_t logicalRows,
                                           std::size_t logicalCols, std::size_t physicalRows,
                                           std::size_t physicalCols, std::int8_t padValue = 0)
{
   const std::size_t elements = physicalRows * physicalCols;
   if (elements == 0)
      return;
   constexpr int blockSize = 256;
   const int gridSize = static_cast<int>((elements + blockSize - 1) / blockSize);
   INTERNAL::QuantizedGemmCudaPadInt8MatrixKernel<<<gridSize, blockSize, 0, stream>>>(
      input, padded, logicalRows, logicalCols, physicalRows, physicalCols, padValue);
   INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaPadInt8MatrixKernel");
}

inline void QuantizedGemmCudaUnpadInt8Matrix(QuantizedGemmCudaStream stream, const std::int8_t *padded,
                                             std::int8_t *output, std::size_t logicalRows,
                                             std::size_t logicalCols, std::size_t physicalRows,
                                             std::size_t physicalCols)
{
   QuantizedGemmCudaUnpadMatrix(stream, padded, output, logicalRows, logicalCols, physicalRows, physicalCols);
}

inline void QuantizedGemmCudaUnpadUInt8Matrix(QuantizedGemmCudaStream stream, const std::uint8_t *padded,
                                              std::uint8_t *output, std::size_t logicalRows,
                                              std::size_t logicalCols, std::size_t physicalRows,
                                              std::size_t physicalCols)
{
   QuantizedGemmCudaUnpadMatrix(stream, padded, output, logicalRows, logicalCols, physicalRows, physicalCols);
}

// Whether the integer epilogue can ride the GEMM's own narrowing store: cuBLASLt writes an int8
// D with round-half-to-even and saturation, leaving only alpha and the per-column bias offset.
inline bool QuantizedGemmCudaLt_NarrowsQuantizedOutput(const QuantizedDenseLinearInvocation &params)
{
   // Forces the readback epilogue, so one binary can run either output path.
   // Read once, off the per-call path.
   static const bool forceReadbackEpilogue = std::getenv("SOFIE_INT8_NO_NARROWED_D") != nullptr;
   if (forceReadbackEpilogue)
      return false;
   if (params.epilogueMode != EQuantizedEpilogueMode::Quantized)
      return false;
   // The narrowing store exists for a signed D only.
   if (params.outputCarrier != EQuantizedOutputCarrier::Int8)
      return false;
   // A per-output-channel weight scale turns alpha into a per-row vector, which the provider
   // rejects on this path.
   if (params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel)
      return false;
   // The output zero point shifts the code after the rounding; carrying it in the offset the
   // store rounds would move half-way ties by its parity.
   if (params.outputZeroPoint != 0)
      return false;
   // The asymmetric-input correction subtracts a per-column multiple of the weight column sums
   // from the accumulator, which a narrowing store never materializes.
   if (params.inputZeroPoint != 0)
      return false;
   // The store saturates to the whole int8 grid; a narrower code range needs the readback pass.
   if (params.outputQMin != -128 || params.outputQMax != 127)
      return false;
   // The transposed descriptor a narrowed store runs reads the input as row-major [m, k]; the
   // Conv direct-input layout stores it the other way round.
   if (params.aColumnMajorInput)
      return false;
   return true;
}

struct QuantizedGemmCudaLtState : QuantizedDeferredEpilogueHolder {
   // The cuBLASLt problem lives in the provider; this state keeps the session-side staging,
   // the asymmetric-input column sums, and the tiled-Conv pipeline.
   QuantBlasCuda fBlas;
   QuantizedCudaScratchView fScratch{};
   void *fWorkspace = nullptr;
   std::int8_t *fInputQuantized = nullptr;
   std::int32_t *fAccumulator = nullptr;
   void *fOutputQuantized = nullptr;
   float *fBiasOutputOffset = nullptr;
   std::size_t fInputQuantizedBytes = 0;
   std::size_t fAccumulatorBytes = 0;
   std::size_t fOutputQuantizedBytes = 0;
   std::size_t fBiasOutputOffsetBytes = 0;
   // Per-output-channel weight sums for the asymmetric-input correction; built once and
   // surviving Reset(), since the weight bound to this state does not change.
   INTERNAL::QuantizedCudaDeviceBuffer<std::int32_t> fInputZpColumnSums;
   bool fInputZpColumnSumsBuilt = false;
   // Tiled-Conv staging pipeline: an internal stream and dependency events overlap the next
   // tile's im2col staging with the current tile's GEMM and epilogue. Reset() keeps them alive.
   INTERNAL::QuantizedCudaOwnedStream fTileStagingStream;
   INTERNAL::QuantizedCudaOwnedEvent fTileEntryEvent;
   INTERNAL::QuantizedCudaOwnedEvent fTileStagingDoneEvents[2];
   INTERNAL::QuantizedCudaOwnedEvent fTileComputeDoneEvents[2];

   void EnsureTilePipeline()
   {
      if (fTileStagingStream.Get() != nullptr)
         return;
      INTERNAL::CheckCudaStatus(
         cudaStreamCreateWithFlags(fTileStagingStream.Receive(), cudaStreamNonBlocking),
         "cudaStreamCreateWithFlags(tile staging)");
      INTERNAL::CheckCudaStatus(
         cudaEventCreateWithFlags(fTileEntryEvent.Receive(), cudaEventDisableTiming),
         "cudaEventCreateWithFlags(tile entry)");
      for (auto &event : fTileStagingDoneEvents)
         INTERNAL::CheckCudaStatus(cudaEventCreateWithFlags(event.Receive(), cudaEventDisableTiming),
                                   "cudaEventCreateWithFlags(tile staging done)");
      for (auto &event : fTileComputeDoneEvents)
         INTERNAL::CheckCudaStatus(cudaEventCreateWithFlags(event.Receive(), cudaEventDisableTiming),
                                   "cudaEventCreateWithFlags(tile compute done)");
   }

   QuantizedGemmCudaLtState() = default;
   QuantizedGemmCudaLtState(const QuantizedGemmCudaLtState &) = delete;
   QuantizedGemmCudaLtState &operator=(const QuantizedGemmCudaLtState &) = delete;
   QuantizedGemmCudaLtState(QuantizedGemmCudaLtState &&) noexcept = default;
   QuantizedGemmCudaLtState &operator=(QuantizedGemmCudaLtState &&) noexcept = default;
   ~QuantizedGemmCudaLtState() = default;

   void Reset() noexcept
   {
      fBlas.resetInt8();
      // Scratch-arena views are not owned; they are only unbound from the torn-down problem.
      fWorkspace = nullptr;
      fInputQuantized = nullptr;
      fAccumulator = nullptr;
      fOutputQuantized = nullptr;
      fBiasOutputOffset = nullptr;
      fInputQuantizedBytes = 0;
      fAccumulatorBytes = 0;
      fOutputQuantizedBytes = 0;
      fBiasOutputOffsetBytes = 0;
   }

   // narrowOutput asks the GEMM to write output codes directly; the caller reads
   // NarrowsOutput() for what the provider accepted and picks its destination from that.
   void Initialize(const QuantizedDenseLinearInvocation &params, bool narrowOutput = false)
   {
      fBlas.prepareInt8(MakeMatmulDesc(params), narrowOutput, true);
   }

   // Attempts initialization for a unit-kernel Conv's direct column-major input layout;
   // returns false, leaving the state reset, when the provider has no algorithm for it.
   bool TryInitializeDirectInput(const QuantizedDenseLinearInvocation &params)
   {
      return fBlas.tryPrepareInt8ColumnMajorA(MakeMatmulDesc(params));
   }

   bool NarrowsOutput() const { return fBlas.int8OutputNarrowed(); }

   bool AColumnMajorInput() const { return fBlas.int8ColumnMajorA(); }
   bool DirectInputLayoutUnsupported() const { return fBlas.int8ColumnMajorARejected(); }

private:
   static QuantBlasCuda::Int8MatmulDesc MakeMatmulDesc(const QuantizedDenseLinearInvocation &params)
   {
      QuantBlasCuda::Int8MatmulDesc desc;
      desc.m = params.m;
      desc.n = params.n;
      desc.k = params.k;
      desc.batchCount = params.batchCount;
      desc.strideA = params.batchStrideA;
      desc.strideB = params.batchStrideB;
      desc.strideC = params.batchStrideC;
      desc.aColumnMajor = params.aColumnMajorInput;
      desc.bias = params.hasBias;
      desc.relu = params.hasRelu;
      desc.workspaceLimitBytes = params.maxWorkspaceBytes;
      return desc;
   }

public:
   void BindScratch(QuantizedCudaScratchView scratch) { fScratch = scratch; }

   void PrepareScratch(const QuantizedDenseLinearInvocation &params)
   {
      QuantizedCudaScratchCursor cursor(fScratch);
      fWorkspace = cursor.Take(params.maxWorkspaceBytes);
      // Batched staging covers every slice, matching the extents QuantizedGemmCudaLt_Call
      // launches with; the accumulator below already did.
      const std::size_t stagedBatch = params.batchCount == 0 ? 1 : params.batchCount;
      fInputQuantizedBytes = stagedBatch * params.m * params.k * sizeof(std::int8_t);
      fInputQuantized = static_cast<std::int8_t *>(cursor.Take(fInputQuantizedBytes));
      fAccumulatorBytes = params.batchCount * params.m * params.n * sizeof(std::int32_t);
      fAccumulator = static_cast<std::int32_t *>(cursor.Take(fAccumulatorBytes));
      fOutputQuantizedBytes = 0;
      fOutputQuantized = nullptr;
      if (params.paddedExecution && params.epilogueMode == EQuantizedEpilogueMode::Quantized) {
         const std::size_t outputElementSize = params.outputCarrier == EQuantizedOutputCarrier::UInt8
                                                 ? sizeof(std::uint8_t) : sizeof(std::int8_t);
         fOutputQuantizedBytes = stagedBatch * params.m * params.n * outputElementSize;
         fOutputQuantized = cursor.Take(fOutputQuantizedBytes);
      }
      fBiasOutputOffsetBytes = params.hasBias && params.batchCount == 1
                                  ? params.n * sizeof(float) : 0;
      fBiasOutputOffset = static_cast<float *>(cursor.Take(fBiasOutputOffsetBytes));
   }

   const float *EnsureBiasOutputOffsetBuffer(const QuantizedDenseLinearInvocation &params, const float *bias,
                                             const float *weightScaleVector, QuantizedGemmCudaStream stream)
   {
      if (bias == nullptr || !params.hasBias)
         return nullptr;

      if (fBiasOutputOffset == nullptr || fBiasOutputOffsetBytes < params.n * sizeof(float))
         throw std::runtime_error("SOFIE quantized CUDA bias staging is absent from the session scratch contract");

      constexpr int threads = 256;
      const int blocks = static_cast<int>((params.n + threads - 1) / threads);
      INTERNAL::QuantizedGemmCudaBiasOutputOffsetKernel<<<blocks, threads, 0, stream>>>(
         fBiasOutputOffset, bias, weightScaleVector, params);
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaBiasOutputOffsetKernel launch");
      return fBiasOutputOffset;
   }

   // Column sums of the constant int8 weight for the asymmetric-input correction; null for
   // a symmetric input, so the epilogue's symmetric path is untouched.
   const std::int32_t *EnsureInputZeroPointColumnSums(const std::int8_t *weight,
                                                      const QuantizedDenseLinearInvocation &params,
                                                      QuantizedGemmCudaStream stream)
   {
      if (params.inputZeroPoint == 0)
         return nullptr;
      // Built once from constant weight storage, so a batch whose B slices differ would
      // read sums belonging to the first slice.
      if (params.batchCount > 1 && params.batchStrideB != 0)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM cannot correct a nonzero input zero point "
                                  "against a per-slice batched weight");
      if (!fInputZpColumnSumsBuilt) {
         INTERNAL::CheckCudaStatus(cudaMalloc(fInputZpColumnSums.Receive(), params.n * sizeof(std::int32_t)),
                                   "cudaMalloc(input zero-point column sums)");
         constexpr int threads = 256;
         const int blocks = static_cast<int>((params.n + threads - 1) / threads);
         INTERNAL::QuantizedGemmCudaWeightColumnSumKernel<<<blocks, threads, 0, stream>>>(
            fInputZpColumnSums.Get(), weight, params.n, params.k);
         INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaWeightColumnSumKernel launch");
         fInputZpColumnSumsBuilt = true;
      }
      return fInputZpColumnSums.Get();
   }

   void RecordDeferredEpilogue(const float *bias, const float *weightScaleVector,
                               const std::int32_t *inputZeroPointColumnSums,
                               const QuantizedDenseLinearInvocation &effectiveParams)
   {
      if (fAccumulator == nullptr)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM deferred its epilogue without an accumulator");
      fDeferredEpilogue.accumulator = fAccumulator;
      fDeferredEpilogue.bias = bias;
      fDeferredEpilogue.weightScaleVector = weightScaleVector;
      fDeferredEpilogue.inputZeroPointColumnSums = inputZeroPointColumnSums;
      fDeferredEpilogue.params = effectiveParams;
   }

   std::int8_t *InputQuantizedBuffer() const { return fInputQuantized; }
   std::int32_t *AccumulatorBuffer() const { return fAccumulator; }
   void *OutputQuantizedBuffer() const { return fOutputQuantized; }
   std::size_t AccumulatorBytes() const { return fAccumulatorBytes; }
   std::size_t WorkspaceSize() const { return fBlas.int8Stats().workspaceBytes; }
   int HeuristicResultCount() const { return fBlas.int8Stats().heuristicCount; }
   int SelectedHeuristicIndex() const { return fBlas.int8Stats().selectedHeuristic; }
   float AutotuneMs() const { return fBlas.int8Stats().autotuneMs; }
   int AutotunedCandidateCount() const { return fBlas.int8Stats().autotunedCandidates; }
   float SelectedCandidateMs() const { return fBlas.int8Stats().selectedCandidateMs; }

   // target is the accumulator, or the output code buffer when a narrowed store was confirmed;
   // biasOutputOffset is the per-column offset in output units a narrowed biased store needs.
   void Execute(void *target, const std::int8_t *inputQuantized, const std::int8_t *weightQuantized,
                const QuantizedDenseLinearInvocation &params, QuantizedGemmCudaStream stream,
                bool narrowOutput = false, const float *biasOutputOffset = nullptr)
   {
      Initialize(params, narrowOutput);
      fBlas.matmulInt8(stream, inputQuantized, weightQuantized, target,
                       static_cast<float>(params.accumulatorToOutputScale), biasOutputOffset,
                       fWorkspace, params.enableAutotuning, params.autotuneIterations);
   }
};

inline void QuantizedGemmCudaLt_Call(QuantizedGemmCudaLtState &state, QuantizedGemmCudaStream stream,
                                     void *output, const void *input, const void *weight, const float *bias,
                                     const float *weightScaleVector,
                                     const QuantizedDenseLinearInvocation &params)
{
   if (output == nullptr || input == nullptr || weight == nullptr) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM received a null required pointer");
   }
   if (params.m == 0 || params.n == 0 || params.k == 0) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM requires nonzero M, N, and K");
   }
   if (params.weightType != EQuantizedWeightCarrier::Int8) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM currently supports signed int8 weights only");
   }
   // A nonzero input zero point is corrected in the epilogue from the weight column sums.
   // A nonzero weight zero point would need per-element activation sums, which no pass builds.
   if (params.weightZeroPoint != 0) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM currently requires the weight zero point to be 0");
   }
   if (params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel && weightScaleVector == nullptr) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM per-channel weight scale mode requires a scale vector");
   }
   if (params.epilogueMode == EQuantizedEpilogueMode::Quantized &&
       params.outputCarrier == EQuantizedOutputCarrier::Float) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM quantized epilogue requires an integer output carrier");
   }
   if (params.epilogueMode != EQuantizedEpilogueMode::Quantized && !params.fuseOutputRequantize &&
       params.outputCarrier != EQuantizedOutputCarrier::Float) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM float epilogues require a float output carrier");
   }
   // Only a fake-quant float epilogue leaves a raw accumulator; see CanDeferOutputEpilogue.
   if (params.deferOutputEpilogue) {
      if (params.epilogueMode != EQuantizedEpilogueMode::ExactFakeQuant)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM can only defer the exact fake-quant float epilogue");
      if (params.fuseOutputRequantize)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM cannot both defer and requantize its output epilogue");
      if (params.outputCarrier != EQuantizedOutputCarrier::Float)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM deferred epilogue requires a float output carrier");
      // Padding is admissible (the consumer uses the epilogue's own logical->physical map);
      // an accumulator the GEMM never wrote is not.
      if (params.batchCount > 1 && params.batchStrideC != 0 &&
          static_cast<std::size_t>(params.batchStrideC) != params.m * params.n)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM deferred epilogue requires a compact batch stride");
   }
   if (params.fuseOutputRequantize) {
      if (params.epilogueMode == EQuantizedEpilogueMode::Quantized)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM output requantize fusion applies to the fake-quant "
                                  "float epilogue, not the quantized epilogue");
      if (params.outputCarrier != EQuantizedOutputCarrier::Int8)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM output requantize fusion requires an int8 output "
                                  "carrier");
      if (!(params.requantizeScale > 0.0))
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM output requantize fusion requires a positive scale");
   }

   QuantizedDenseLinearInvocation effectiveParams = params;
   if (effectiveParams.logicalM == 0)
      effectiveParams.logicalM = effectiveParams.m;
   if (effectiveParams.logicalN == 0)
      effectiveParams.logicalN = effectiveParams.n;
   if (effectiveParams.logicalK == 0)
      effectiveParams.logicalK = effectiveParams.k;
   if (effectiveParams.paddedExecution &&
       (effectiveParams.logicalM > effectiveParams.m || effectiveParams.logicalN > effectiveParams.n ||
        effectiveParams.logicalK > effectiveParams.k)) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM padded execution received logical dimensions larger than physical dimensions");
   }
   if (effectiveParams.accumulatorToOutputScale == 0.0)
      effectiveParams.accumulatorToOutputScale =
         (effectiveParams.alpha * effectiveParams.inputScale * effectiveParams.weightScale) / effectiveParams.outputScale;
   // Once per launch, so the epilogue's per-element divide becomes an fma.
   if (effectiveParams.outputScaleReciprocal == 0.0 && effectiveParams.outputScale > 0.0) {
      const auto reciprocal = INTERNAL::QuantizedMakeScaleReciprocal(effectiveParams.outputScale);
      effectiveParams.outputScaleReciprocal = reciprocal.value;
      effectiveParams.outputScaleReciprocalError = reciprocal.error;
   }

   // The staging and epilogue kernels are elementwise passes over the whole buffer, so
   // their extents are per-batch counts times the batch, indexed [batch][m][n].
   const std::size_t batchCount = effectiveParams.batchCount == 0 ? 1 : effectiveParams.batchCount;
   const std::size_t inputElements = batchCount * effectiveParams.m * effectiveParams.k;
   const std::size_t outputElements = batchCount * effectiveParams.m * effectiveParams.n;
   const std::size_t logicalOutputElements =
      batchCount * effectiveParams.logicalM * effectiveParams.logicalN;
   // The provider decides whether this shape can narrow, and the destination follows from the
   // answer, so both are settled before the GEMM is launched.
   const bool narrowRequested = QuantizedGemmCudaLt_NarrowsQuantizedOutput(effectiveParams);
   state.Initialize(effectiveParams, narrowRequested);
   state.PrepareScratch(effectiveParams);
   const bool narrowedOutput = narrowRequested && state.NarrowsOutput();

   constexpr int threads = 256;
   const std::int8_t *inputQuantized = nullptr;
   if (effectiveParams.inputCarrier == EQuantizedInputCarrier::Int8) {
      inputQuantized = static_cast<const std::int8_t *>(input);
      if (effectiveParams.paddedExecution) {
         QuantizedGemmCudaPadInt8Matrix(stream, inputQuantized, state.InputQuantizedBuffer(),
                                                  effectiveParams.logicalM, effectiveParams.logicalK,
                                                  effectiveParams.m, effectiveParams.k, 0);
         inputQuantized = state.InputQuantizedBuffer();
      }
   } else {
      // Float input carrier: quantize float->int8 internally. Output/N padding is fine here;
      // input-dimension padding would need a second pass over the quantized buffer.
      if (effectiveParams.paddedExecution &&
          (effectiveParams.logicalM < effectiveParams.m || effectiveParams.logicalK < effectiveParams.k)) {
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM float input carrier with padded input dimensions is not yet supported");
      }
      const int inputBlocks = static_cast<int>((inputElements + threads - 1) / threads);
      const float *inputFloat = static_cast<const float *>(input);
      INTERNAL::QuantizedGemmCudaQuantizeInputKernel<<<inputBlocks, threads, 0, stream>>>(
         inputFloat, state.InputQuantizedBuffer(), inputElements, effectiveParams.inputScale, effectiveParams.inputZeroPoint,
         effectiveParams.inputQMin, effectiveParams.inputQMax);
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaQuantizeInputKernel launch");
      inputQuantized = state.InputQuantizedBuffer();
   }

   // The quantized epilogue's destination and its bias offsets are also the narrowed store's,
   // so both are prepared ahead of the GEMM that may now write them itself.
   void *epilogueOutput = nullptr;
   const float *biasOutputOffset = nullptr;
   if (effectiveParams.epilogueMode == EQuantizedEpilogueMode::Quantized) {
      biasOutputOffset = state.EnsureBiasOutputOffsetBuffer(effectiveParams, bias, weightScaleVector, stream);
      epilogueOutput = effectiveParams.paddedExecution ? state.OutputQuantizedBuffer() : output;
      if (epilogueOutput == nullptr) {
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM padded execution did not allocate an output scratch buffer");
      }
   }

   void *gemmTarget = narrowedOutput ? epilogueOutput : static_cast<void *>(state.AccumulatorBuffer());
   state.Execute(gemmTarget, inputQuantized, static_cast<const std::int8_t *>(weight), effectiveParams,
                 stream, narrowedOutput, biasOutputOffset);

   const std::int32_t *inputZpColumnSums = state.EnsureInputZeroPointColumnSums(
      static_cast<const std::int8_t *>(weight), effectiveParams, stream);
   const int outputBlocks = static_cast<int>((outputElements + threads - 1) / threads);
   if (effectiveParams.epilogueMode == EQuantizedEpilogueMode::Quantized) {
      if (narrowedOutput) {
         // The GEMM already wrote the codes; only a padded execution still owes the slice.
      } else if (effectiveParams.outputCarrier == EQuantizedOutputCarrier::UInt8) {
         auto *quantizedOutput = static_cast<std::uint8_t *>(epilogueOutput);
         if (effectiveParams.hasBias && bias != nullptr) {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, true, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, inputZpColumnSums, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, true, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, inputZpColumnSums, effectiveParams);
            }
         } else {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, false, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, inputZpColumnSums, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, false, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, inputZpColumnSums, effectiveParams);
            }
         }
      } else {
         auto *quantizedOutput = static_cast<std::int8_t *>(epilogueOutput);
         if (effectiveParams.hasBias && bias != nullptr) {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, true, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, inputZpColumnSums, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, true, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, inputZpColumnSums, effectiveParams);
            }
         } else {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, false, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, inputZpColumnSums, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, false, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, inputZpColumnSums, effectiveParams);
            }
         }
      }
      if (!narrowedOutput)
         INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaQuantizedEpilogueKernel launch");
      if (effectiveParams.paddedExecution) {
         if (effectiveParams.outputCarrier == EQuantizedOutputCarrier::UInt8) {
            QuantizedGemmCudaUnpadUInt8Matrix(stream, static_cast<const std::uint8_t *>(state.OutputQuantizedBuffer()),
                                                        static_cast<std::uint8_t *>(output),
                                                        effectiveParams.logicalM, effectiveParams.logicalN,
                                                        effectiveParams.m, effectiveParams.n);
         } else {
            QuantizedGemmCudaUnpadInt8Matrix(stream, static_cast<const std::int8_t *>(state.OutputQuantizedBuffer()),
                                                       static_cast<std::int8_t *>(output),
                                                       effectiveParams.logicalM, effectiveParams.logicalN,
                                                       effectiveParams.m, effectiveParams.n);
         }
      }
   } else {
      // Fake-quant float output: the epilogue writes the LOGICAL output directly
      // (slicing the padded accumulator), so padded execution needs no scratch.
      const int logicalOutputBlocks = static_cast<int>((logicalOutputElements + threads - 1) / threads);
      if (effectiveParams.deferOutputEpilogue) {
         // No epilogue launch and no write to `output`: the accumulator stays where the GEMM
         // left it for the consumer.
         // effectiveParams, not params: this is the struct the epilogue kernel would receive.
         state.RecordDeferredEpilogue(bias, weightScaleVector, inputZpColumnSums, effectiveParams);
      } else if (effectiveParams.fuseOutputRequantize) {
         // Writes the logical output compactly, so a padded GEMM needs no scratch or unpad.
         INTERNAL::QuantizedGemmCudaFusedRequantizeEpilogueKernel<<<logicalOutputBlocks, threads, 0, stream>>>(
            static_cast<std::int8_t *>(output), state.AccumulatorBuffer(), bias, weightScaleVector,
            inputZpColumnSums, effectiveParams);
         INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaFusedRequantizeEpilogueKernel launch");
      } else {
         auto *floatOutput = static_cast<float *>(output);
         INTERNAL::QuantizedGemmCudaEpilogueKernel<<<logicalOutputBlocks, threads, 0, stream>>>(floatOutput, state.AccumulatorBuffer(), bias, weightScaleVector, inputZpColumnSums, effectiveParams);
         INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaEpilogueKernel launch");
      }
   }
}

#endif // SOFIE_USE_CUBLASLT

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_INT8
