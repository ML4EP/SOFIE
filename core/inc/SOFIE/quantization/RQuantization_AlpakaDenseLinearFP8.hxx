#ifndef SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_FP8
#define SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_FP8

// The FP8 (E4M3) dense-linear path: carrier and capability mapping, the cuBLASLt FP8 state and
// its descriptor build, the bias epilogue and padded-output slice, and the call.

#include "SOFIE/quantization/RQuantization_AlpakaDenseLinearCommon.hxx"

namespace SOFIE {

#ifndef SOFIE_USE_CUBLASLT
// Generated code emits BindScratch and WorkspaceSize unconditionally, so the stub
// carries both as no-ops.
struct QuantizedGemmCudaLtFP8State {
   void BindScratch(QuantizedCudaScratchView) {}
   std::size_t WorkspaceSize() const { return 0; }
};
#else
struct QuantizedGemmCudaLtFP8State {
   // The cuBLASLt problem lives in the provider; this state keeps the session-side scratch
   // and the BF16 bias conversion the library epilogue needs.
   QuantBlasCuda fBlas;
   void *fWorkspace = nullptr;
   void *fOutputStaging = nullptr;
   // A bias folded into the cuBLASLt epilogue, held in BF16 (the only bias type the FP8
   // heuristic accepts); converted once, outliving the problem rebuilds Reset() performs.
   INTERNAL::QuantizedCudaDeviceBuffer<void> fFusedBias;
   const float *fFusedBiasSource = nullptr;
   bool fFuseBias = false;
   QuantizedCudaScratchView fScratch{};

   QuantizedGemmCudaLtFP8State() = default;
   QuantizedGemmCudaLtFP8State(const QuantizedGemmCudaLtFP8State &) = delete;
   QuantizedGemmCudaLtFP8State &operator=(const QuantizedGemmCudaLtFP8State &) = delete;
   QuantizedGemmCudaLtFP8State(QuantizedGemmCudaLtFP8State &&) noexcept = default;
   QuantizedGemmCudaLtFP8State &operator=(QuantizedGemmCudaLtFP8State &&) noexcept = default;
   ~QuantizedGemmCudaLtFP8State() = default;

   void Reset() noexcept;
   // Separate from Reset() on purpose: Reset() tears the problem down on every shape
   // change, and the converted bias is shape-independent and costs a device sync to rebuild.
   void ResetFusedBias() noexcept;
   bool TryFuseBias(const float *bias, const QuantizedFP8DenseLinearInvocation &params,
                    QuantizedGemmCudaStream stream);
   void BindScratch(QuantizedCudaScratchView scratch) { fScratch = scratch; }
   void PrepareScratch(const QuantizedFP8DenseLinearInvocation &params);
   void *OutputStagingBuffer() const { return fOutputStaging; }
   void Initialize(const QuantizedFP8DenseLinearInvocation &params);
   void Execute(void *output, const void *input, const void *weight, const QuantizedFP8DenseLinearInvocation &params,
                QuantizedGemmCudaStream stream);
   std::size_t WorkspaceSize() const { return fBlas.fp8Stats().workspaceBytes; }
};
#endif // SOFIE_USE_CUBLASLT

inline const char *QuantizedGemmCudaLtFP8_OutputProfileName(ELowPrecisionFormat carrier)
{
   switch (carrier) {
   case ELowPrecisionFormat::Float16:
      return "f16";
   case ELowPrecisionFormat::BFloat16:
      return "bf16";
   case ELowPrecisionFormat::Float32:
      return "f32";
   case ELowPrecisionFormat::FP8E4M3:
      return "fp8e4m3";
   case ELowPrecisionFormat::FP8E5M2:
      return "fp8e5m2";
   }
   return "unknown";
}

inline bool QuantizedGemmCudaLtFP8_IsExecutableE4M3TN(const QuantizedFP8DenseLinearInvocation &params)
{
   // E4M3 is executable as an output carrier, which is what keeps an FP8 layer chain in
   // FP8. It requires the BF16 C matrix set up below.
   const bool supportedOutput = params.outputCarrier == ELowPrecisionFormat::Float16 ||
                                params.outputCarrier == ELowPrecisionFormat::BFloat16 ||
                                params.outputCarrier == ELowPrecisionFormat::Float32 ||
                                params.outputCarrier == ELowPrecisionFormat::FP8E4M3;
   return params.m != 0 && params.n != 0 && params.k != 0 && params.batchCount != 0 &&
          params.inputFormat == ELowPrecisionFormat::FP8E4M3 &&
          params.weightFormat == ELowPrecisionFormat::FP8E4M3 &&
          supportedOutput &&
          params.accumulation == ELowPrecisionFormat::Float32;
}

inline std::string QuantizedGemmCudaLtFP8_CapabilityTag(const QuantizedFP8DenseLinearInvocation &params)
{
   return std::string("fp8_dense_linear_cublaslt_e4m3_tn_") +
          QuantizedGemmCudaLtFP8_OutputProfileName(params.outputCarrier);
}

inline QuantizedDenseLinearBackendCapability QuantizedGemmCudaLtFP8_QueryCapability(
   const QuantizedFP8DenseLinearInvocation &params)
{
#ifdef SOFIE_USE_CUBLASLT
   if (QuantizedGemmCudaLtFP8_IsExecutableE4M3TN(params)) {
      QuantizedDenseLinearBackendCapability capability;
      capability.backend = EQuantizedBackend::ALPAKA;
      capability.executable = true;
      capability.profile = EQuantizedComputeProfile::FP8E4M3DenseLinearRank2;
      capability.inputCarrier = ELowPrecisionCarrier::FP8E4M3;
      capability.weightCarrier = ELowPrecisionCarrier::FP8E4M3;
      capability.outputCarrier = LowPrecisionCarrierForFormat(params.outputCarrier);
      capability.tag = QuantizedGemmCudaLtFP8_CapabilityTag(params);
      capability.reason = "SOFIE cuBLASLt FP8 E4M3 TN " + std::string(QuantizedGemmCudaLtFP8_OutputProfileName(params.outputCarrier)) +
                          " path is executable for this backend";
      return capability;
   }
#endif
   return MakeFP8DenseLinearBackendUnsupportedCapability(
      EQuantizedBackend::ALPAKA,
      LowPrecisionCarrierForFormat(params.inputFormat),
      LowPrecisionCarrierForFormat(params.weightFormat),
      LowPrecisionCarrierForFormat(params.outputCarrier),
      "SOFIE FP8 cuBLASLt dense-linear boundary supports executable E4M3 x E4M3 TN E4M3/Float16/BFloat16/Float32 output only in this build/backend");
}

#ifdef SOFIE_USE_CUBLASLT
inline cudaDataType_t QuantizedGemmCudaLtFP8_OutputDataType(ELowPrecisionFormat carrier)
{
   switch (carrier) {
   case ELowPrecisionFormat::Float16:
      return CUDA_R_16F;
   case ELowPrecisionFormat::BFloat16:
      return CUDA_R_16BF;
   case ELowPrecisionFormat::Float32:
      return CUDA_R_32F;
   case ELowPrecisionFormat::FP8E4M3:
      return CUDA_R_8F_E4M3;
   case ELowPrecisionFormat::FP8E5M2:
      return CUDA_R_8F_E5M2;
   }
   return CUDA_R_32F;
}

// Whether cuBLASLt is given a D scale: exactly when D narrows to FP8 on a non-unit grid.
// The output then holds CODES, so a value-unit bias divides by the output scale first.
inline bool QuantizedGemmCudaLtFP8_ProgramsOutputScale(const QuantizedFP8DenseLinearInvocation &params)
{
   const auto type = QuantizedGemmCudaLtFP8_OutputDataType(params.outputCarrier);
   return params.outputScale != 1.0f && (type == CUDA_R_8F_E4M3 || type == CUDA_R_8F_E5M2);
}
#endif

#ifdef SOFIE_USE_CUBLASLT

namespace INTERNAL {

template <typename OutputT>
__device__ inline float QuantizedGemmCudaFP8OutputToFloat(OutputT value)
{
   return static_cast<float>(value);
}

template <typename OutputT>
__device__ inline OutputT QuantizedGemmCudaFP8OutputFromFloat(float value)
{
   return static_cast<OutputT>(value);
}

// biasToOutputUnits converts the bias from value units to the units `output` holds: 1 when
// the GEMM wrote values, 1/outputScale when it narrowed to FP8 and the buffer holds codes.
template <typename OutputT>
__global__ void QuantizedGemmCudaLtFP8BiasEpilogueKernel(OutputT *__restrict__ output,
                                                         const float *__restrict__ bias,
                                                         float biasToOutputUnits,
                                                         QuantizedFP8DenseLinearInvocation params)
{
   // Batched slices are contiguous, so one flat index covers them; only the clamp reaches
   // here batched, since no bias-bearing batched FP8 call exists.
   const std::size_t elements = params.m * params.n * (params.batchCount > 1 ? params.batchCount : 1);
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;

   // Output features run along m under NT, along n otherwise.
   const std::size_t col = params.weightIsMatrixA ? (idx % params.m) : (idx % params.n);
   // A padded column is dropped by the slice that follows, and the bias holds only the
   // logical extent, so reading it here would run past the end.
   if (params.paddedExecution && col >= params.logicalM)
      return;
   float value = QuantizedGemmCudaFP8OutputToFloat(output[idx]);
   if (params.hasBias && bias != nullptr)
      value += params.beta * bias[col] * biasToOutputUnits;
   // Relu commutes with the positive scale above, so testing the code is testing the value.
   if (params.hasRelu && !(value > 0.0f))
      value = 0.0f;
   if (params.hasOutputClamp)
      value = fminf(fmaxf(value, params.outputClampLow), params.outputClampHigh);
   output[idx] = QuantizedGemmCudaFP8OutputFromFloat<OutputT>(value);
}

// The padded call writes [rows, physicalCols] row-major; the graph value is the leading
// [rows, logicalCols] of each row, copied out by the shared unpad kernel.
inline void QuantizedGemmCudaLtFP8SlicePaddedOutput(QuantizedGemmCudaStream stream, void *output,
                                                    const void *staging,
                                                    const QuantizedFP8DenseLinearInvocation &params)
{
   const std::size_t elements = params.n * params.logicalM;
   if (elements == 0)
      return;
   if (params.outputCarrier != ELowPrecisionFormat::Float32)
      throw std::runtime_error("SOFIE FP8 padded output staging supports only the Float32 output carrier");
   constexpr int threads = 256;
   const int blocks = static_cast<int>((elements + threads - 1) / threads);
   QuantizedGemmCudaUnpadMatrixKernel<float><<<blocks, threads, 0, stream>>>(
      static_cast<const float *>(staging), static_cast<float *>(output),
      params.n, params.logicalM, params.n, params.m);
   CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaUnpadMatrixKernel(FP8 slice)");
}

inline void QuantizedGemmCudaLtFP8ApplyBiasEpilogue(QuantizedGemmCudaStream stream, void *output,
                                                     const float *bias,
                                                     const QuantizedFP8DenseLinearInvocation &params)
{
   const bool appliesBias = params.hasBias && bias != nullptr && params.beta != 0.0f;
   if (!appliesBias && !params.hasRelu && !params.hasOutputClamp)
      return;
   const std::size_t elements = params.m * params.n * (params.batchCount > 1 ? params.batchCount : 1);
   if (elements == 0)
      return;
   // With a programmed D scale `output` holds codes, so the value-unit bias converts first;
   // cuBLASLt's fused epilogue adds bias to the accumulator then scales, the same arithmetic.
   const float biasToOutputUnits =
      QuantizedGemmCudaLtFP8_ProgramsOutputScale(params) ? 1.0f / params.outputScale : 1.0f;
   constexpr int threads = 256;
   const int blocks = static_cast<int>((elements + threads - 1) / threads);
   switch (params.outputCarrier) {
   case ELowPrecisionFormat::Float32:
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<float><<<blocks, threads, 0, stream>>>(static_cast<float *>(output), bias, biasToOutputUnits, params);
      break;
   case ELowPrecisionFormat::Float16:
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<__half><<<blocks, threads, 0, stream>>>(static_cast<__half *>(output), bias, biasToOutputUnits, params);
      break;
   case ELowPrecisionFormat::BFloat16:
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(static_cast<__nv_bfloat16 *>(output), bias, biasToOutputUnits, params);
      break;
   case ELowPrecisionFormat::FP8E4M3:
      // E4M3 activation carrier: the GEMM already stored E4M3 codes, so bias and Relu are
      // applied in place through float, on the code grid.
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<__nv_fp8_e4m3><<<blocks, threads, 0, stream>>>(static_cast<__nv_fp8_e4m3 *>(output), bias, biasToOutputUnits, params);
      break;
   default:
      throw std::runtime_error("SOFIE FP8 bias epilogue supports E4M3, Float32, Float16, and BFloat16 output carriers");
   }
   CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaLtFP8BiasEpilogueKernel");
}
} // namespace INTERNAL


inline void QuantizedGemmCudaLtFP8State::Reset() noexcept
{
   fBlas.resetFp8();
   // Scratch-arena views are not owned; they are only unbound from the torn-down problem.
   // fFusedBias itself survives: it is keyed to the bias values, not to this shape.
   fWorkspace = nullptr;
   fOutputStaging = nullptr;
}

inline void QuantizedGemmCudaLtFP8State::ResetFusedBias() noexcept
{
   fFusedBias.Reset();
   fFusedBiasSource = nullptr;
   fFuseBias = false;
}

// Decides whether this call's bias can ride in the cuBLASLt epilogue, materialising it in
// BF16. Must run before Initialize(): the epilogue is an input to the heuristic query.
inline bool QuantizedGemmCudaLtFP8State::TryFuseBias(const float *bias,
                                                      const QuantizedFP8DenseLinearInvocation &params,
                                                      QuantizedGemmCudaStream stream)
{
   // Each guard marks a case where the fused epilogue is not known to match the standalone
   // kernel; an output clamp always needs the kernel, which keeps bias-then-clamp order.
   const bool eligible = params.hasBias && bias != nullptr && params.beta != 0.0f &&
                         params.weightIsMatrixA && !params.paddedExecution &&
                         params.batchCount <= 1 && !params.hasOutputClamp;
   if (!eligible) {
      if (fFuseBias)
         ResetFusedBias();
      return false;
   }
   if (fFuseBias && fFusedBiasSource == bias)
      return true;

   ResetFusedBias();

   // The bias is produced on the alpaka queue; a cudaMemcpy on another stream is not ordered
   // against that work, and zeros would pass the exactness check below on unwritten memory.
   INTERNAL::CheckCudaStatus(cudaStreamSynchronize(stream), "cudaStreamSynchronize(FP8 fused bias)");

   const std::size_t features = params.m;
   std::vector<float> host(features);
   INTERNAL::CheckCudaStatus(
      cudaMemcpy(host.data(), bias, features * sizeof(float), cudaMemcpyDeviceToHost),
      "cudaMemcpy(FP8 fused bias readback)");

   // BF16 is the only bias type the FP8 heuristic accepts, so a bias that does not survive
   // the narrowing exactly cannot be fused; the standalone epilogue kernel still runs.
   std::vector<__nv_bfloat16> narrowed(features);
   for (std::size_t i = 0; i < features; ++i) {
      const float wanted = params.beta * host[i];
      const auto candidate = __float2bfloat16(wanted);
      if (static_cast<float>(candidate) != wanted)
         return false;
      narrowed[i] = candidate;
   }

   void *device = nullptr;
   INTERNAL::CheckCudaStatus(cudaMalloc(&device, features * sizeof(__nv_bfloat16)),
                             "cudaMalloc(FP8 fused bias)");
   const auto upload = cudaMemcpy(device, narrowed.data(), features * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice);
   if (upload != cudaSuccess) {
      cudaFree(device);
      INTERNAL::CheckCudaStatus(upload, "cudaMemcpy(FP8 fused bias upload)");
   }
   fFusedBias = device;
   fFusedBiasSource = bias;
   fFuseBias = true;
   return true;
}

inline void QuantizedGemmCudaLtFP8State::Initialize(const QuantizedFP8DenseLinearInvocation &params)
{
   QuantBlasCuda::Fp8MatmulDesc desc;
   desc.m = params.m;
   desc.n = params.n;
   desc.k = params.k;
   desc.batchCount = params.batchCount;
   desc.strideA = params.batchStrideA;
   desc.strideB = params.batchStrideB;
   desc.strideC = params.batchStrideC;
   desc.aScale = params.weightIsMatrixA ? params.weightScale : params.inputScale;
   desc.bScale = params.weightIsMatrixA ? params.inputScale : params.weightScale;
   // cuBLASLt multiplies D by this before narrowing, so encoding onto a grid of step
   // `outputScale` means handing it the reciprocal.
   desc.dScale = QuantizedGemmCudaLtFP8_ProgramsOutputScale(params) ? 1.0f / params.outputScale : 1.0f;
   desc.dType = QuantizedGemmCudaLtFP8_OutputDataType(params.outputCarrier);
   desc.bias = fFuseBias ? fFusedBias.Get() : nullptr;
   desc.relu = params.hasRelu;
   desc.workspaceLimitBytes = params.maxWorkspaceBytes;
   fBlas.prepareFp8(desc);
}

inline void QuantizedGemmCudaLtFP8State::PrepareScratch(const QuantizedFP8DenseLinearInvocation &params)
{
   QuantizedCudaScratchCursor cursor(fScratch);
   fWorkspace = cursor.Take(params.maxWorkspaceBytes);
   fOutputStaging = params.paddedExecution
                       ? cursor.Take(params.batchCount * params.m * params.n * sizeof(float))
                       : nullptr;
}

inline void QuantizedGemmCudaLtFP8State::Execute(void *output, const void *input, const void *weight,
                                                  const QuantizedFP8DenseLinearInvocation &params,
                                                  QuantizedGemmCudaStream stream)
{
   Initialize(params);
   PrepareScratch(params);
   // A padded call runs at the physical width, so it writes staging and the caller slices
   // the graph value out of it.
   void *target = params.paddedExecution ? fOutputStaging : output;
   if (target == nullptr)
      throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear padded call has no output staging buffer");
   fBlas.matmulFp8(stream, input, weight, target, params.alpha, fWorkspace,
                   params.enableAutotuning, params.autotuneIterations);
}

#endif // SOFIE_USE_CUBLASLT

inline void QuantizedGemmCudaLtFP8_Call(QuantizedGemmCudaLtFP8State &state, QuantizedGemmCudaStream stream,
                                        void *output, const void *input, const void *weight, const float *bias,
                                        const QuantizedFP8DenseLinearInvocation &params)
{
   // Capability text is built only on failure: this guard sits on the per-call hot path.
   if (!QuantizedGemmCudaLtFP8_IsExecutableE4M3TN(params)) {
      throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear lowering is not executable: " +
                               QuantizedGemmCudaLtFP8_QueryCapability(params).reason);
   }
   if (output == nullptr || input == nullptr || weight == nullptr) {
      throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear path received a null required pointer");
   }
   if (params.hasBias && bias == nullptr) {
      throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear path expected a bias pointer");
   }

#ifndef SOFIE_USE_CUBLASLT
   throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear path was selected, but SOFIE_USE_CUBLASLT is not enabled");
#else
   // Folding the bias into the cuBLASLt epilogue removes one m*n launch per dense layer; it
   // must be decided before Execute, where the descriptor is built and the algorithm chosen.
   const bool biasIsFused = state.TryFuseBias(bias, params, stream);
   state.Execute(output, input, weight, params, stream);
   // The epilogue applies where the GEMM wrote, which is staging for a padded call.
   void *target = params.paddedExecution ? state.OutputStagingBuffer() : output;
   // A fused epilogue carries the Relu with the bias (RELU_BIAS), so this covers both.
   if (!biasIsFused)
      INTERNAL::QuantizedGemmCudaLtFP8ApplyBiasEpilogue(stream, target, bias, params);
   if (params.paddedExecution)
      INTERNAL::QuantizedGemmCudaLtFP8SlicePaddedOutput(stream, output, target, params);
#endif
}

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_FP8
