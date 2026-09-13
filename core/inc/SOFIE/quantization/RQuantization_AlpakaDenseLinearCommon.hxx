#ifndef SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_COMMON
#define SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_COMMON

// CUDA-side scaffolding both dense-linear precisions sit on: owned CUDA handles, the deferred-
// epilogue holder, and the unpad. The cuBLASLt problem lives in RQuantization_BlasCudaQuant.hxx.

#include "SOFIE/quantization/RQuantization_AlpakaCommon.hxx"
#include "SOFIE/quantization/RQuantization_AlpakaPrimitives.hxx"
#include "SOFIE/quantization/RQuantization_BlasCudaQuant.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef SOFIE_USE_CUBLASLT
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace SOFIE {

#ifdef SOFIE_USE_CUBLASLT
namespace INTERNAL {

// Move-only owner of one cuBLASLt/CUDA handle: destroys in the destructor, nulls on move.
// Makes the cuBLASLt state structs default-movable.
template <typename Handle, auto DestroyFn>
struct QuantizedCudaOwnedHandle {
   Handle fValue = nullptr;

   QuantizedCudaOwnedHandle() = default;
   QuantizedCudaOwnedHandle(const QuantizedCudaOwnedHandle &) = delete;
   QuantizedCudaOwnedHandle &operator=(const QuantizedCudaOwnedHandle &) = delete;
   QuantizedCudaOwnedHandle(QuantizedCudaOwnedHandle &&other) noexcept : fValue(other.fValue)
   {
      other.fValue = nullptr;
   }
   QuantizedCudaOwnedHandle &operator=(QuantizedCudaOwnedHandle &&other) noexcept
   {
      if (this != &other) {
         Reset();
         fValue = other.fValue;
         other.fValue = nullptr;
      }
      return *this;
   }
   // Adopts a handle created elsewhere (e.g. CreateRowMajorLayout), destroying any current one.
   QuantizedCudaOwnedHandle &operator=(Handle value) noexcept
   {
      Reset();
      fValue = value;
      return *this;
   }
   ~QuantizedCudaOwnedHandle() { Reset(); }

   void Reset() noexcept
   {
      if (fValue != nullptr) {
         DestroyFn(fValue);
         fValue = nullptr;
      }
   }
   // For the create call: destroys any current handle and exposes the slot to write into.
   Handle *Receive()
   {
      Reset();
      return &fValue;
   }
   Handle Get() const { return fValue; }
   operator Handle() const { return fValue; }
};

// Owned device allocation (cudaMalloc'd), as opposed to the non-owning scratch-arena views.
template <typename T>
using QuantizedCudaDeviceBuffer = QuantizedCudaOwnedHandle<T *, cudaFree>;
using QuantizedCudaOwnedStream = QuantizedCudaOwnedHandle<cudaStream_t, cudaStreamDestroy>;
using QuantizedCudaOwnedEvent = QuantizedCudaOwnedHandle<cudaEvent_t, cudaEventDestroy>;

} // namespace INTERNAL
#endif // SOFIE_USE_CUBLASLT

// Shared by both builds of QuantizedGemmCudaLtState below, since generated code reads it
// through DeferredEpilogue() either way. Borrowed pointers only, so no teardown ordering.
struct QuantizedDeferredEpilogueHolder {
   // Valid only after a call that ran with params.deferOutputEpilogue.
   const QuantizedDeferredEpilogue &DeferredEpilogue() const { return fDeferredEpilogue; }

   QuantizedDeferredEpilogue fDeferredEpilogue{};
};

#ifdef SOFIE_USE_CUBLASLT

namespace INTERNAL {

template <typename T>
__global__ void QuantizedGemmCudaUnpadMatrixKernel(const T *__restrict__ padded,
                                                   T *__restrict__ output,
                                                   std::size_t logicalRows, std::size_t logicalCols,
                                                   std::size_t physicalRows, std::size_t physicalCols)
{
   const std::size_t elements = logicalRows * logicalCols;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;
   const std::size_t row = idx / logicalCols;
   const std::size_t col = idx % logicalCols;
   output[idx] = row < physicalRows && col < physicalCols ? padded[row * physicalCols + col] : T{};
}


} // namespace INTERNAL

template <typename T>
inline void QuantizedGemmCudaUnpadMatrix(QuantizedGemmCudaStream stream, const T *padded, T *output,
                                         std::size_t logicalRows, std::size_t logicalCols,
                                         std::size_t physicalRows, std::size_t physicalCols)
{
   const std::size_t elements = logicalRows * logicalCols;
   if (elements == 0)
      return;
   constexpr int blockSize = 256;
   const int gridSize = static_cast<int>((elements + blockSize - 1) / blockSize);
   INTERNAL::QuantizedGemmCudaUnpadMatrixKernel<T><<<gridSize, blockSize, 0, stream>>>(
      padded, output, logicalRows, logicalCols, physicalRows, physicalCols);
   INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaUnpadMatrixKernel");
}

#endif // SOFIE_USE_CUBLASLT

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_COMMON
