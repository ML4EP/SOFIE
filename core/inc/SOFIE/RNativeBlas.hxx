#ifndef SOFIE_RNATIVEBLAS_HXX
#define SOFIE_RNATIVEBLAS_HXX
#include "SOFIE/RFloat16.hxx"
#include <alpaka/alpaka.hpp>
#include <limits>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace SOFIE {
template<class Queue> class NativeBlas {
public:
   explicit NativeBlas(Queue &) {}
   template<class Value>
   void Gemm(bool transposeLeft, bool transposeRight, std::size_t rows, std::size_t columns,
             std::size_t inner, float alpha, const Value *left, const Value *right, Value *output)
   {
      for (std::size_t row = 0; row < rows; ++row)
         for (std::size_t column = 0; column < columns; ++column) {
            float sum = 0;
            for (std::size_t index = 0; index < inner; ++index)
               sum += float(left[transposeLeft ? index * rows + row : row * inner + index]) *
                      float(right[transposeRight ? column * inner + index : index * columns + column]);
            output[row * columns + column] = Value(alpha * sum);
         }
   }
   template<class Value>
   void GemmBatched(bool transposeLeft, bool transposeRight, std::size_t rows, std::size_t columns,
                    std::size_t inner, float alpha, const Value *left, const Value *right, Value *output,
                    std::size_t strideLeft, std::size_t strideRight, std::size_t batches)
   {
      for (std::size_t batch = 0; batch < batches; ++batch)
         Gemm(transposeLeft, transposeRight, rows, columns, inner, alpha,
              left + batch * strideLeft, right + batch * strideRight, output + batch * rows * columns);
   }
};
struct NativeGemmEpilogue {
   template<class Acc, class Value, std::size_t Rank>
   ALPAKA_FN_ACC void operator()(Acc const &acc, Value *output, const Value *bias,
      std::size_t length, std::array<std::size_t, Rank> shape,
      std::array<std::size_t, Rank> biasShape, float beta, int activation, float slope) const
   {
      const auto index = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (index >= length) return;
      float value = float(output[index]);
      if (bias && beta != 0) {
         std::size_t remaining = index, biasIndex = 0, stride = 1;
         for (std::size_t dimension = Rank; dimension-- > 0;) {
            const auto coordinate = remaining % shape[dimension];
            remaining /= shape[dimension];
            biasIndex += (biasShape[dimension] == 1 ? 0 : coordinate) * stride;
            stride *= biasShape[dimension];
         }
         value += beta * float(bias[biasIndex]);
      }
      if (activation == 1 && value < 0) value = 0;
      if (activation == 4 && value < 0) value *= slope;
      output[index] = Value(value);
   }
};
inline long long BlasStride(std::size_t value)
{
   if (value > static_cast<std::size_t>(std::numeric_limits<long long>::max()))
      throw std::overflow_error("SOFIE BLAS stride exceeds int64");
   return static_cast<long long>(value);
}
inline int BlasDimension(std::size_t value)
{
   if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("SOFIE BLAS dimension exceeds int32");
   return static_cast<int>(value);
}
}

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <hipblas/hipblas.h>
namespace SOFIE {
template<> class NativeBlas<alpaka::QueueHipRtNonBlocking> {
   hipblasHandle_t fHandle{};
   static void Check(hipblasStatus_t status)
   {
      if (status != HIPBLAS_STATUS_SUCCESS)
         throw std::runtime_error("SOFIE hipblas error " + std::to_string(static_cast<int>(status)));
   }
public:
   explicit NativeBlas(alpaka::QueueHipRtNonBlocking &queue)
   {
      Check(hipblasCreate(&fHandle));
      try {
         Check(hipblasSetStream(fHandle, alpaka::getNativeHandle(queue)));
         Check(hipblasSetPointerMode(fHandle, HIPBLAS_POINTER_MODE_HOST));
      } catch (...) {
         hipblasDestroy(fHandle);
         throw;
      }
   }
   NativeBlas(const NativeBlas &) = delete;
   NativeBlas &operator=(const NativeBlas &) = delete;
   ~NativeBlas() { hipblasDestroy(fHandle); }
   template<class Value>
   void Gemm(bool transposeLeft, bool transposeRight, std::size_t rows, std::size_t columns,
             std::size_t inner, float alpha, const Value *left, const Value *right, Value *output)
   {
      static_assert(std::is_same_v<Value, Float16> || std::is_same_v<Value, float>);
      const int rowCount = BlasDimension(rows);
      const int columnCount = BlasDimension(columns);
      const int innerCount = BlasDimension(inner);
      if (!rowCount || !columnCount) return;
      const auto leftOp = transposeLeft ? HIPBLAS_OP_T : HIPBLAS_OP_N;
      const auto rightOp = transposeRight ? HIPBLAS_OP_T : HIPBLAS_OP_N;
      const auto type = std::is_same_v<Value, Float16> ? HIPBLAS_R_16F : HIPBLAS_R_32F;
      const float beta = 0;
#ifdef HIPBLAS_V2
      const auto compute = HIPBLAS_COMPUTE_32F;
#else
      const auto compute = HIPBLAS_R_32F;
#endif
      Check(hipblasGemmEx(fHandle, rightOp, leftOp, columnCount, rowCount, innerCount,
         &alpha, right, type, std::max(1, transposeRight ? innerCount : columnCount),
         left, type, std::max(1, transposeLeft ? rowCount : innerCount),
         &beta, output, type, std::max(1, columnCount), compute, HIPBLAS_GEMM_DEFAULT));
   }
   template<class Value>
   void GemmBatched(bool transposeLeft, bool transposeRight, std::size_t rows, std::size_t columns,
                    std::size_t inner, float alpha, const Value *left, const Value *right, Value *output,
                    std::size_t strideLeft, std::size_t strideRight, std::size_t batches)
   {
      static_assert(std::is_same_v<Value, Float16> || std::is_same_v<Value, float>);
      const int rowCount = BlasDimension(rows), columnCount = BlasDimension(columns);
      const int innerCount = BlasDimension(inner), batchCount = BlasDimension(batches);
      const auto leftStride = BlasStride(strideLeft), rightStride = BlasStride(strideRight);
      const auto outputStride = static_cast<long long>(rowCount) * columnCount;
      if (!rowCount || !columnCount || !batchCount) return;
      const auto leftOp = transposeLeft ? HIPBLAS_OP_T : HIPBLAS_OP_N;
      const auto rightOp = transposeRight ? HIPBLAS_OP_T : HIPBLAS_OP_N;
      const auto type = std::is_same_v<Value, Float16> ? HIPBLAS_R_16F : HIPBLAS_R_32F;
      const float beta = 0;
#ifdef HIPBLAS_V2
      const auto compute = HIPBLAS_COMPUTE_32F;
#else
      const auto compute = HIPBLAS_R_32F;
#endif
      Check(hipblasGemmStridedBatchedEx(fHandle, rightOp, leftOp, columnCount, rowCount, innerCount,
         &alpha, right, type, std::max(1, transposeRight ? innerCount : columnCount), rightStride,
         left, type, std::max(1, transposeLeft ? rowCount : innerCount), leftStride,
         &beta, output, type, std::max(1, columnCount), outputStride, batchCount, compute,
         HIPBLAS_GEMM_DEFAULT));
   }
};
}
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <cublas_v2.h>
namespace SOFIE {
template<> class NativeBlas<alpaka::QueueCudaRtNonBlocking> {
   cublasHandle_t fHandle{};
   static void Check(cublasStatus_t status)
   {
      if (status != CUBLAS_STATUS_SUCCESS)
         throw std::runtime_error("SOFIE cublas error " + std::to_string(static_cast<int>(status)));
   }
public:
   explicit NativeBlas(alpaka::QueueCudaRtNonBlocking &queue)
   {
      Check(cublasCreate(&fHandle));
      try {
         Check(cublasSetStream(fHandle, alpaka::getNativeHandle(queue)));
         Check(cublasSetPointerMode(fHandle, CUBLAS_POINTER_MODE_HOST));
      } catch (...) {
         cublasDestroy(fHandle);
         throw;
      }
   }
   NativeBlas(const NativeBlas &) = delete;
   NativeBlas &operator=(const NativeBlas &) = delete;
   ~NativeBlas() { cublasDestroy(fHandle); }
   template<class Value>
   void Gemm(bool transposeLeft, bool transposeRight, std::size_t rows, std::size_t columns,
             std::size_t inner, float alpha, const Value *left, const Value *right, Value *output)
   {
      static_assert(std::is_same_v<Value, Float16> || std::is_same_v<Value, float>);
      const int rowCount = BlasDimension(rows);
      const int columnCount = BlasDimension(columns);
      const int innerCount = BlasDimension(inner);
      if (!rowCount || !columnCount) return;
      const auto leftOp = transposeLeft ? CUBLAS_OP_T : CUBLAS_OP_N;
      const auto rightOp = transposeRight ? CUBLAS_OP_T : CUBLAS_OP_N;
      const auto type = std::is_same_v<Value, Float16> ? CUDA_R_16F : CUDA_R_32F;
      const float beta = 0;
      const auto compute = CUBLAS_COMPUTE_32F;
      Check(cublasGemmEx(fHandle, rightOp, leftOp, columnCount, rowCount, innerCount,
         &alpha, right, type, std::max(1, transposeRight ? innerCount : columnCount),
         left, type, std::max(1, transposeLeft ? rowCount : innerCount),
         &beta, output, type, std::max(1, columnCount), compute, CUBLAS_GEMM_DEFAULT));
   }
   template<class Value>
   void GemmBatched(bool transposeLeft, bool transposeRight, std::size_t rows, std::size_t columns,
                    std::size_t inner, float alpha, const Value *left, const Value *right, Value *output,
                    std::size_t strideLeft, std::size_t strideRight, std::size_t batches)
   {
      static_assert(std::is_same_v<Value, Float16> || std::is_same_v<Value, float>);
      const int rowCount = BlasDimension(rows), columnCount = BlasDimension(columns);
      const int innerCount = BlasDimension(inner), batchCount = BlasDimension(batches);
      const auto leftStride = BlasStride(strideLeft), rightStride = BlasStride(strideRight);
      const auto outputStride = static_cast<long long>(rowCount) * columnCount;
      if (!rowCount || !columnCount || !batchCount) return;
      const auto leftOp = transposeLeft ? CUBLAS_OP_T : CUBLAS_OP_N;
      const auto rightOp = transposeRight ? CUBLAS_OP_T : CUBLAS_OP_N;
      const auto type = std::is_same_v<Value, Float16> ? CUDA_R_16F : CUDA_R_32F;
      const float beta = 0;
      const auto compute = CUBLAS_COMPUTE_32F;
      Check(cublasGemmStridedBatchedEx(fHandle, rightOp, leftOp, columnCount, rowCount, innerCount,
         &alpha, right, type, std::max(1, transposeRight ? innerCount : columnCount), rightStride,
         left, type, std::max(1, transposeLeft ? rowCount : innerCount), leftStride,
         &beta, output, type, std::max(1, columnCount), outputStride, batchCount, compute,
         CUBLAS_GEMM_DEFAULT));
   }
};
}
#endif
#endif
