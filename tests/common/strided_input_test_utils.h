#pragma once

// Helpers of the tests of the stride aware inference path, shared by
// cpu/TestStridedInput.cxx, alpaka/TestAlpakaStridedInput.cxx and models/StridedInputModelGenerator.cxx:
// the backing storage of a strided view and the reference results the tests compare with.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

// Backing storage of a strided view: element (i0, i1, ...) of the logical tensor `shape` lives at
// sum_d(i_d * strides[d]). Elements not belonging to the view are set to a poison value.
template <typename T = float>
struct StridedBuffer {
   std::vector<T> storage;
   std::vector<T> logical; // the values of the logical tensor, contiguous row-major

   StridedBuffer(const std::vector<size_t> &shape, const std::vector<size_t> &strides,
                 std::function<T(size_t)> gen = [](size_t k) { return T(0.37f * float(k) - 2.f); })
   {
      size_t length = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
      size_t maxOffset = 0;
      for (size_t d = 0; d < shape.size(); d++)
         maxOffset += (shape[d] - 1) * strides[d];
      storage.assign(maxOffset + 1, T(99));
      logical.resize(length);
      for (size_t k = 0; k < length; k++) {
         size_t rem = k, offset = 0;
         for (size_t d = shape.size(); d-- > 0;) {
            offset += (rem % shape[d]) * strides[d];
            rem /= shape[d];
         }
         logical[k] = gen ? gen(k) : T(0.37f * float(k) - 2.f);
         storage[offset] = logical[k];
      }
   }
};

// deterministic weights of the Gemm models of StridedInputModelGenerator.cxx
inline std::vector<float> GemmValues(size_t n, float scale)
{
   std::vector<float> v(n);
   for (size_t i = 0; i < n; i++)
      v[i] = scale * float((i * 7) % 11) - 0.5f;
   return v;
}

// Y = Relu(X * W + b), X 3x5, W 5x4
inline std::vector<float> RefGemmBiasRelu(const std::vector<float> &x)
{
   auto W = GemmValues(20, 0.1f), B = GemmValues(4, 0.3f);
   std::vector<float> y(12);
   for (size_t i = 0; i < 3; i++)
      for (size_t j = 0; j < 4; j++) {
         float acc = B[j];
         for (size_t k = 0; k < 5; k++)
            acc += x[i * 5 + k] * W[k * 4 + j];
         y[i * 4 + j] = std::max(acc, 0.f);
      }
   return y;
}

// Y = X^T * W^T, X stored 5x3, W stored 4x5
inline std::vector<float> RefGemmTransAB(const std::vector<float> &x)
{
   auto W = GemmValues(20, 0.1f);
   std::vector<float> y(12);
   for (size_t i = 0; i < 3; i++)
      for (size_t j = 0; j < 4; j++) {
         float acc = 0;
         for (size_t k = 0; k < 5; k++)
            acc += x[k * 3 + i] * W[j * 5 + k];
         y[i * 4 + j] = acc;
      }
   return y;
}

inline std::vector<float> AffineValues(size_t n, float a, float b)
{
   std::vector<float> v(n);
   for (size_t i = 0; i < n; i++)
      v[i] = a + b * float(i);
   return v;
}

// softmax (or log softmax) over `axis` of a 2x3x4 tensor
inline std::vector<float> RefSoftmax(const std::vector<float> &x, size_t axis, bool logSoftmax)
{
   const size_t shape[3] = {2, 3, 4};
   const size_t stride[3] = {12, 4, 1};
   std::vector<float> y(x.size());
   for (size_t i = 0; i < 24; i++) {
      // position along the axis and offset of the first element of the slice containing i
      size_t pos = (i / stride[axis]) % shape[axis];
      size_t base = i - pos * stride[axis];
      float vmax = x[base], sum = 0;
      for (size_t k = 0; k < shape[axis]; k++)
         vmax = std::max(vmax, x[base + k * stride[axis]]);
      for (size_t k = 0; k < shape[axis]; k++)
         sum += std::exp(x[base + k * stride[axis]] - vmax);
      y[i] = logSoftmax ? x[i] - vmax - std::log(sum) : std::exp(x[i] - vmax) / sum;
   }
   return y;
}

inline std::vector<float> RefReduceSumMid(const std::vector<float> &x)
{
   std::vector<float> y(8, 0.f);
   for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 3; j++)
         for (size_t k = 0; k < 4; k++)
            y[i * 4 + k] += x[(i * 3 + j) * 4 + k];
   return y;
}

inline std::vector<float> RefReduceMeanLast(const std::vector<float> &x)
{
   std::vector<float> y(6, 0.f);
   for (size_t r = 0; r < 6; r++)
      for (size_t k = 0; k < 4; k++)
         y[r] += x[r * 4 + k] / 4.f;
   return y;
}

inline std::vector<float> RefReduceMaxFirst(const std::vector<float> &x)
{
   std::vector<float> y(12);
   for (size_t k = 0; k < 12; k++)
      y[k] = std::max(x[k], x[12 + k]);
   return y;
}

// layer normalization over the dimensions {3, 4} of each of the 2 batches (scale and bias of shape {3, 4})
inline std::vector<float> RefLayerNorm(const std::vector<float> &x)
{
   auto scale = AffineValues(12, 0.5f, 0.1f), bias = AffineValues(12, -1.f, 0.05f);
   std::vector<float> y(24);
   for (size_t b = 0; b < 2; b++) {
      float mean = 0, var = 0;
      for (size_t k = 0; k < 12; k++)
         mean += x[b * 12 + k] / 12.f;
      for (size_t k = 0; k < 12; k++)
         var += (x[b * 12 + k] - mean) * (x[b * 12 + k] - mean) / 12.f;
      for (size_t k = 0; k < 12; k++)
         y[b * 12 + k] = scale[k] * (x[b * 12 + k] - mean) / std::sqrt(var + 1.e-5f) + bias[k];
   }
   return y;
}

inline std::vector<float> RefRMSNorm(const std::vector<float> &x)
{
   auto scale = AffineValues(12, 0.5f, 0.1f);
   std::vector<float> y(24);
   for (size_t b = 0; b < 2; b++) {
      float ms = 0;
      for (size_t k = 0; k < 12; k++)
         ms += x[b * 12 + k] * x[b * 12 + k] / 12.f;
      for (size_t k = 0; k < 12; k++)
         y[b * 12 + k] = scale[k] * x[b * 12 + k] / std::sqrt(ms + 1.e-5f);
   }
   return y;
}

// batch normalization of the channel dimension (size 3) of a 2x3x4 tensor
inline std::vector<float> RefBatchNorm(const std::vector<float> &x)
{
   auto scale = AffineValues(3, 0.5f, 0.25f), bias = AffineValues(3, -0.5f, 0.1f), mean = AffineValues(3, 0.1f, 0.2f),
        var = AffineValues(3, 0.5f, 0.3f);
   std::vector<float> y(24);
   for (size_t i = 0; i < 24; i++) {
      size_t c = (i / 4) % 3;
      y[i] = (x[i] - mean[c]) * scale[c] / std::sqrt(var[c] + 1.e-5f) + bias[c];
   }
   return y;
}

// group normalization of a 2x4x3 tensor in 2 groups of 2 channels
inline std::vector<float> RefGroupNorm(const std::vector<float> &x)
{
   auto scale = AffineValues(4, 0.5f, 0.25f), bias = AffineValues(4, -0.5f, 0.1f);
   std::vector<float> y(24);
   for (size_t n = 0; n < 2; n++)
      for (size_t g = 0; g < 2; g++) {
         const size_t base = n * 12 + g * 6;
         float mean = 0, var = 0;
         for (size_t k = 0; k < 6; k++)
            mean += x[base + k] / 6.f;
         for (size_t k = 0; k < 6; k++)
            var += (x[base + k] - mean) * (x[base + k] - mean) / 6.f;
         for (size_t k = 0; k < 6; k++) {
            size_t c = g * 2 + k / 3;
            y[base + k] = (x[base + k] - mean) / std::sqrt(var + 1.e-5f) * scale[c] + bias[c];
         }
      }
   return y;
}

inline std::vector<float> RefInstanceNorm(const std::vector<float> &x)
{
   auto scale = AffineValues(3, 0.5f, 0.25f), bias = AffineValues(3, -0.5f, 0.1f);
   std::vector<float> y(24);
   for (size_t r = 0; r < 6; r++) {
      float mean = 0, var = 0;
      for (size_t k = 0; k < 4; k++)
         mean += x[r * 4 + k] / 4.f;
      for (size_t k = 0; k < 4; k++)
         var += (x[r * 4 + k] - mean) * (x[r * 4 + k] - mean) / 4.f;
      for (size_t k = 0; k < 4; k++)
         y[r * 4 + k] = scale[r % 3] * (x[r * 4 + k] - mean) / std::sqrt(var + 1.e-5f) + bias[r % 3];
   }
   return y;
}

inline std::vector<float> RefL2Norm(const std::vector<float> &x)
{
   std::vector<float> y(24);
   for (size_t r = 0; r < 6; r++) {
      float norm = 0;
      for (size_t k = 0; k < 4; k++)
         norm += x[r * 4 + k] * x[r * 4 + k];
      norm = std::max(std::sqrt(norm), 1.e-6f);
      for (size_t k = 0; k < 4; k++)
         y[r * 4 + k] = x[r * 4 + k] / norm;
   }
   return y;
}

inline std::vector<float> RefCumSum(const std::vector<float> &x)
{
   std::vector<float> y(24);
   for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 3; j++)
         for (size_t k = 0; k < 4; k++) {
            float acc = 0;
            for (size_t jj = 0; jj <= j; jj++)
               acc += x[(i * 3 + jj) * 4 + k];
            y[(i * 3 + j) * 4 + k] = acc;
         }
   return y;
}

// reverse and exclusive cumulative sum along the last axis
inline std::vector<float> RefCumSumRev(const std::vector<float> &x)
{
   std::vector<float> y(24);
   for (size_t r = 0; r < 6; r++)
      for (size_t k = 0; k < 4; k++) {
         float acc = 0;
         for (size_t kk = k + 1; kk < 4; kk++)
            acc += x[r * 4 + kk];
         y[r * 4 + k] = acc;
      }
   return y;
}

// 2x2 pooling with strides 2 of a 1x2x4x4 tensor
inline std::vector<float> RefPool(const std::vector<float> &x, bool max)
{
   std::vector<float> y(8);
   for (size_t c = 0; c < 2; c++)
      for (size_t oh = 0; oh < 2; oh++)
         for (size_t ow = 0; ow < 2; ow++) {
            float v = max ? -1.e30f : 0.f;
            for (size_t i = 0; i < 2; i++)
               for (size_t j = 0; j < 2; j++) {
                  float e = x[c * 16 + (2 * oh + i) * 4 + 2 * ow + j];
                  v = max ? std::max(v, e) : v + e / 4.f;
               }
            y[c * 4 + oh * 2 + ow] = v;
         }
   return y;
}

// valid convolution of a 2x3x6x6 tensor with the 4x3x3x3 weights and the bias of StridedInputModelGenerator.cxx
inline std::vector<float> RefConv(const std::vector<float> &x)
{
   auto w = AffineValues(108, -1.f, 0.02f), b = AffineValues(4, 0.1f, 0.2f);
   std::vector<float> y(2 * 4 * 4 * 4);
   for (size_t n = 0; n < 2; n++)
      for (size_t oc = 0; oc < 4; oc++)
         for (size_t oh = 0; oh < 4; oh++)
            for (size_t ow = 0; ow < 4; ow++) {
               float acc = b[oc];
               for (size_t ic = 0; ic < 3; ic++)
                  for (size_t kh = 0; kh < 3; kh++)
                     for (size_t kw = 0; kw < 3; kw++)
                        acc += x[((n * 3 + ic) * 6 + oh + kh) * 6 + ow + kw] * w[((oc * 3 + ic) * 3 + kh) * 3 + kw];
               y[((n * 4 + oc) * 4 + oh) * 4 + ow] = acc;
            }
   return y;
}

// transposed convolution of a 1x2x3x3 tensor with the 2x3x2x2 weights and the bias of StridedInputModelGenerator.cxx
inline std::vector<float> RefConvTranspose(const std::vector<float> &x)
{
   auto w = AffineValues(24, -0.5f, 0.05f), b = AffineValues(3, 0.1f, 0.2f);
   std::vector<float> y(3 * 4 * 4);
   for (size_t oc = 0; oc < 3; oc++)
      for (size_t k = 0; k < 16; k++)
         y[oc * 16 + k] = b[oc];
   for (size_t ic = 0; ic < 2; ic++)
      for (size_t oc = 0; oc < 3; oc++)
         for (size_t ih = 0; ih < 3; ih++)
            for (size_t iw = 0; iw < 3; iw++)
               for (size_t kh = 0; kh < 2; kh++)
                  for (size_t kw = 0; kw < 2; kw++)
                     y[(oc * 4 + ih + kh) * 4 + iw + kw] += x[(ic * 3 + ih) * 3 + iw] * w[((ic * 3 + oc) * 2 + kh) * 2 + kw];
   return y;
}

// scaled dot product attention of 1x2x4x3 tensors
inline std::vector<float> RefSDPA(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v)
{
   std::vector<float> y(24);
   for (size_t h = 0; h < 2; h++)
      for (size_t s = 0; s < 4; s++) {
         float scores[4], vmax = -1.e30f, sum = 0;
         for (size_t j = 0; j < 4; j++) {
            float dot = 0;
            for (size_t d = 0; d < 3; d++)
               dot += q[(h * 4 + s) * 3 + d] * k[(h * 4 + j) * 3 + d];
            scores[j] = dot / std::sqrt(3.f);
            vmax = std::max(vmax, scores[j]);
         }
         for (size_t j = 0; j < 4; j++) {
            scores[j] = std::exp(scores[j] - vmax);
            sum += scores[j];
         }
         for (size_t d = 0; d < 3; d++) {
            float acc = 0;
            for (size_t j = 0; j < 4; j++)
               acc += scores[j] / sum * v[(h * 4 + j) * 3 + d];
            y[(h * 4 + s) * 3 + d] = acc;
         }
      }
   return y;
}
