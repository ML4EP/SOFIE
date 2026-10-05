// Tests of the stride aware GPU inference path (Options::kStridedInput): the Session constructor takes the strides
// (in elements) of the input tensors and the kernels of the operators reading the graph inputs access them through
// the strides. StridedInputModelGenerator.cxx emits the headers this file #includes.

#include "TestAlpakaCommon.h"

#include "StridedUnaryChain_GPU_ALPAKA.hxx"
#include "StridedReluTwice_GPU_ALPAKA.hxx"
#include "StridedTanhDyn_GPU_ALPAKA.hxx"
#include "StridedLeakyRelu_GPU_ALPAKA.hxx"
#include "StridedElu_GPU_ALPAKA.hxx"
#include "StridedSelu_GPU_ALPAKA.hxx"
#include "StridedClip_GPU_ALPAKA.hxx"
#include "StridedIdentity_GPU_ALPAKA.hxx"
#include "StridedNot_GPU_ALPAKA.hxx"
#include "StridedBitwiseNot_GPU_ALPAKA.hxx"
#include "StridedCast_GPU_ALPAKA.hxx"
#include "StridedIsNaN_GPU_ALPAKA.hxx"
#include "StridedBinary_GPU_ALPAKA.hxx"
#include "StridedCompare_GPU_ALPAKA.hxx"
#include "StridedBitwiseAnd_GPU_ALPAKA.hxx"
#include "StridedNarySum_GPU_ALPAKA.hxx"
#include "StridedWhere_GPU_ALPAKA.hxx"
#include "StridedIf_GPU_ALPAKA.hxx"
#include "StridedTranspose_GPU_ALPAKA.hxx"
#include "StridedSlice_GPU_ALPAKA.hxx"
#include "StridedGather_GPU_ALPAKA.hxx"
#include "StridedConcat_GPU_ALPAKA.hxx"
#include "StridedSplit_GPU_ALPAKA.hxx"
#include "StridedPad_GPU_ALPAKA.hxx"
#include "StridedTile_GPU_ALPAKA.hxx"
#include "StridedExpand_GPU_ALPAKA.hxx"
#include "StridedSoftmaxMid_GPU_ALPAKA.hxx"
#include "StridedSoftmaxLast_GPU_ALPAKA.hxx"
#include "StridedLogSoftmax_GPU_ALPAKA.hxx"
#include "StridedReduceSumMid_GPU_ALPAKA.hxx"
#include "StridedReduceMeanLast_GPU_ALPAKA.hxx"
#include "StridedReduceMaxFirst_GPU_ALPAKA.hxx"
#include "StridedLayerNorm_GPU_ALPAKA.hxx"
#include "StridedRMSNorm_GPU_ALPAKA.hxx"
#include "StridedBatchNorm_GPU_ALPAKA.hxx"
#include "StridedGroupNorm_GPU_ALPAKA.hxx"
#include "StridedL2Norm_GPU_ALPAKA.hxx"
#include "StridedCumSum_GPU_ALPAKA.hxx"
#include "StridedCumSumRev_GPU_ALPAKA.hxx"
#include "StridedMaxPool_GPU_ALPAKA.hxx"
#include "StridedAvgPool_GPU_ALPAKA.hxx"
#include "StridedTopK_GPU_ALPAKA.hxx"
#include "StridedReshape_GPU_ALPAKA.hxx"
#include "StridedFlatten_GPU_ALPAKA.hxx"
#include "StridedTrilu_GPU_ALPAKA.hxx"
#include "StridedNonZero_GPU_ALPAKA.hxx"
#include "StridedGatherND_GPU_ALPAKA.hxx"
#include "StridedGatherNDElems_GPU_ALPAKA.hxx"
#include "StridedScatterND_GPU_ALPAKA.hxx"
#include "StridedScatterElements_GPU_ALPAKA.hxx"
#include "StridedConv_GPU_ALPAKA.hxx"
#include "StridedSDPA_GPU_ALPAKA.hxx"
#include "StridedMamba_GPU_ALPAKA.hxx"
#include "StridedRWKV_GPU_ALPAKA.hxx"
#include "StridedGriffin_GPU_ALPAKA.hxx"
#include "StridedGemmAB_GPU_ALPAKA.hxx"
#include "StridedGemmABt_GPU_ALPAKA.hxx"
#include "StridedGemmBiasRelu_GPU_ALPAKA.hxx"
#include "StridedGemmTransAB_GPU_ALPAKA.hxx"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

namespace {

// Backing storage of a strided view: element (i0, i1, ...) of the logical tensor `shape` lives at
// sum_d(i_d * strides[d]). Elements not belonging to the view are set to a poison value.
template <typename T = float>
struct StridedData {
   std::vector<T> storage;
   std::vector<T> logical; // the values of the logical tensor, contiguous row-major

   StridedData(const std::vector<size_t> &shape, const std::vector<size_t> &strides,
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

// same deterministic weights as StridedInputModelGenerator.cxx
std::vector<float> GemmValues(size_t n, float scale)
{
   std::vector<float> v(n);
   for (size_t i = 0; i < n; i++)
      v[i] = scale * float((i * 7) % 11) - 0.5f;
   return v;
}

// Y = Relu(X * W + b), X 3x5, W 5x4
std::vector<float> RefGemmBiasRelu(const std::vector<float> &x)
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
std::vector<float> RefGemmTransAB(const std::vector<float> &x)
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

// ---------------------------------------------------------------------------------------------------------------
// reductions and normalizations: the references work on the logical (contiguous) tensor
// ---------------------------------------------------------------------------------------------------------------

std::vector<float> AffineValues(size_t n, float a, float b)
{
   std::vector<float> v(n);
   for (size_t i = 0; i < n; i++)
      v[i] = a + b * float(i);
   return v;
}

// softmax (or log softmax) over `axis` of a 2x3x4 tensor
std::vector<float> RefSoftmax(const std::vector<float> &x, size_t axis, bool logSoftmax)
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

std::vector<float> RefReduceSumMid(const std::vector<float> &x)
{
   std::vector<float> y(8, 0.f);
   for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 3; j++)
         for (size_t k = 0; k < 4; k++)
            y[i * 4 + k] += x[(i * 3 + j) * 4 + k];
   return y;
}

std::vector<float> RefReduceMeanLast(const std::vector<float> &x)
{
   std::vector<float> y(6, 0.f);
   for (size_t r = 0; r < 6; r++)
      for (size_t k = 0; k < 4; k++)
         y[r] += x[r * 4 + k] / 4.f;
   return y;
}

std::vector<float> RefReduceMaxFirst(const std::vector<float> &x)
{
   std::vector<float> y(12);
   for (size_t k = 0; k < 12; k++)
      y[k] = std::max(x[k], x[12 + k]);
   return y;
}

// layer normalization over the dimensions {3, 4} of each of the 2 batches (scale and bias of shape {3, 4})
std::vector<float> RefLayerNorm(const std::vector<float> &x)
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

std::vector<float> RefRMSNorm(const std::vector<float> &x)
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
std::vector<float> RefBatchNorm(const std::vector<float> &x)
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
std::vector<float> RefGroupNorm(const std::vector<float> &x)
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

std::vector<float> RefInstanceNorm(const std::vector<float> &x)
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

std::vector<float> RefL2Norm(const std::vector<float> &x)
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

std::vector<float> RefCumSum(const std::vector<float> &x)
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
std::vector<float> RefCumSumRev(const std::vector<float> &x)
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



// ---------------------------------------------------------------------------------------------------------------
// pooling and top-k
// ---------------------------------------------------------------------------------------------------------------

// 2x2 pooling with strides 2 of a 1x2x4x4 tensor
std::vector<float> RefPool(const std::vector<float> &x, bool max)
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

// ---------------------------------------------------------------------------------------------------------------
// convolutions, einsum, attention, recurrent operators and scans of strided inputs
// ---------------------------------------------------------------------------------------------------------------

// valid convolution of a 2x3x6x6 tensor with the 4x3x3x3 weights and the bias of StridedInputModelGenerator.cxx
std::vector<float> RefConv(const std::vector<float> &x)
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

// scaled dot product attention of 1x2x4x3 tensors
std::vector<float> RefSDPA(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v)
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


} // namespace

class SofieAlpakaStridedTest : public SofieAlpakaTest {
protected:
   // run `infer` on the strided data (already on the device) and compare with `ref`
   template <typename Result, typename T>
   void Check(Result &result, const std::vector<T> &ref)
   {
      alpaka::wait(queue);
      // the output buffer can be larger than the result (e.g. NonZero allocates for the worst case): only the first
      // elements, compared with the reference, are meaningful
      const Idx length = alpaka::getExtentProduct(result);
      ASSERT_GE(length, ref.size());
      auto result_h = alpaka::allocBuf<T, Idx>(host, Ext1D::all(length));
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
      const T *res = reinterpret_cast<const T *>(alpaka::getPtrNative(result_h));
      for (size_t i = 0; i < ref.size(); ++i)
         EXPECT_LE(std::abs(double(res[i]) - double(ref[i])), DEFAULT_TOLERANCE * std::max(1.0, std::fabs(double(ref[i])))) << "i=" << i;
   }

   // two host results of the same length
   void Check2(const std::vector<float> &res, const std::vector<float> &ref)
   {
      ASSERT_EQ(res.size(), ref.size());
      for (size_t i = 0; i < ref.size(); ++i)
         EXPECT_LE(std::abs(double(res[i]) - double(ref[i])), 1.e-5) << "i=" << i;
   }

   template <typename T>
   auto ToDevice(const StridedData<T> &d)
   {
      return makeDeviceBuf<T>(host, device, queue, d.storage.data(), d.storage.size());
   }
};

TEST_F(SofieAlpakaStridedTest, ContiguousDefault)
{
   StridedData<> in({3, 5}, {5, 1});
   auto input_d = makeDeviceBuf<float>(host, device, queue, in.storage.data(), in.storage.size());
   SOFIE_StridedUnaryChain::Session<alpaka::TagGpuCudaRt> session;
   auto result = session.infer(input_d);
   std::vector<float> ref(15);
   for (size_t i = 0; i < 15; i++)
      ref[i] = 1.f / (1.f + std::exp(in.logical[i]));
   Check(result, ref);
}

TEST_F(SofieAlpakaStridedTest, RowPadding)
{
   StridedData<> in({3, 5}, {8, 1});
   auto input_d = makeDeviceBuf<float>(host, device, queue, in.storage.data(), in.storage.size());
   SOFIE_StridedUnaryChain::Session<alpaka::TagGpuCudaRt> session("", {{8, 1}});
   auto result = session.infer(input_d);
   std::vector<float> ref(15);
   for (size_t i = 0; i < 15; i++)
      ref[i] = 1.f / (1.f + std::exp(in.logical[i]));
   Check(result, ref);
}

TEST_F(SofieAlpakaStridedTest, Transposed)
{
   StridedData<> in({3, 5}, {1, 3});
   auto input_d = makeDeviceBuf<float>(host, device, queue, in.storage.data(), in.storage.size());
   SOFIE_StridedUnaryChain::Session<alpaka::TagGpuCudaRt> session("", {{1, 3}});
   auto result = session.infer(input_d);
   std::vector<float> ref(15);
   for (size_t i = 0; i < 15; i++)
      ref[i] = 1.f / (1.f + std::exp(in.logical[i]));
   Check(result, ref);
}

TEST_F(SofieAlpakaStridedTest, SameOperatorKindOnStridedAndContiguousInput)
{
   StridedData<> in({2, 3, 4}, {1, 8, 2});
   auto input_d = makeDeviceBuf<float>(host, device, queue, in.storage.data(), in.storage.size());
   SOFIE_StridedReluTwice::Session<alpaka::TagGpuCudaRt> session("", {{1, 8, 2}});
   auto result = session.infer(input_d);
   std::vector<float> ref(24);
   for (size_t i = 0; i < 24; i++)
      ref[i] = std::max(in.logical[i], 0.f);
   Check(result, ref);
}

TEST_F(SofieAlpakaStridedTest, DynamicShapeStridedSubBatch)
{
   // maximum batch 8, run a batch of 6 rows read from a matrix with leading dimension 7
   StridedData<> in({6, 5}, {7, 1});
   auto input_d = makeDeviceBuf<float>(host, device, queue, in.storage.data(), in.storage.size());
   SOFIE_StridedTanhDyn::Session<alpaka::TagGpuCudaRt> session("", 8, {{7, 1}});
   auto result = session.infer(6, input_d);
   std::vector<float> ref(30);
   for (size_t i = 0; i < 30; i++)
      ref[i] = std::tanh(in.logical[i]);
   Check(result, ref);
}

TEST_F(SofieAlpakaStridedTest, InvalidStridesAreRejected)
{
   EXPECT_THROW((SOFIE_StridedUnaryChain::Session<alpaka::TagGpuCudaRt>("", {{5, 1}, {5, 1}})), std::runtime_error);
   EXPECT_THROW((SOFIE_StridedUnaryChain::Session<alpaka::TagGpuCudaRt>("", {{5, 1, 1}})), std::runtime_error);
}

TEST_F(SofieAlpakaStridedTest, GemmBiasReluLayouts)
{
   // row-major, row padded and column-major 3x5 inputs
   for (const auto &strides : std::vector<std::vector<size_t>>{{5, 1}, {8, 1}, {1, 3}}) {
      StridedData<> in({3, 5}, strides);
      auto input_d = makeDeviceBuf<float>(host, device, queue, in.storage.data(), in.storage.size());
      SOFIE_StridedGemmBiasRelu::Session<alpaka::TagGpuCudaRt> session("StridedGemmBiasRelu_GPU_ALPAKA.dat", {strides});
      auto result = session.infer(input_d);
      Check(result, RefGemmBiasRelu(in.logical));
   }
}

TEST_F(SofieAlpakaStridedTest, GemmTransposedOperands)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{3, 1}, {8, 1}, {1, 5}}) {
      StridedData<> in({5, 3}, strides);
      auto input_d = makeDeviceBuf<float>(host, device, queue, in.storage.data(), in.storage.size());
      SOFIE_StridedGemmTransAB::Session<alpaka::TagGpuCudaRt> session("StridedGemmTransAB_GPU_ALPAKA.dat", {strides});
      auto result = session.infer(input_d);
      Check(result, RefGemmTransAB(in.logical));
   }
}

TEST_F(SofieAlpakaStridedTest, GemmWithoutUnitStrideThrows)
{
   StridedData<> in({3, 5}, {20, 2});
   auto input_d = makeDeviceBuf<float>(host, device, queue, in.storage.data(), in.storage.size());
   SOFIE_StridedGemmBiasRelu::Session<alpaka::TagGpuCudaRt> session("StridedGemmBiasRelu_GPU_ALPAKA.dat", {{20, 2}});
   EXPECT_THROW(session.infer(input_d), std::runtime_error);
}

// ---------------------------------------------------------------------------------------------------------------
// elementwise operators: a permuted 2x3x4 view (strides {1, 8, 2}) and a padded one (strides {16, 5, 1})
// ---------------------------------------------------------------------------------------------------------------

// one model per operator: the operators of a model sharing a memory pool slot would overwrite each other's output
#define STRIDED_UNARY_TEST(NAME, TYPE, GENERATOR, REFERENCE)                                                           \
   TEST_F(SofieAlpakaStridedTest, NAME)                                                                               \
   {                                                                                                                  \
      for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}, {12, 4, 1}}) {              \
         StridedData<TYPE> in({2, 3, 4}, strides, GENERATOR);                                                         \
         auto input_d = ToDevice(in);                                                                                 \
         SOFIE_Strided##NAME::Session<alpaka::TagGpuCudaRt> session("", {strides});                                  \
         auto y = session.infer(input_d);                                                                             \
         std::vector<TYPE> ref;                                                                                       \
         for (TYPE x : in.logical)                                                                                    \
            ref.push_back(REFERENCE);                                                                                 \
         Check(y, ref);                                                                                               \
      }                                                                                                               \
   }

STRIDED_UNARY_TEST(LeakyRelu, float, nullptr, x >= 0 ? x : 0.1f * x)
STRIDED_UNARY_TEST(Elu, float, nullptr, x >= 0 ? x : 1.5f * (std::exp(x) - 1))
STRIDED_UNARY_TEST(Selu, float, nullptr, 1.05f * (std::max(0.f, x) + std::min(0.f, 1.6f * (std::exp(x) - 1))))
STRIDED_UNARY_TEST(Clip, float, nullptr, std::min(1.f, std::max(-0.5f, x)))
STRIDED_UNARY_TEST(Identity, float, nullptr, x)
STRIDED_UNARY_TEST(Not, int32_t, [](size_t k) { return int32_t(k % 5) - 2; }, !x)
STRIDED_UNARY_TEST(BitwiseNot, int32_t, [](size_t k) { return int32_t(k % 5) - 2; }, ~x)

TEST_F(SofieAlpakaStridedTest, CastAndIsNaN)
{
   StridedData<> in({2, 3, 4}, {1, 8, 2}, [](size_t k) { return 0.9f * float(k) - 7.3f; });
   auto input_d = ToDevice(in);
   SOFIE_StridedCast::Session<alpaka::TagGpuCudaRt> cast("", {{1, 8, 2}});
   auto castResult = cast.infer(input_d);
   std::vector<int32_t> castRef;
   for (float x : in.logical)
      castRef.push_back(static_cast<int32_t>(x));
   Check(castResult, castRef);

   StridedData<> withNaN({2, 3, 4}, {16, 5, 1}, [](size_t k) { return k % 5 == 0 ? std::nanf("") : float(k); });
   auto nan_d = ToDevice(withNaN);
   SOFIE_StridedIsNaN::Session<alpaka::TagGpuCudaRt> isnan("", {{16, 5, 1}});
   auto nanResult = isnan.infer(nan_d);
   std::vector<uint8_t> nanRef;
   for (float x : withNaN.logical)
      nanRef.push_back(std::isnan(x));
   Check(nanResult, nanRef);
}

// ---------------------------------------------------------------------------------------------------------------
// operators with several (strided) inputs and broadcasting
// ---------------------------------------------------------------------------------------------------------------

TEST_F(SofieAlpakaStridedTest, BinaryWithBroadcast)
{
   StridedData<> a({2, 3, 4}, {1, 8, 2});
   StridedData<> b({3, 1}, {2, 7}, [](size_t k) { return 1.5f * float(k) + 0.25f; });
   auto a_d = ToDevice(a);
   auto b_d = ToDevice(b);
   SOFIE_StridedBinary::Session<alpaka::TagGpuCudaRt> session("", {{1, 8, 2}, {2, 7}});
   auto y = session.infer(a_d, b_d);
   std::vector<float> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = a.logical[k] - b.logical[(k / 4) % 3];
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, BinaryOnlyOneInputStrided)
{
   StridedData<> a({2, 3, 4}, {12, 4, 1});
   StridedData<> b({3, 1}, {2, 7}, [](size_t k) { return 1.5f * float(k) + 0.25f; });
   auto a_d = ToDevice(a);
   auto b_d = ToDevice(b);
   SOFIE_StridedBinary::Session<alpaka::TagGpuCudaRt> session("", {{}, {2, 7}});
   auto y = session.infer(a_d, b_d);
   std::vector<float> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = a.logical[k] - b.logical[(k / 4) % 3];
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, Comparison)
{
   StridedData<> a({2, 3, 4}, {1, 8, 2});
   StridedData<> b({4}, {3}, [](size_t k) { return 0.4f * float(k); });
   auto a_d = ToDevice(a);
   auto b_d = ToDevice(b);
   SOFIE_StridedCompare::Session<alpaka::TagGpuCudaRt> session("", {{1, 8, 2}, {3}});
   auto y = session.infer(a_d, b_d);
   std::vector<uint8_t> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = a.logical[k] > b.logical[k % 4];
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, BitwiseAnd)
{
   StridedData<int32_t> a({2, 3, 4}, {12, 4, 1}, [](size_t k) { return int32_t(3 * k + 1); });
   StridedData<int32_t> b({2, 3, 4}, {1, 8, 2}, [](size_t k) { return int32_t(5 * k + 2); });
   auto a_d = ToDevice(a);
   auto b_d = ToDevice(b);
   SOFIE_StridedBitwiseAnd::Session<alpaka::TagGpuCudaRt> session("", {{12, 4, 1}, {1, 8, 2}});
   auto y = session.infer(a_d, b_d);
   std::vector<int32_t> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = a.logical[k] & b.logical[k];
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, NarySum)
{
   StridedData<> a({2, 3, 4}, {1, 8, 2});
   StridedData<> b({3, 4}, {1, 3}, [](size_t k) { return 0.1f * float(k); });
   StridedData<> c({4}, {2}, [](size_t k) { return 10.f * float(k); });
   auto a_d = ToDevice(a);
   auto b_d = ToDevice(b);
   auto c_d = ToDevice(c);
   SOFIE_StridedNarySum::Session<alpaka::TagGpuCudaRt> session("", {{1, 8, 2}, {1, 3}, {2}});
   auto y = session.infer(a_d, b_d, c_d);
   std::vector<float> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = a.logical[k] + b.logical[k % 12] + c.logical[k % 4];
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, Where)
{
   StridedData<uint8_t> c({3, 1}, {2, 5}, [](size_t k) { return uint8_t(k % 2 == 0); });
   StridedData<> x({2, 3, 4}, {1, 8, 2});
   StridedData<> y({4}, {3}, [](size_t k) { return -3.f * float(k); });
   auto c_d = ToDevice(c);
   auto x_d = ToDevice(x);
   auto y_d = ToDevice(y);
   SOFIE_StridedWhere::Session<alpaka::TagGpuCudaRt> session("", {{2, 5}, {1, 8, 2}, {3}});
   auto z = session.infer(c_d, x_d, y_d);
   std::vector<float> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = c.logical[(k / 4) % 3] ? x.logical[k] : y.logical[k % 4];
   Check(z, ref);
}

// ---------------------------------------------------------------------------------------------------------------
// operators moving the data of strided inputs
// ---------------------------------------------------------------------------------------------------------------

TEST_F(SofieAlpakaStridedTest, Transpose)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}, {12, 4, 1}}) {
      StridedData<> in({2, 3, 4}, strides);
      auto in_d = ToDevice(in);
      SOFIE_StridedTranspose::Session<alpaka::TagGpuCudaRt> s("", {strides});
      std::vector<float> ref(24);
      for (size_t l = 0; l < 4; l++)
         for (size_t i = 0; i < 2; i++)
            for (size_t j = 0; j < 3; j++)
               ref[(l * 2 + i) * 3 + j] = in.logical[(i * 3 + j) * 4 + l];
      auto y = s.infer(in_d);
      Check(y, ref);
   }
}

TEST_F(SofieAlpakaStridedTest, Slice)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{6, 1}, {9, 1}, {1, 4}, {2, 9}}) {
      StridedData<> in({4, 6}, strides);
      auto in_d = ToDevice(in);
      SOFIE_StridedSlice::Session<alpaka::TagGpuCudaRt> s("", {strides});
      std::vector<float> ref;
      for (size_t i = 0; i < 2; i++)
         for (size_t j = 0; j < 2; j++)
            ref.push_back(in.logical[(2 * i) * 6 + 1 + 2 * j]);
      auto y = s.infer(in_d);
      Check(y, ref);
   }
}

TEST_F(SofieAlpakaStridedTest, Gather)
{
   StridedData<> x({4, 5}, {1, 4});
   StridedData<int64_t> idx({3}, {2}, [](size_t k) { return int64_t((2 * k + 3) % 4); });
   auto x_d = ToDevice(x);
   auto idx_d = ToDevice(idx);
   SOFIE_StridedGather::Session<alpaka::TagGpuCudaRt> s("", {{1, 4}, {2}});
   std::vector<float> ref;
   for (size_t i = 0; i < 3; i++)
      for (size_t c = 0; c < 5; c++)
         ref.push_back(x.logical[idx.logical[i] * 5 + c]);
   auto y = s.infer(x_d, idx_d);
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, ConcatMixedStridedAndContiguousInputs)
{
   StridedData<> a({2, 3}, {1, 2});
   StridedData<> b({2, 2}, {2, 1}, [](size_t k) { return 100.f + float(k); });
   StridedData<> c({2, 1}, {5, 3}, [](size_t k) { return 200.f + float(k); });
   auto a_d = ToDevice(a);
   auto b_d = ToDevice(b);
   auto c_d = ToDevice(c);
   SOFIE_StridedConcat::Session<alpaka::TagGpuCudaRt> s("", {{1, 2}, {}, {5, 3}});
   std::vector<float> ref;
   for (size_t r = 0; r < 2; r++) {
      for (size_t j = 0; j < 3; j++)
         ref.push_back(a.logical[r * 3 + j]);
      for (size_t j = 0; j < 2; j++)
         ref.push_back(b.logical[r * 2 + j]);
      ref.push_back(c.logical[r]);
   }
   auto y = s.infer(a_d, b_d, c_d);
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, Split)
{
   StridedData<> in({2, 6}, {1, 2});
   auto in_d = ToDevice(in);
   SOFIE_StridedSplit::Session<alpaka::TagGpuCudaRt> s("", {{1, 2}});
   auto y = s.infer(in_d);
   std::vector<float> ref0, ref1;
   for (size_t r = 0; r < 2; r++)
      for (size_t j = 0; j < 3; j++) {
         ref0.push_back(in.logical[r * 6 + j]);
         ref1.push_back(in.logical[r * 6 + 3 + j]);
      }
   Check(std::get<0>(y), ref0);
   Check(std::get<1>(y), ref1);
}

TEST_F(SofieAlpakaStridedTest, Pad)
{
   StridedData<> in({2, 3}, {1, 2});
   auto in_d = ToDevice(in);
   SOFIE_StridedPad::Session<alpaka::TagGpuCudaRt> s("StridedPad_GPU_ALPAKA.dat", {{1, 2}});
   std::vector<float> ref(20, 0.f);
   for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 3; j++)
         ref[(i + 1) * 5 + j] = in.logical[i * 3 + j];
   auto y = s.infer(in_d);
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, Tile)
{
   StridedData<> in({2, 3}, {1, 2});
   auto in_d = ToDevice(in);
   SOFIE_StridedTile::Session<alpaka::TagGpuCudaRt> s("", {{1, 2}});
   std::vector<float> ref(36);
   for (size_t i = 0; i < 4; i++)
      for (size_t j = 0; j < 9; j++)
         ref[i * 9 + j] = in.logical[(i % 2) * 3 + (j % 3)];
   auto y = s.infer(in_d);
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, Expand)
{
   StridedData<> in({3, 1}, {2, 7});
   auto in_d = ToDevice(in);
   SOFIE_StridedExpand::Session<alpaka::TagGpuCudaRt> s("", {{2, 7}});
   std::vector<float> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = in.logical[(k / 4) % 3];
   auto y = s.infer(in_d);
   Check(y, ref);
}

#define STRIDED_X_TEST(NAME, WEIGHTS, SHAPE, REFERENCE)                                                                \
   TEST_F(SofieAlpakaStridedTest, NAME)                                                                               \
   {                                                                                                                  \
      for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}, {12, 4, 1}}) {              \
         StridedData<> in(SHAPE, strides);                                                                            \
         auto in_d = ToDevice(in);                                                                                    \
         SOFIE_Strided##NAME::Session<alpaka::TagGpuCudaRt> s(WEIGHTS, {strides});                                   \
         auto y = s.infer(in_d);                                                                                      \
         Check(y, std::vector<float>(REFERENCE(in.logical)));                                                         \
      }                                                                                                               \
   }

STRIDED_X_TEST(SoftmaxMid, "", (std::vector<size_t>{2, 3, 4}), [](const std::vector<float> &x) { return RefSoftmax(x, 1, false); })
STRIDED_X_TEST(SoftmaxLast, "", (std::vector<size_t>{2, 3, 4}), [](const std::vector<float> &x) { return RefSoftmax(x, 2, false); })
STRIDED_X_TEST(LogSoftmax, "", (std::vector<size_t>{2, 3, 4}), [](const std::vector<float> &x) { return RefSoftmax(x, 2, true); })
STRIDED_X_TEST(ReduceSumMid, "", (std::vector<size_t>{2, 3, 4}), RefReduceSumMid)
STRIDED_X_TEST(ReduceMeanLast, "", (std::vector<size_t>{2, 3, 4}), RefReduceMeanLast)
STRIDED_X_TEST(ReduceMaxFirst, "", (std::vector<size_t>{2, 3, 4}), RefReduceMaxFirst)
STRIDED_X_TEST(LayerNorm, "StridedLayerNorm_GPU_ALPAKA.dat", (std::vector<size_t>{2, 3, 4}), RefLayerNorm)
STRIDED_X_TEST(RMSNorm, "StridedRMSNorm_GPU_ALPAKA.dat", (std::vector<size_t>{2, 3, 4}), RefRMSNorm)
STRIDED_X_TEST(BatchNorm, "StridedBatchNorm_GPU_ALPAKA.dat", (std::vector<size_t>{2, 3, 4}), RefBatchNorm)
STRIDED_X_TEST(L2Norm, "", (std::vector<size_t>{2, 3, 4}), RefL2Norm)
STRIDED_X_TEST(CumSum, "", (std::vector<size_t>{2, 3, 4}), RefCumSum)
STRIDED_X_TEST(CumSumRev, "", (std::vector<size_t>{2, 3, 4}), RefCumSumRev)

TEST_F(SofieAlpakaStridedTest, GroupNorm)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 3}, {40, 5, 1}, {12, 3, 1}}) {
      StridedData<> in({2, 4, 3}, strides);
      auto in_d = ToDevice(in);
      SOFIE_StridedGroupNorm::Session<alpaka::TagGpuCudaRt> s("StridedGroupNorm_GPU_ALPAKA.dat", {strides});
      auto y = s.infer(in_d);
      Check(y, RefGroupNorm(in.logical));
   }
}

TEST_F(SofieAlpakaStridedTest, Pooling)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{32, 16, 4, 1}, {64, 24, 5, 1}, {32, 1, 8, 2}}) {
      StridedData<> in({1, 2, 4, 4}, strides);
      auto in_d = ToDevice(in);
      SOFIE_StridedMaxPool::Session<alpaka::TagGpuCudaRt> maxPool("", {strides});
      auto yMax = maxPool.infer(in_d);
      Check(yMax, RefPool(in.logical, true));
      SOFIE_StridedAvgPool::Session<alpaka::TagGpuCudaRt> avgPool("", {strides});
      auto yAvg = avgPool.infer(in_d);
      Check(yAvg, RefPool(in.logical, false));
   }
}

TEST_F(SofieAlpakaStridedTest, TopK)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}, {12, 4, 1}}) {
      StridedData<> in({2, 3, 4}, strides, [](size_t k) { return float((k * 13) % 24); });
      auto in_d = ToDevice(in);
      SOFIE_StridedTopK::Session<alpaka::TagGpuCudaRt> s("", {strides});
      auto result = s.infer(in_d);
      std::vector<float> refValues;
      std::vector<int64_t> refIndices;
      for (size_t i = 0; i < 2; i++)
         for (size_t r = 0; r < 2; r++)
            for (size_t k = 0; k < 4; k++) {
               std::vector<std::pair<float, int64_t>> col;
               for (size_t j = 0; j < 3; j++)
                  col.push_back({in.logical[(i * 3 + j) * 4 + k], int64_t(j)});
               std::sort(col.begin(), col.end(), [](auto &a, auto &b) { return a.first > b.first; });
               refValues.push_back(col[r].first);
               refIndices.push_back(col[r].second);
            }
      Check(std::get<0>(result), refValues);
      Check(std::get<1>(result), refIndices);
   }
}

// ---------------------------------------------------------------------------------------------------------------
// shape changes, triangular part, non zero elements, gather and scatter of strided inputs
// ---------------------------------------------------------------------------------------------------------------

TEST_F(SofieAlpakaStridedTest, ReshapeAndFlatten)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}, {12, 4, 1}}) {
      StridedData<> in({2, 3, 4}, strides);
      auto in_d = ToDevice(in);
      SOFIE_StridedReshape::Session<alpaka::TagGpuCudaRt> reshape("", {strides});
      auto y = reshape.infer(in_d);
      Check(y, in.logical);
      SOFIE_StridedFlatten::Session<alpaka::TagGpuCudaRt> flatten("", {strides});
      auto yf = flatten.infer(in_d);
      Check(yf, in.logical);
   }
}

TEST_F(SofieAlpakaStridedTest, Trilu)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}}) {
      StridedData<> in({2, 3, 4}, strides);
      auto in_d = ToDevice(in);
      SOFIE_StridedTrilu::Session<alpaka::TagGpuCudaRt> s("", {strides});
      std::vector<float> ref(24);
      for (size_t b = 0; b < 2; b++)
         for (size_t r = 0; r < 3; r++)
            for (size_t c = 0; c < 4; c++)
               ref[(b * 3 + r) * 4 + c] = c >= r ? in.logical[(b * 3 + r) * 4 + c] : 0.f;
      auto y = s.infer(in_d);
      Check(y, ref);
   }
}

TEST_F(SofieAlpakaStridedTest, NonZero)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}}) {
      StridedData<> in({2, 3, 4}, strides, [](size_t k) { return k % 3 == 0 ? 0.f : float(k); });
      auto in_d = ToDevice(in);
      SOFIE_StridedNonZero::Session<alpaka::TagGpuCudaRt> s("", {strides});
      std::vector<int64_t> nonZero[3];
      for (size_t k = 0; k < 24; k++)
         if (in.logical[k] != 0.f) {
            nonZero[0].push_back(int64_t(k / 12));
            nonZero[1].push_back(int64_t((k / 4) % 3));
            nonZero[2].push_back(int64_t(k % 4));
         }
      std::vector<int64_t> ref;
      for (auto &v : nonZero)
         ref.insert(ref.end(), v.begin(), v.end());
      auto y = s.infer(in_d);
      // the output buffer has the capacity for all the elements: only the first rank * count are meaningful
      Check(y, ref);
   }
}

TEST_F(SofieAlpakaStridedTest, GatherND)
{
   StridedData<> x({4, 5}, {1, 4});
   StridedData<int64_t> rows({3, 1}, {2, 5}, [](size_t k) { return int64_t((2 * k + 3) % 4); });
   auto x_d = ToDevice(x);
   auto rows_d = ToDevice(rows);
   SOFIE_StridedGatherND::Session<alpaka::TagGpuCudaRt> s("", {{1, 4}, {2, 5}});
   std::vector<float> ref;
   for (size_t i = 0; i < 3; i++)
      for (size_t c = 0; c < 5; c++)
         ref.push_back(x.logical[rows.logical[i] * 5 + c]);
   auto y = s.infer(x_d, rows_d);
   Check(y, ref);

   StridedData<int64_t> elems({3, 2}, {1, 3}, [](size_t k) { return int64_t(k % 4 == 0 ? 3 : (k % 5)); });
   auto elems_d = ToDevice(elems);
   SOFIE_StridedGatherNDElems::Session<alpaka::TagGpuCudaRt> e("", {{1, 4}, {1, 3}});
   std::vector<float> refElems;
   for (size_t i = 0; i < 3; i++)
      refElems.push_back(x.logical[elems.logical[i * 2] * 5 + elems.logical[i * 2 + 1]]);
   auto ye = e.infer(x_d, elems_d);
   Check(ye, refElems);
}

TEST_F(SofieAlpakaStridedTest, ScatterND)
{
   StridedData<> x({4, 5}, {1, 4});
   StridedData<int64_t> idx({2, 1}, {3, 7}, [](size_t k) { return int64_t(k == 0 ? 2 : 0); });
   StridedData<> upd({2, 5}, {1, 2}, [](size_t k) { return 100.f + float(k); });
   auto x_d = ToDevice(x);
   auto idx_d = ToDevice(idx);
   auto upd_d = ToDevice(upd);
   SOFIE_StridedScatterND::Session<alpaka::TagGpuCudaRt> s("", {{1, 4}, {3, 7}, {1, 2}});
   std::vector<float> ref = x.logical;
   for (size_t n = 0; n < 2; n++)
      for (size_t c = 0; c < 5; c++)
         ref[idx.logical[n] * 5 + c] = upd.logical[n * 5 + c];
   auto y = s.infer(x_d, idx_d, upd_d);
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, ScatterElements)
{
   StridedData<> x({3, 4}, {1, 3});
   StridedData<int64_t> idx({2, 4}, {1, 2}, [](size_t k) { return int64_t(k < 4 ? k % 3 : (k + 1) % 3); });
   StridedData<> upd({2, 4}, {8, 2}, [](size_t k) { return 100.f + float(k); });
   auto x_d = ToDevice(x);
   auto idx_d = ToDevice(idx);
   auto upd_d = ToDevice(upd);
   SOFIE_StridedScatterElements::Session<alpaka::TagGpuCudaRt> s("", {{1, 3}, {1, 2}, {8, 2}});
   std::vector<float> ref = x.logical;
   for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 4; j++)
         ref[idx.logical[i * 4 + j] * 4 + j] = upd.logical[i * 4 + j];
   auto y = s.infer(x_d, idx_d, upd_d);
   Check(y, ref);
}

TEST_F(SofieAlpakaStridedTest, Convolution)
{
   // NCHW, NCHW with a padded channel dimension and NHWC layouts
   for (const auto &strides : std::vector<std::vector<size_t>>{{108, 36, 6, 1}, {200, 50, 6, 1}, {108, 1, 18, 3}}) {
      StridedData<> in({2, 3, 6, 6}, strides);
      auto in_d = ToDevice(in);
      SOFIE_StridedConv::Session<alpaka::TagGpuCudaRt> s("StridedConv_GPU_ALPAKA.dat", {strides});
      auto y = s.infer(in_d);
      Check(y, RefConv(in.logical));
   }
}

TEST_F(SofieAlpakaStridedTest, ScaledDotProductAttention)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{24, 12, 3, 1}, {60, 30, 6, 1}, {24, 3, 6, 1}}) {
      StridedData<> q({1, 2, 4, 3}, strides, [](size_t k) { return 0.3f * float((k * 5) % 7) - 1.f; });
      StridedData<> k({1, 2, 4, 3}, strides, [](size_t i) { return 0.2f * float((i * 3) % 11) - 1.f; });
      StridedData<> v({1, 2, 4, 3}, strides, [](size_t i) { return 0.5f * float(i % 5) - 1.f; });
      auto q_d = ToDevice(q);
      auto k_d = ToDevice(k);
      auto v_d = ToDevice(v);
      SOFIE_StridedSDPA::Session<alpaka::TagGpuCudaRt> s("", {strides, strides, strides});
      auto y = s.infer(q_d, k_d, v_d);
      Check(y, RefSDPA(q.logical, k.logical, v.logical));
   }
}

// the result of a strided input must be the one of the same data stored contiguously (a fresh Session each time since
// the scans keep a state)
TEST_F(SofieAlpakaStridedTest, ScanOperators)
{
   auto gen = [](size_t salt) {
      return [salt](size_t k) { return 0.15f * float((k * 7 + salt * 3) % 9) - 0.5f; };
   };
   auto toHost = [&](auto &result, size_t n) {
      alpaka::wait(queue);
      auto h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{n}));
      alpaka::memcpy(queue, h, result);
      alpaka::wait(queue);
      const float *p = reinterpret_cast<const float *>(alpaka::getPtrNative(h));
      return std::vector<float>(p, p + n);
   };
   // Mamba: u [1,2,4], delta [1,2,4], A [2,3], B [1,3,4], C [1,3,4], D [2]
   {
      auto run = [&](const std::vector<std::vector<size_t>> &st) {
         StridedData<> u({1, 2, 4}, st[0], gen(1)), delta({1, 2, 4}, st[1], [](size_t k) { return 0.05f * float(k % 4 + 1); }),
            A({2, 3}, st[2], [](size_t k) { return -0.3f * float(k % 3 + 1); }), B({1, 3, 4}, st[3], gen(4)),
            C({1, 3, 4}, st[4], gen(5)), D({2}, st[5], gen(6));
         auto u_d = ToDevice(u), delta_d = ToDevice(delta), A_d = ToDevice(A), B_d = ToDevice(B), C_d = ToDevice(C),
              D_d = ToDevice(D);
         SOFIE_StridedMamba::Session<alpaka::TagGpuCudaRt> s("", st);
         auto y = s.infer(u_d, delta_d, A_d, B_d, C_d, D_d);
         return toHost(y, 8);
      };
      auto ref = run({{8, 4, 1}, {8, 4, 1}, {3, 1}, {12, 4, 1}, {12, 4, 1}, {1}});
      Check2(run({{1, 2, 6}, {9, 4, 1}, {1, 2}, {1, 3, 9}, {20, 5, 1}, {3}}), ref);
   }
   // RWKV: r, k, v, w [1,2,3,2] and u [2,2]
   {
      auto run = [&](const std::vector<std::vector<size_t>> &st) {
         StridedData<> r({1, 2, 3, 2}, st[0], gen(1)), k({1, 2, 3, 2}, st[1], gen(2)), v({1, 2, 3, 2}, st[2], gen(3)),
            w({1, 2, 3, 2}, st[3], gen(4)), u({2, 2}, st[4], gen(5));
         auto r_d = ToDevice(r), k_d = ToDevice(k), v_d = ToDevice(v), w_d = ToDevice(w), u_d = ToDevice(u);
         SOFIE_StridedRWKV::Session<alpaka::TagGpuCudaRt> s("", st);
         auto y = s.infer(r_d, k_d, v_d, w_d, u_d);
         return toHost(y, 12);
      };
      auto ref = run({{12, 6, 2, 1}, {12, 6, 2, 1}, {12, 6, 2, 1}, {12, 6, 2, 1}, {2, 1}});
      Check2(run({{1, 2, 4, 12}, {24, 8, 2, 1}, {1, 2, 4, 12}, {50, 20, 3, 1}, {1, 2}}), ref);
   }
   // Griffin RG-LRU: x and the gate a (in (0,1)) [1,3,4]
   {
      auto run = [&](const std::vector<std::vector<size_t>> &st) {
         StridedData<> x({1, 3, 4}, st[0], gen(1)), a({1, 3, 4}, st[1], [](size_t k) { return 0.1f + 0.07f * float(k % 11); });
         auto x_d = ToDevice(x), a_d = ToDevice(a);
         SOFIE_StridedGriffin::Session<alpaka::TagGpuCudaRt> s("", st);
         auto y = s.infer(x_d, a_d);
         return toHost(y, 12);
      };
      auto ref = run({{12, 4, 1}, {12, 4, 1}});
      Check2(run({{1, 2, 6}, {40, 8, 1}}), ref);
   }
}

TEST_F(SofieAlpakaStridedTest, GemmBothOperandsStrided)
{
   const std::vector<std::vector<size_t>> layoutsA{{5, 1}, {8, 1}, {1, 3}}, layoutsB{{4, 1}, {6, 1}, {1, 5}};
   for (size_t i = 0; i < layoutsA.size(); i++) {
      StridedData<> a({3, 5}, layoutsA[i]);
      StridedData<> b({5, 4}, layoutsB[i], [](size_t k) { return 0.1f * float(k % 7) - 0.3f; });
      auto a_d = ToDevice(a);
      auto b_d = ToDevice(b);
      SOFIE_StridedGemmAB::Session<alpaka::TagGpuCudaRt> s("", {layoutsA[i], layoutsB[i]});
      std::vector<float> ref(12, 0.f);
      for (size_t r = 0; r < 3; r++)
         for (size_t c = 0; c < 4; c++)
            for (size_t k = 0; k < 5; k++)
               ref[r * 4 + c] += a.logical[r * 5 + k] * b.logical[k * 4 + c];
      auto y = s.infer(a_d, b_d);
      Check(y, ref);
   }
}

TEST_F(SofieAlpakaStridedTest, GemmTransposedOperandsBothStrided)
{
   // Y = A^T * B^T with A stored 5x3 and B stored 4x5
   const std::vector<std::vector<size_t>> layoutsA{{3, 1}, {8, 1}, {1, 5}}, layoutsB{{5, 1}, {7, 1}, {1, 4}};
   for (size_t i = 0; i < layoutsA.size(); i++) {
      StridedData<> a({5, 3}, layoutsA[i]);
      StridedData<> b({4, 5}, layoutsB[i], [](size_t k) { return 0.1f * float(k % 7) - 0.3f; });
      auto a_d = ToDevice(a);
      auto b_d = ToDevice(b);
      SOFIE_StridedGemmABt::Session<alpaka::TagGpuCudaRt> s("", {layoutsA[i], layoutsB[i]});
      std::vector<float> ref(12, 0.f);
      for (size_t r = 0; r < 3; r++)
         for (size_t c = 0; c < 4; c++)
            for (size_t k = 0; k < 5; k++)
               ref[r * 4 + c] += a.logical[k * 3 + r] * b.logical[c * 5 + k];
      auto y = s.infer(a_d, b_d);
      Check(y, ref);
   }
}

TEST_F(SofieAlpakaStridedTest, GemmOperandWithoutUnitStrideThrows)
{
   StridedData<> a({3, 5}, {5, 1});
   StridedData<> b({5, 4}, {8, 2});
   auto a_d = ToDevice(a);
   auto b_d = ToDevice(b);
   SOFIE_StridedGemmAB::Session<alpaka::TagGpuCudaRt> s("", {{5, 1}, {8, 2}});
   EXPECT_THROW(s.infer(a_d, b_d), std::runtime_error);
}

// ---------------------------------------------------------------------------------------------------------------
// If operator: the sub-graphs (branches) read the strided input of the main model
// ---------------------------------------------------------------------------------------------------------------

TEST_F(SofieAlpakaStridedTest, IfBranches)
{
   // 3x5 views: row padding, transposed and contiguous
   for (const auto &strides : std::vector<std::vector<size_t>>{{8, 1}, {1, 3}, {5, 1}}) {
      StridedData<> x({3, 5}, strides);
      auto x_d = ToDevice(x);
      // the condition has no strides (empty array)
      SOFIE_StridedIf::Session<alpaka::TagGpuCudaRt> session("", {std::vector<size_t>{}, strides});
      for (uint8_t cond : {1, 0, 1}) {
         StridedData<uint8_t> c({1}, {1}, [cond](size_t) { return cond; });
         auto c_d = ToDevice(c);
         std::vector<float> ref(15);
         for (size_t i = 0; i < 15; i++)
            ref[i] = cond ? x.logical[i] * x.logical[i] : 1.f / (1.f + std::exp(x.logical[i]));
         auto y = session.infer(c_d, x_d);
         Check(y, ref);
      }
   }
}
