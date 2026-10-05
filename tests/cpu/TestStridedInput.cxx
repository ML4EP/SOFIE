// Tests of the stride aware inference path (Options::kStridedInput): the Session constructor takes the strides
// (in elements) of the input tensors and the operators reading the graph inputs access them through the strides.
// StridedInputModelGenerator.cxx emits the headers this file #includes (see the "sofie-strided-emit" CTest fixture
// in CMakeLists.txt).

#include "StridedUnaryChain.hxx"
#include "StridedReluTwice.hxx"
#include "StridedTanhDyn.hxx"
#include "StridedLeakyRelu.hxx"
#include "StridedElu.hxx"
#include "StridedSelu.hxx"
#include "StridedClip.hxx"
#include "StridedIdentity.hxx"
#include "StridedNot.hxx"
#include "StridedBitwiseNot.hxx"
#include "StridedActivationsCpu.hxx"
#include "StridedCast.hxx"
#include "StridedIsNaN.hxx"
#include "StridedBinary.hxx"
#include "StridedCompare.hxx"
#include "StridedBitwiseAnd.hxx"
#include "StridedNarySum.hxx"
#include "StridedWhere.hxx"
#include "StridedTranspose.hxx"
#include "StridedSlice.hxx"
#include "StridedGather.hxx"
#include "StridedConcat.hxx"
#include "StridedSplit.hxx"
#include "StridedPad.hxx"
#include "StridedTile.hxx"
#include "StridedExpand.hxx"
#include "StridedSoftmaxMid.hxx"
#include "StridedSoftmaxLast.hxx"
#include "StridedLogSoftmax.hxx"
#include "StridedReduceSumMid.hxx"
#include "StridedReduceMeanLast.hxx"
#include "StridedReduceMaxFirst.hxx"
#include "StridedLayerNorm.hxx"
#include "StridedRMSNorm.hxx"
#include "StridedBatchNorm.hxx"
#include "StridedGroupNorm.hxx"
#include "StridedInstanceNorm.hxx"
#include "StridedL2Norm.hxx"
#include "StridedCumSum.hxx"
#include "StridedCumSumRev.hxx"
#include "StridedMaxPool.hxx"
#include "StridedAvgPool.hxx"
#include "StridedTopK.hxx"
#include "StridedReshape.hxx"
#include "StridedFlatten.hxx"
#include "StridedTrilu.hxx"
#include "StridedNonZero.hxx"
#include "StridedGatherND.hxx"
#include "StridedGatherNDElems.hxx"
#include "StridedScatterND.hxx"
#include "StridedScatterElements.hxx"
#include "StridedConv.hxx"
#include "StridedConvTranspose.hxx"
#include "StridedEinsum.hxx"
#include "StridedSDPA.hxx"
#include "StridedRNN.hxx"
#include "StridedLSTM.hxx"
#include "StridedGRU.hxx"
#include "StridedMamba.hxx"
#include "StridedRWKV.hxx"
#include "StridedGriffin.hxx"
#include "StridedGemmAB.hxx"
#include "StridedGemmABt.hxx"
#include "StridedGemmBiasRelu.hxx"
#include "StridedGemmTransAB.hxx"

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gemm.hxx"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

namespace {

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

template <typename T>
void ExpectNear(const std::vector<T> &res, const std::vector<T> &ref)
{
   ASSERT_EQ(res.size(), ref.size());
   for (size_t i = 0; i < ref.size(); i++)
      EXPECT_NEAR(double(res[i]), double(ref[i]), 1.e-5 * std::max(1.0, std::fabs(double(ref[i])))) << "i=" << i;
}

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

// an operator which does not read its input through strides (the default of ROperator::SupportsStridedInput)
class ROperator_NoStrides final : public SOFIE::ROperator {
   std::string fNX, fNY;

public:
   ROperator_NoStrides(const std::string &x, const std::string &y) : fNX(x), fNY(y)
   {
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }
   void Initialize(SOFIE::RModel &model) override
   {
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), model.GetDimTensorShape(fNX));
   }
   std::string Generate(std::string) override { return ""; }
};


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

// transposed convolution of a 1x2x3x3 tensor with the 2x3x2x2 weights and the bias of StridedInputModelGenerator.cxx
std::vector<float> RefConvTranspose(const std::vector<float> &x)
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

TEST(StridedInput, ContiguousDefault)
{
   // no strides passed: same result as the plain contiguous inference
   StridedBuffer in({3, 5}, {5, 1});
   SOFIE_StridedUnaryChain::Session s;
   std::vector<float> ref(15);
   for (size_t i = 0; i < 15; i++)
      ref[i] = 1.f / (1.f + std::exp(in.logical[i]));
   ExpectNear(s.infer(in.storage.data()), ref);
}

TEST(StridedInput, ContiguousExplicitStrides)
{
   StridedBuffer in({3, 5}, {5, 1});
   SOFIE_StridedUnaryChain::Session s("", {{5, 1}});
   std::vector<float> ref(15);
   for (size_t i = 0; i < 15; i++)
      ref[i] = 1.f / (1.f + std::exp(in.logical[i]));
   ExpectNear(s.infer(in.storage.data()), ref);
}

TEST(StridedInput, RowPadding)
{
   // 3x5 view of a row-major matrix with leading dimension 8 (e.g. a sub-matrix)
   StridedBuffer in({3, 5}, {8, 1});
   SOFIE_StridedUnaryChain::Session s("", {{8, 1}});
   std::vector<float> ref(15);
   for (size_t i = 0; i < 15; i++)
      ref[i] = 1.f / (1.f + std::exp(in.logical[i]));
   ExpectNear(s.infer(in.storage.data()), ref);
}

TEST(StridedInput, Transposed)
{
   // 3x5 view of a column-major matrix (the transposed layout): strides {1, 3}
   StridedBuffer in({3, 5}, {1, 3});
   SOFIE_StridedUnaryChain::Session s("", {{1, 3}});
   std::vector<float> ref(15);
   for (size_t i = 0; i < 15; i++)
      ref[i] = 1.f / (1.f + std::exp(in.logical[i]));
   ExpectNear(s.infer(in.storage.data()), ref);
}

TEST(StridedInput, SameOperatorKindOnStridedAndContiguousInput)
{
   // 3D permuted view: logical shape {2,3,4}, strides {1, 8, 2}
   StridedBuffer in({2, 3, 4}, {1, 8, 2});
   SOFIE_StridedReluTwice::Session s("", {{1, 8, 2}});
   std::vector<float> ref(24);
   for (size_t i = 0; i < 24; i++)
      ref[i] = std::max(in.logical[i], 0.f);
   ExpectNear(s.infer(in.storage.data()), ref);
}

TEST(StridedInput, DynamicShapeStridedSubBatch)
{
   // maximum batch 8, run a batch of 6 rows read from a matrix with leading dimension 7
   StridedBuffer in({6, 5}, {7, 1});
   SOFIE_StridedTanhDyn::Session s("", 8, {{7, 1}});
   std::vector<float> ref(30);
   for (size_t i = 0; i < 30; i++)
      ref[i] = std::tanh(in.logical[i]);
   ExpectNear(s.infer(6, in.storage.data()), ref);
}

TEST(StridedInput, DynamicShapeContiguousDefaultUsesCurrentShape)
{
   // with no strides given the contiguous ones follow the shape of each call, not the maximum one
   StridedBuffer in({3, 5}, {5, 1});
   SOFIE_StridedTanhDyn::Session s("", 8);
   std::vector<float> ref(15);
   for (size_t i = 0; i < 15; i++)
      ref[i] = std::tanh(in.logical[i]);
   ExpectNear(s.infer(3, in.storage.data()), ref);
}

TEST(StridedInput, InvalidStridesAreRejected)
{
   // wrong number of arrays
   EXPECT_THROW(SOFIE_StridedUnaryChain::Session("", {{5, 1}, {5, 1}}), std::runtime_error);
   // wrong rank
   EXPECT_THROW(SOFIE_StridedUnaryChain::Session("", {{5, 1, 1}}), std::runtime_error);
}

TEST(StridedInput, UnsupportedOperatorIsRejectedAtGeneration)
{
   // an operator which does not read its input through strides: asking for a strided input must fail loudly
   // instead of generating code reading the strided input as if it were contiguous
   SOFIE::RModel model("StridedUnsupported", "now");
   model.AddInputTensorInfo("X", SOFIE::ETensorType::FLOAT, std::vector<SOFIE::Dim>{SOFIE::Dim(4)});
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_NoStrides>("X", "Y"));
   model.AddOutputTensorNameList({"Y"});
   EXPECT_THROW(model.Generate(SOFIE::Options::kStridedInput, -1, 0, false), std::runtime_error);
}

TEST(StridedInput, RequiresSession)
{
   SOFIE::RModel model("StridedNoSession", "now");
   model.AddInputTensorInfo("X", SOFIE::ETensorType::FLOAT, std::vector<SOFIE::Dim>{SOFIE::Dim(4)});
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_NoStrides>("X", "Y"));
   model.AddOutputTensorNameList({"Y"});
   EXPECT_THROW(model.Generate(SOFIE::Options::kStridedInput | SOFIE::Options::kNoSession, -1, 0, false),
                std::runtime_error);
}

TEST(StridedInput, GemmContiguousDefault)
{
   StridedBuffer in({3, 5}, {5, 1});
   SOFIE_StridedGemmBiasRelu::Session s("StridedGemmBiasRelu.dat");
   ExpectNear(s.infer(in.storage.data()), RefGemmBiasRelu(in.logical));
}

TEST(StridedInput, GemmRowPadding)
{
   StridedBuffer in({3, 5}, {8, 1});
   SOFIE_StridedGemmBiasRelu::Session s("StridedGemmBiasRelu.dat", {{8, 1}});
   ExpectNear(s.infer(in.storage.data()), RefGemmBiasRelu(in.logical));
}

TEST(StridedInput, GemmColumnMajorInput)
{
   // 3x5 view of a column-major matrix: the unit stride is the first one
   StridedBuffer in({3, 5}, {1, 3});
   SOFIE_StridedGemmBiasRelu::Session s("StridedGemmBiasRelu.dat", {{1, 3}});
   ExpectNear(s.infer(in.storage.data()), RefGemmBiasRelu(in.logical));
}

TEST(StridedInput, GemmTransposedOperands)
{
   // stored 5x3 input with Gemm transA = 1 (row-major, padded and column-major layouts)
   for (const auto &strides : std::vector<std::vector<size_t>>{{3, 1}, {8, 1}, {1, 5}}) {
      StridedBuffer in({5, 3}, strides);
      SOFIE_StridedGemmTransAB::Session s("StridedGemmTransAB.dat", {strides});
      ExpectNear(s.infer(in.storage.data()), RefGemmTransAB(in.logical));
   }
}

TEST(StridedInput, GemmWithoutUnitStrideThrows)
{
   // a view with no unit stride cannot be described to BLAS: fail loudly instead of computing garbage
   StridedBuffer in({3, 5}, {20, 2});
   SOFIE_StridedGemmBiasRelu::Session s("StridedGemmBiasRelu.dat", {{20, 2}});
   EXPECT_THROW(s.infer(in.storage.data()), std::runtime_error);
}

TEST(StridedInput, GemmStridedBiasIsRejectedAtGeneration)
{
   // only the operands A and B of a Gemm can be strided
   SOFIE::RModel model("StridedGemmC", "now");
   model.AddInputTensorInfo("A", SOFIE::ETensorType::FLOAT, std::vector<SOFIE::Dim>{SOFIE::Dim(2), SOFIE::Dim(3)});
   model.AddInputTensorInfo("W", SOFIE::ETensorType::FLOAT, std::vector<SOFIE::Dim>{SOFIE::Dim(3), SOFIE::Dim(4)});
   model.AddInputTensorInfo("C", SOFIE::ETensorType::FLOAT, std::vector<SOFIE::Dim>{SOFIE::Dim(2), SOFIE::Dim(4)});
   model.AddInputTensorName("A");
   model.AddInputTensorName("W");
   model.AddInputTensorName("C");
   model.AddOperator(std::make_unique<SOFIE::ROperator_Gemm<float>>(1.0, 1.0, 0, 0, "A", "W", "C", "Y"));
   model.AddOutputTensorNameList({"Y"});
   EXPECT_THROW(model.Generate(SOFIE::Options::kStridedInput, -1, 0, false), std::runtime_error);
}

// ---------------------------------------------------------------------------------------------------------------
// elementwise operators: a permuted 2x3x4 view (strides {1, 8, 2}) and a padded one (strides {16, 5, 1})
// ---------------------------------------------------------------------------------------------------------------

#define STRIDED_UNARY_TEST(NAME, TYPE, GENERATOR, REFERENCE)                                                           \
   TEST(StridedInput, NAME)                                                                                           \
   {                                                                                                                  \
      for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}, {12, 4, 1}}) {              \
         StridedBuffer<TYPE> in({2, 3, 4}, strides, GENERATOR);                                                       \
         SOFIE_Strided##NAME::Session s("", {strides});                                                              \
         auto y = s.infer(in.storage.data());                                                                         \
         std::vector<TYPE> ref;                                                                                       \
         for (TYPE x : in.logical)                                                                                    \
            ref.push_back(REFERENCE);                                                                                 \
         ASSERT_EQ(y.size(), ref.size());                                                                             \
         for (size_t i = 0; i < ref.size(); i++)                                                                      \
            EXPECT_NEAR(double(y[i]), double(ref[i]), 1.e-5) << "i=" << i;                                           \
      }                                                                                                               \
   }

STRIDED_UNARY_TEST(LeakyRelu, float, nullptr, x >= 0 ? x : 0.1f * x)
STRIDED_UNARY_TEST(Elu, float, nullptr, x >= 0 ? x : 1.5f * (std::exp(x) - 1))
STRIDED_UNARY_TEST(Selu, float, nullptr, 1.05f * (std::max(0.f, x) + std::min(0.f, 1.6f * (std::exp(x) - 1))))
STRIDED_UNARY_TEST(Clip, float, nullptr, std::min(1.f, std::max(-0.5f, x)))
STRIDED_UNARY_TEST(Identity, float, nullptr, x) // Identity materializes the strided input
STRIDED_UNARY_TEST(Not, int32_t, [](size_t k) { return int32_t(k % 5) - 2; }, !x)
STRIDED_UNARY_TEST(BitwiseNot, int32_t, [](size_t k) { return int32_t(k % 5) - 2; }, ~x)

TEST(StridedInput, ActivationsWithoutGpuKernels)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}}) {
      StridedBuffer<> in({2, 3, 4}, strides);
      SOFIE_StridedActivationsCpu::Session s("", {strides});
      auto y = s.infer(in.storage.data());
      std::vector<float> erf, swish, gelu, hsig, hswish;
      for (float x : in.logical) {
         erf.push_back(std::erf(x));
         swish.push_back(x / (1 + std::exp(-x)));
         gelu.push_back(0.5f * x * (1.f + std::erf(x * 0.7071067811865475f)));
         hsig.push_back(std::max(0.f, std::min(1.f, 0.2f * x + 0.5f)));
         hswish.push_back(x * std::max(0.f, std::min(1.f, x / 6.f + 0.5f)));
      }
      ExpectNear(y[0], erf);
      ExpectNear(y[1], swish);
      ExpectNear(y[2], gelu);
      ExpectNear(y[3], hsig);
      ExpectNear(y[4], hswish);
   }
}

TEST(StridedInput, CastAndIsNaN)
{
   StridedBuffer<> in({2, 3, 4}, {1, 8, 2}, [](size_t k) { return 0.9f * float(k) - 7.3f; });
   SOFIE_StridedCast::Session cast("", {{1, 8, 2}});
   std::vector<int32_t> castRef;
   for (float x : in.logical)
      castRef.push_back(static_cast<int32_t>(x));
   ASSERT_EQ(cast.infer(in.storage.data()), castRef);

   StridedBuffer<> withNaN({2, 3, 4}, {16, 5, 1}, [](size_t k) { return k % 5 == 0 ? std::nanf("") : float(k); });
   SOFIE_StridedIsNaN::Session isnan("", {{16, 5, 1}});
   std::vector<uint8_t> nanRef;
   for (float x : withNaN.logical)
      nanRef.push_back(std::isnan(x));
   ASSERT_EQ(isnan.infer(withNaN.storage.data()), nanRef);
}

// ---------------------------------------------------------------------------------------------------------------
// operators with several (strided) inputs and broadcasting
// ---------------------------------------------------------------------------------------------------------------

TEST(StridedInput, BinaryWithBroadcast)
{
   StridedBuffer<> a({2, 3, 4}, {1, 8, 2});
   StridedBuffer<> b({3, 1}, {2, 7}, [](size_t k) { return 1.5f * float(k) + 0.25f; });
   SOFIE_StridedBinary::Session s("", {{1, 8, 2}, {2, 7}});
   std::vector<float> ref(24);
   for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 3; j++)
         for (size_t k = 0; k < 4; k++)
            ref[(i * 3 + j) * 4 + k] = a.logical[(i * 3 + j) * 4 + k] - b.logical[j];
   ExpectNear(s.infer(a.storage.data(), b.storage.data()), ref);
}

TEST(StridedInput, BinaryOnlyOneInputStrided)
{
   // an empty stride array means a contiguous input
   StridedBuffer<> a({2, 3, 4}, {12, 4, 1});
   StridedBuffer<> b({3, 1}, {2, 7}, [](size_t k) { return 1.5f * float(k) + 0.25f; });
   SOFIE_StridedBinary::Session s("", {{}, {2, 7}});
   std::vector<float> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = a.logical[k] - b.logical[(k / 4) % 3];
   ExpectNear(s.infer(a.storage.data(), b.storage.data()), ref);
}

TEST(StridedInput, Comparison)
{
   StridedBuffer<> a({2, 3, 4}, {1, 8, 2});
   StridedBuffer<> b({4}, {3}, [](size_t k) { return 0.4f * float(k); });
   SOFIE_StridedCompare::Session s("", {{1, 8, 2}, {3}});
   std::vector<uint8_t> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = a.logical[k] > b.logical[k % 4];
   ASSERT_EQ(s.infer(a.storage.data(), b.storage.data()), ref);
}

TEST(StridedInput, BitwiseAnd)
{
   StridedBuffer<int32_t> a({2, 3, 4}, {12, 4, 1}, [](size_t k) { return int32_t(3 * k + 1); });
   StridedBuffer<int32_t> b({2, 3, 4}, {1, 8, 2}, [](size_t k) { return int32_t(5 * k + 2); });
   SOFIE_StridedBitwiseAnd::Session s("", {{12, 4, 1}, {1, 8, 2}});
   std::vector<int32_t> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = a.logical[k] & b.logical[k];
   ASSERT_EQ(s.infer(a.storage.data(), b.storage.data()), ref);
}

TEST(StridedInput, NarySum)
{
   StridedBuffer<> a({2, 3, 4}, {1, 8, 2});
   StridedBuffer<> b({3, 4}, {1, 3}, [](size_t k) { return 0.1f * float(k); });
   StridedBuffer<> c({4}, {2}, [](size_t k) { return 10.f * float(k); });
   SOFIE_StridedNarySum::Session s("", {{1, 8, 2}, {1, 3}, {2}});
   std::vector<float> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = a.logical[k] + b.logical[k % 12] + c.logical[k % 4];
   ExpectNear(s.infer(a.storage.data(), b.storage.data(), c.storage.data()), ref);
}

TEST(StridedInput, Where)
{
   StridedBuffer<uint8_t> c({3, 1}, {2, 5}, [](size_t k) { return uint8_t(k % 2 == 0); });
   StridedBuffer<> x({2, 3, 4}, {1, 8, 2});
   StridedBuffer<> y({4}, {3}, [](size_t k) { return -3.f * float(k); });
   SOFIE_StridedWhere::Session s("", {{2, 5}, {1, 8, 2}, {3}});
   std::vector<float> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = c.logical[(k / 4) % 3] ? x.logical[k] : y.logical[k % 4];
   ExpectNear(s.infer(c.storage.data(), x.storage.data(), y.storage.data()), ref);
}

// ---------------------------------------------------------------------------------------------------------------
// operators moving the data of strided inputs
// ---------------------------------------------------------------------------------------------------------------

TEST(StridedInput, Transpose)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}, {12, 4, 1}}) {
      StridedBuffer<> in({2, 3, 4}, strides);
      SOFIE_StridedTranspose::Session s("", {strides});
      std::vector<float> ref(24);
      for (size_t l = 0; l < 4; l++)
         for (size_t i = 0; i < 2; i++)
            for (size_t j = 0; j < 3; j++)
               ref[(l * 2 + i) * 3 + j] = in.logical[(i * 3 + j) * 4 + l];
      ExpectNear(s.infer(in.storage.data()), ref);
   }
}

TEST(StridedInput, Slice)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{6, 1}, {9, 1}, {1, 4}, {2, 9}}) {
      StridedBuffer<> in({4, 6}, strides);
      SOFIE_StridedSlice::Session s("", {strides});
      std::vector<float> ref;
      for (size_t i = 0; i < 2; i++)
         for (size_t j = 0; j < 2; j++)
            ref.push_back(in.logical[(2 * i) * 6 + 1 + 2 * j]);
      ExpectNear(s.infer(in.storage.data()), ref);
   }
}

TEST(StridedInput, Gather)
{
   StridedBuffer<> x({4, 5}, {1, 4});
   StridedBuffer<int64_t> idx({3}, {2}, [](size_t k) { return int64_t((2 * k + 3) % 4); });
   SOFIE_StridedGather::Session s("", {{1, 4}, {2}});
   std::vector<float> ref;
   for (size_t i = 0; i < 3; i++)
      for (size_t c = 0; c < 5; c++)
         ref.push_back(x.logical[idx.logical[i] * 5 + c]);
   ExpectNear(s.infer(x.storage.data(), idx.storage.data()), ref);
}

TEST(StridedInput, ConcatMixedStridedAndContiguousInputs)
{
   StridedBuffer<> a({2, 3}, {1, 2});
   StridedBuffer<> b({2, 2}, {2, 1}, [](size_t k) { return 100.f + float(k); });
   StridedBuffer<> c({2, 1}, {5, 3}, [](size_t k) { return 200.f + float(k); });
   SOFIE_StridedConcat::Session s("", {{1, 2}, {}, {5, 3}});
   std::vector<float> ref;
   for (size_t r = 0; r < 2; r++) {
      for (size_t j = 0; j < 3; j++)
         ref.push_back(a.logical[r * 3 + j]);
      for (size_t j = 0; j < 2; j++)
         ref.push_back(b.logical[r * 2 + j]);
      ref.push_back(c.logical[r]);
   }
   ExpectNear(s.infer(a.storage.data(), b.storage.data(), c.storage.data()), ref);
}

TEST(StridedInput, Split)
{
   StridedBuffer<> in({2, 6}, {1, 2});
   SOFIE_StridedSplit::Session s("", {{1, 2}});
   auto y = s.infer(in.storage.data());
   std::vector<float> ref0, ref1;
   for (size_t r = 0; r < 2; r++)
      for (size_t j = 0; j < 3; j++) {
         ref0.push_back(in.logical[r * 6 + j]);
         ref1.push_back(in.logical[r * 6 + 3 + j]);
      }
   ExpectNear(y[0], ref0);
   ExpectNear(y[1], ref1);
}

TEST(StridedInput, Pad)
{
   StridedBuffer<> in({2, 3}, {1, 2});
   SOFIE_StridedPad::Session s("", {{1, 2}});
   std::vector<float> ref(20, 0.f);
   for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 3; j++)
         ref[(i + 1) * 5 + j] = in.logical[i * 3 + j];
   ExpectNear(s.infer(in.storage.data()), ref);
}

TEST(StridedInput, Tile)
{
   StridedBuffer<> in({2, 3}, {1, 2});
   SOFIE_StridedTile::Session s("", {{1, 2}});
   std::vector<float> ref(36);
   for (size_t i = 0; i < 4; i++)
      for (size_t j = 0; j < 9; j++)
         ref[i * 9 + j] = in.logical[(i % 2) * 3 + (j % 3)];
   ExpectNear(s.infer(in.storage.data()), ref);
}

TEST(StridedInput, Expand)
{
   StridedBuffer<> in({3, 1}, {2, 7});
   SOFIE_StridedExpand::Session s("", {{2, 7}});
   std::vector<float> ref(24);
   for (size_t k = 0; k < 24; k++)
      ref[k] = in.logical[(k / 4) % 3];
   ExpectNear(s.infer(in.storage.data()), ref);
}

#define STRIDED_X_TEST(NAME, WEIGHTS, SHAPE, REFERENCE)                                                                \
   TEST(StridedInput, NAME)                                                                                           \
   {                                                                                                                  \
      for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}, {12, 4, 1}}) {              \
         StridedBuffer<> in(SHAPE, strides);                                                                          \
         SOFIE_Strided##NAME::Session s(WEIGHTS, {strides});                                                         \
         ExpectNear(s.infer(in.storage.data()), REFERENCE(in.logical));                                              \
      }                                                                                                               \
   }

STRIDED_X_TEST(SoftmaxMid, "", (std::vector<size_t>{2, 3, 4}), [](const std::vector<float> &x) { return RefSoftmax(x, 1, false); })
STRIDED_X_TEST(SoftmaxLast, "", (std::vector<size_t>{2, 3, 4}), [](const std::vector<float> &x) { return RefSoftmax(x, 2, false); })
STRIDED_X_TEST(LogSoftmax, "", (std::vector<size_t>{2, 3, 4}), [](const std::vector<float> &x) { return RefSoftmax(x, 2, true); })
STRIDED_X_TEST(ReduceSumMid, "", (std::vector<size_t>{2, 3, 4}), RefReduceSumMid)
STRIDED_X_TEST(ReduceMeanLast, "", (std::vector<size_t>{2, 3, 4}), RefReduceMeanLast)
STRIDED_X_TEST(ReduceMaxFirst, "", (std::vector<size_t>{2, 3, 4}), RefReduceMaxFirst)
STRIDED_X_TEST(LayerNorm, "StridedLayerNorm.dat", (std::vector<size_t>{2, 3, 4}), RefLayerNorm)
STRIDED_X_TEST(RMSNorm, "StridedRMSNorm.dat", (std::vector<size_t>{2, 3, 4}), RefRMSNorm)
STRIDED_X_TEST(BatchNorm, "StridedBatchNorm.dat", (std::vector<size_t>{2, 3, 4}), RefBatchNorm)
STRIDED_X_TEST(InstanceNorm, "StridedInstanceNorm.dat", (std::vector<size_t>{2, 3, 4}), RefInstanceNorm)
STRIDED_X_TEST(L2Norm, "", (std::vector<size_t>{2, 3, 4}), RefL2Norm)
STRIDED_X_TEST(CumSum, "", (std::vector<size_t>{2, 3, 4}), RefCumSum)
STRIDED_X_TEST(CumSumRev, "", (std::vector<size_t>{2, 3, 4}), RefCumSumRev)

TEST(StridedInput, GroupNorm)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 3}, {40, 5, 1}, {12, 3, 1}}) {
      StridedBuffer<> in({2, 4, 3}, strides);
      SOFIE_StridedGroupNorm::Session s("StridedGroupNorm.dat", {strides});
      ExpectNear(s.infer(in.storage.data()), RefGroupNorm(in.logical));
   }
}

TEST(StridedInput, Pooling)
{
   // NCHW view of contiguous data, of padded data and of NHWC data
   for (const auto &strides : std::vector<std::vector<size_t>>{{32, 16, 4, 1}, {64, 24, 5, 1}, {32, 1, 8, 2}}) {
      StridedBuffer<> in({1, 2, 4, 4}, strides);
      SOFIE_StridedMaxPool::Session maxPool("", {strides});
      ExpectNear(maxPool.infer(in.storage.data()), RefPool(in.logical, true));
      SOFIE_StridedAvgPool::Session avgPool("", {strides});
      ExpectNear(avgPool.infer(in.storage.data()), RefPool(in.logical, false));
   }
}

TEST(StridedInput, TopK)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}, {12, 4, 1}}) {
      StridedBuffer<> in({2, 3, 4}, strides, [](size_t k) { return float((k * 13) % 24); });
      SOFIE_StridedTopK::Session s("", {strides});
      auto [values, indices] = s.infer(in.storage.data());
      // the two largest values along the axis of size 3, in decreasing order
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
      ExpectNear(values, refValues);
      EXPECT_EQ(indices, refIndices);
   }
}

// ---------------------------------------------------------------------------------------------------------------
// shape changes, triangular part, non zero elements, gather and scatter of strided inputs
// ---------------------------------------------------------------------------------------------------------------

TEST(StridedInput, ReshapeAndFlatten)
{
   // the output of a reshaped strided input cannot alias it: it is a contiguous copy
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}, {12, 4, 1}}) {
      StridedBuffer<> in({2, 3, 4}, strides);
      SOFIE_StridedReshape::Session reshape("", {strides});
      ExpectNear(reshape.infer(in.storage.data()), in.logical);
      SOFIE_StridedFlatten::Session flatten("", {strides});
      ExpectNear(flatten.infer(in.storage.data()), in.logical);
   }
}

TEST(StridedInput, Trilu)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}}) {
      StridedBuffer<> in({2, 3, 4}, strides);
      SOFIE_StridedTrilu::Session s("", {strides});
      std::vector<float> ref(24);
      for (size_t b = 0; b < 2; b++)
         for (size_t r = 0; r < 3; r++)
            for (size_t c = 0; c < 4; c++)
               ref[(b * 3 + r) * 4 + c] = c >= r ? in.logical[(b * 3 + r) * 4 + c] : 0.f;
      ExpectNear(s.infer(in.storage.data()), ref);
   }
}

TEST(StridedInput, NonZero)
{
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 8, 2}, {16, 5, 1}}) {
      StridedBuffer<> in({2, 3, 4}, strides, [](size_t k) { return k % 3 == 0 ? 0.f : float(k); });
      SOFIE_StridedNonZero::Session s("", {strides});
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
      EXPECT_EQ(s.infer(in.storage.data()), ref);
   }
}

TEST(StridedInput, GatherND)
{
   StridedBuffer<> x({4, 5}, {1, 4});
   StridedBuffer<int64_t> rows({3, 1}, {2, 5}, [](size_t k) { return int64_t((2 * k + 3) % 4); });
   SOFIE_StridedGatherND::Session s("", {{1, 4}, {2, 5}});
   std::vector<float> ref;
   for (size_t i = 0; i < 3; i++)
      for (size_t c = 0; c < 5; c++)
         ref.push_back(x.logical[rows.logical[i] * 5 + c]);
   ExpectNear(s.infer(x.storage.data(), rows.storage.data()), ref);

   StridedBuffer<int64_t> elems({3, 2}, {1, 3}, [](size_t k) { return int64_t(k % 4 == 0 ? 3 : (k % 5)); });
   SOFIE_StridedGatherNDElems::Session e("", {{1, 4}, {1, 3}});
   std::vector<float> refElems;
   for (size_t i = 0; i < 3; i++)
      refElems.push_back(x.logical[elems.logical[i * 2] * 5 + elems.logical[i * 2 + 1]]);
   ExpectNear(e.infer(x.storage.data(), elems.storage.data()), refElems);
}

TEST(StridedInput, ScatterND)
{
   StridedBuffer<> x({4, 5}, {1, 4});
   StridedBuffer<int64_t> idx({2, 1}, {3, 7}, [](size_t k) { return int64_t(k == 0 ? 2 : 0); });
   StridedBuffer<> upd({2, 5}, {1, 2}, [](size_t k) { return 100.f + float(k); });
   SOFIE_StridedScatterND::Session s("", {{1, 4}, {3, 7}, {1, 2}});
   std::vector<float> ref = x.logical;
   for (size_t n = 0; n < 2; n++)
      for (size_t c = 0; c < 5; c++)
         ref[idx.logical[n] * 5 + c] = upd.logical[n * 5 + c];
   ExpectNear(s.infer(x.storage.data(), idx.storage.data(), upd.storage.data()), ref);
}

TEST(StridedInput, ScatterElements)
{
   StridedBuffer<> x({3, 4}, {1, 3});
   StridedBuffer<int64_t> idx({2, 4}, {1, 2}, [](size_t k) { return int64_t(k < 4 ? k % 3 : (k + 1) % 3); });
   StridedBuffer<> upd({2, 4}, {8, 2}, [](size_t k) { return 100.f + float(k); });
   SOFIE_StridedScatterElements::Session s("", {{1, 3}, {1, 2}, {8, 2}});
   std::vector<float> ref = x.logical;
   for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 4; j++)
         ref[idx.logical[i * 4 + j] * 4 + j] = upd.logical[i * 4 + j];
   ExpectNear(s.infer(x.storage.data(), idx.storage.data(), upd.storage.data()), ref);
}

TEST(StridedInput, Convolution)
{
   // NCHW, NCHW with a padded channel dimension and NHWC layouts
   for (const auto &strides : std::vector<std::vector<size_t>>{{108, 36, 6, 1}, {200, 50, 6, 1}, {108, 1, 18, 3}}) {
      StridedBuffer<> in({2, 3, 6, 6}, strides);
      SOFIE_StridedConv::Session s("StridedConv.dat", {strides});
      ExpectNear(s.infer(in.storage.data()), RefConv(in.logical));
   }
}

TEST(StridedInput, ConvTranspose)
{
   // NCHW, NCHW with a padded channel dimension and NHWC layouts
   for (const auto &strides : std::vector<std::vector<size_t>>{{18, 9, 3, 1}, {50, 25, 3, 1}, {18, 1, 6, 2}}) {
      StridedBuffer<> in({1, 2, 3, 3}, strides);
      SOFIE_StridedConvTranspose::Session s("StridedConvTranspose.dat", {strides});
      ExpectNear(s.infer(in.storage.data()), RefConvTranspose(in.logical));
   }
}

TEST(StridedInput, ConvTransposeWithoutBlasLayoutThrows)
{
   // padded rows: the spatial dimensions are not contiguous, which BLAS cannot describe
   StridedBuffer<> in({1, 2, 3, 3}, {40, 20, 5, 1});
   SOFIE_StridedConvTranspose::Session s("StridedConvTranspose.dat", {{40, 20, 5, 1}});
   EXPECT_THROW(s.infer(in.storage.data()), std::runtime_error);
}

TEST(StridedInput, Einsum)
{
   StridedBuffer<> a({3, 4}, {1, 3});
   StridedBuffer<> b({4, 5}, {8, 1}, [](size_t k) { return 0.1f * float(k) - 1.f; });
   SOFIE_StridedEinsum::Session s("", {{1, 3}, {8, 1}});
   std::vector<float> ref(15, 0.f);
   for (size_t i = 0; i < 3; i++)
      for (size_t j = 0; j < 4; j++)
         for (size_t k = 0; k < 5; k++)
            ref[i * 5 + k] += a.logical[i * 4 + j] * b.logical[j * 5 + k];
   ExpectNear(s.infer(a.storage.data(), b.storage.data()), ref);
}

TEST(StridedInput, ScaledDotProductAttention)
{
   // [B, H, S, D] views: contiguous, padded and stored as [B, S, H, D]
   for (const auto &strides : std::vector<std::vector<size_t>>{{24, 12, 3, 1}, {60, 30, 6, 1}, {24, 3, 6, 1}}) {
      StridedBuffer<> q({1, 2, 4, 3}, strides, [](size_t k) { return 0.3f * float((k * 5) % 7) - 1.f; });
      StridedBuffer<> k({1, 2, 4, 3}, strides, [](size_t i) { return 0.2f * float((i * 3) % 11) - 1.f; });
      StridedBuffer<> v({1, 2, 4, 3}, strides, [](size_t i) { return 0.5f * float(i % 5) - 1.f; });
      SOFIE_StridedSDPA::Session s("", {strides, strides, strides});
      ExpectNear(s.infer(q.storage.data(), k.storage.data(), v.storage.data()), RefSDPA(q.logical, k.logical, v.logical));
   }
}

// recurrent operators: the result for a strided input must be the one of the same data stored contiguously
template <class SessionT, class... Args>
void CheckRecurrent(const std::string &weights)
{
   auto gen = [](size_t k) { return 0.4f * float((k * 7) % 5) - 0.8f; };
   StridedBuffer<> contiguous({3, 2, 2}, {4, 2, 1}, gen);
   SessionT reference(weights, {{4, 2, 1}});
   auto ref = reference.infer(contiguous.storage.data());
   for (const auto &strides : std::vector<std::vector<size_t>>{{1, 3, 6}, {10, 5, 1}}) {
      StridedBuffer<> in({3, 2, 2}, strides, gen);
      SessionT s(weights, {strides});
      ExpectNear(s.infer(in.storage.data()), ref);
   }
}

TEST(StridedInput, RecurrentOperators)
{
   CheckRecurrent<SOFIE_StridedRNN::Session>("StridedRNN.dat");
   CheckRecurrent<SOFIE_StridedLSTM::Session>("StridedLSTM.dat");
   CheckRecurrent<SOFIE_StridedGRU::Session>("StridedGRU.dat");
}

TEST(StridedInput, ScanOperators)
{
   // the result of a strided input is the one of the same data stored contiguously (a fresh Session each time since
   // the scans keep a state)
   auto gen = [](size_t salt) {
      return [salt](size_t k) { return 0.15f * float((k * 7 + salt * 3) % 9) - 0.5f; };
   };
   // Mamba: u [1,2,4], delta [1,2,4], A [2,3], B [1,3,4], C [1,3,4], D [2]
   {
      auto run = [&](const std::vector<std::vector<size_t>> &st) {
         StridedBuffer<> u({1, 2, 4}, st[0], gen(1)), delta({1, 2, 4}, st[1], [](size_t k) { return 0.05f * float(k % 4 + 1); }),
            A({2, 3}, st[2], [](size_t k) { return -0.3f * float(k % 3 + 1); }), B({1, 3, 4}, st[3], gen(4)),
            C({1, 3, 4}, st[4], gen(5)), D({2}, st[5], gen(6));
         SOFIE_StridedMamba::Session s("", st);
         return s.infer(u.storage.data(), delta.storage.data(), A.storage.data(), B.storage.data(), C.storage.data(),
                        D.storage.data());
      };
      auto ref = run({{8, 4, 1}, {8, 4, 1}, {3, 1}, {12, 4, 1}, {12, 4, 1}, {1}});
      ExpectNear(run({{1, 2, 6}, {9, 4, 1}, {1, 2}, {1, 3, 9}, {20, 5, 1}, {3}}), ref);
   }
   // RWKV: r, k, v, w [1,2,3,2] and u [2,2]
   {
      auto run = [&](const std::vector<std::vector<size_t>> &st) {
         StridedBuffer<> r({1, 2, 3, 2}, st[0], gen(1)), k({1, 2, 3, 2}, st[1], gen(2)), v({1, 2, 3, 2}, st[2], gen(3)),
            w({1, 2, 3, 2}, st[3], gen(4)), u({2, 2}, st[4], gen(5));
         SOFIE_StridedRWKV::Session s("", st);
         return s.infer(r.storage.data(), k.storage.data(), v.storage.data(), w.storage.data(), u.storage.data());
      };
      auto ref = run({{12, 6, 2, 1}, {12, 6, 2, 1}, {12, 6, 2, 1}, {12, 6, 2, 1}, {2, 1}});
      ExpectNear(run({{1, 2, 4, 12}, {24, 8, 2, 1}, {1, 2, 4, 12}, {50, 20, 3, 1}, {1, 2}}), ref);
   }
   // Griffin RG-LRU: x and the gate a (in (0,1)) [1,3,4]
   {
      auto run = [&](const std::vector<std::vector<size_t>> &st) {
         StridedBuffer<> x({1, 3, 4}, st[0], gen(1)), a({1, 3, 4}, st[1], [](size_t k) { return 0.1f + 0.07f * float(k % 11); });
         SOFIE_StridedGriffin::Session s("", st);
         return s.infer(x.storage.data(), a.storage.data());
      };
      auto ref = run({{12, 4, 1}, {12, 4, 1}});
      ExpectNear(run({{1, 2, 6}, {40, 8, 1}}), ref);
   }
}

TEST(StridedInput, GemmBothOperandsStrided)
{
   const std::vector<std::vector<size_t>> layoutsA{{5, 1}, {8, 1}, {1, 3}}, layoutsB{{4, 1}, {6, 1}, {1, 5}};
   for (size_t i = 0; i < layoutsA.size(); i++) {
      StridedBuffer<> a({3, 5}, layoutsA[i]);
      StridedBuffer<> b({5, 4}, layoutsB[i], [](size_t k) { return 0.1f * float(k % 7) - 0.3f; });
      SOFIE_StridedGemmAB::Session s("", {layoutsA[i], layoutsB[i]});
      std::vector<float> ref(12, 0.f);
      for (size_t r = 0; r < 3; r++)
         for (size_t c = 0; c < 4; c++)
            for (size_t k = 0; k < 5; k++)
               ref[r * 4 + c] += a.logical[r * 5 + k] * b.logical[k * 4 + c];
      ExpectNear(s.infer(a.storage.data(), b.storage.data()), ref);
   }
}

TEST(StridedInput, GemmTransposedOperandsBothStrided)
{
   // Y = A^T * B^T with A stored 5x3 and B stored 4x5
   const std::vector<std::vector<size_t>> layoutsA{{3, 1}, {8, 1}, {1, 5}}, layoutsB{{5, 1}, {7, 1}, {1, 4}};
   for (size_t i = 0; i < layoutsA.size(); i++) {
      StridedBuffer<> a({5, 3}, layoutsA[i]);
      StridedBuffer<> b({4, 5}, layoutsB[i], [](size_t k) { return 0.1f * float(k % 7) - 0.3f; });
      SOFIE_StridedGemmABt::Session s("", {layoutsA[i], layoutsB[i]});
      std::vector<float> ref(12, 0.f);
      for (size_t r = 0; r < 3; r++)
         for (size_t c = 0; c < 4; c++)
            for (size_t k = 0; k < 5; k++)
               ref[r * 4 + c] += a.logical[k * 3 + r] * b.logical[c * 5 + k];
      ExpectNear(s.infer(a.storage.data(), b.storage.data()), ref);
   }
}

TEST(StridedInput, GemmOperandWithoutUnitStrideThrows)
{
   StridedBuffer<> a({3, 5}, {5, 1});
   StridedBuffer<> b({5, 4}, {8, 2});
   SOFIE_StridedGemmAB::Session s("", {{5, 1}, {8, 2}});
   EXPECT_THROW(s.infer(a.storage.data(), b.storage.data()), std::runtime_error);
}
