// Standalone generator for the strided input tests (Options::kStridedInput).
//
// Builds a few small models whose first operator reads a graph input, and emits for each one the CPU Session
// and the GPU (Alpaka) Session accepting the strides of the input tensor:
//  - StridedUnaryChain: Neg -> Sigmoid on a 2D input (the first operator is stride aware, the second is not)
//  - StridedReluTwice : Relu -> Relu on a 3D input (same operator kind, with and without strided input)
//  - StridedTanhDyn   : Tanh on an input with a dynamic leading dimension
//  - StridedLeakyRelu, StridedElu, StridedSelu, StridedClip, StridedIdentity: one elementwise operator of a float input
//  - StridedNot, StridedBitwiseNot: one elementwise operator of an int32 input
//  - StridedActivationsCpu: Erf, Swish, Gelu, HardSigmoid and HardSwish of one input (CPU only, no GPU kernels)
//  - StridedCast, StridedIsNaN: Cast (float to int32) and IsNaN of a float input
//  - StridedBinary, StridedCompare, StridedBitwiseAnd, StridedNarySum, StridedWhere: operators with several inputs
//    (broadcasting, except for the bitwise And), each of them being a strided graph input
//  - StridedTranspose, StridedSlice, StridedGather, StridedConcat, StridedSplit, StridedPad, StridedTile,
//    StridedExpand: operators moving the data of strided inputs
//  - StridedSoftmaxMid, StridedSoftmaxLast, StridedLogSoftmax, StridedReduceSumMid, StridedReduceMeanLast,
//    StridedReduceMaxFirst, StridedLayerNorm, StridedRMSNorm, StridedBatchNorm, StridedGroupNorm, StridedL2Norm,
//    StridedCumSum, StridedCumSumRev: reductions and normalizations of the input, StridedInstanceNorm (CPU only)
//  - StridedMaxPool, StridedAvgPool, StridedTopK
//  - StridedReshape, StridedFlatten, StridedTrilu, StridedNonZero, StridedGatherND, StridedGatherNDElems,
//    StridedScatterND, StridedScatterElements
//  - StridedConv, StridedConvTranspose (CPU only), StridedEinsum (CPU only), StridedSDPA
//  - StridedRNN, StridedLSTM, StridedGRU (CPU only), StridedMamba, StridedRWKV, StridedGriffin
//  - StridedGemmBiasRelu: Y = Relu(X * W + b), X is 3x5 (with the Relu fused into the Gemm on the GPU)
//  - StridedGemmTransAB : Y = X^T * W^T, X is stored 5x3 and W is stored 4x5
// They are used by cpu/TestStridedInput.cxx and alpaka/TestAlpakaStridedInput.cxx

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_BasicUnary.hxx"
#include "SOFIE/ROperator_BasicBinary.hxx"
#include "SOFIE/ROperator_BasicIs.hxx"
#include "SOFIE/ROperator_BasicNary.hxx"
#include "SOFIE/ROperator_Cast.hxx"
#include "SOFIE/ROperator_Clip.hxx"
#include "SOFIE/ROperator_Comparison.hxx"
#include "SOFIE/ROperator_Elu.hxx"
#include "SOFIE/ROperator_Erf.hxx"
#include "SOFIE/ROperator_Gelu.hxx"
#include "SOFIE/ROperator_BatchNormalization.hxx"
#include "SOFIE/ROperator_Conv.hxx"
#include "SOFIE/ROperator_ConvTranspose.hxx"
#include "SOFIE/ROperator_Einsum.hxx"
#include "SOFIE/ROperator_GRU.hxx"
#include "SOFIE/ROperator_GriffinRGLRU.hxx"
#include "SOFIE/ROperator_LSTM.hxx"
#include "SOFIE/ROperator_MambaScan.hxx"
#include "SOFIE/ROperator_RNN.hxx"
#include "SOFIE/ROperator_RWKV_WKV6.hxx"
#include "SOFIE/ROperator_SDPA.hxx"
#include "SOFIE/ROperator_Concat.hxx"
#include "SOFIE/ROperator_CumSum.hxx"
#include "SOFIE/ROperator_GroupNorm.hxx"
#include "SOFIE/ROperator_InstanceNormalization.hxx"
#include "SOFIE/ROperator_L2Normalization.hxx"
#include "SOFIE/ROperator_LayerNormalization.hxx"
#include "SOFIE/ROperator_RMSNorm.hxx"
#include "SOFIE/ROperator_GatherND.hxx"
#include "SOFIE/ROperator_NonZero.hxx"
#include "SOFIE/ROperator_Pool.hxx"
#include "SOFIE/ROperator_Reshape.hxx"
#include "SOFIE/ROperator_ScatterElements.hxx"
#include "SOFIE/ROperator_ScatterND.hxx"
#include "SOFIE/ROperator_Trilu.hxx"
#include "SOFIE/ROperator_Reduce.hxx"
#include "SOFIE/ROperator_Softmax.hxx"
#include "SOFIE/ROperator_Expand.hxx"
#include "SOFIE/ROperator_Gather.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_Pad.hxx"
#include "SOFIE/ROperator_Slice.hxx"
#include "SOFIE/ROperator_Split.hxx"
#include "SOFIE/ROperator_Tile.hxx"
#include "SOFIE/ROperator_TopK.hxx"
#include "SOFIE/ROperator_Transpose.hxx"
#include "SOFIE/ROperator_HardSigmoid.hxx"
#include "SOFIE/ROperator_HardSwish.hxx"
#include "SOFIE/ROperator_Identity.hxx"
#include "SOFIE/ROperator_LeakyRelu.hxx"
#include "SOFIE/ROperator_Logic.hxx"
#include "SOFIE/ROperator_Not.hxx"
#include "SOFIE/ROperator_Selu.hxx"
#include "SOFIE/ROperator_Swish.hxx"
#include "SOFIE/ROperator_Where.hxx"
#include "SOFIE/ROperator_Relu.hxx"
#include "SOFIE/ROperator_Sigmoid.hxx"
#include "SOFIE/ROperator_Tanh.hxx"

#include <iostream>

using namespace SOFIE;

namespace {

RModel BuildUnaryChain()
{
   RModel model("StridedUnaryChain", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_BasicUnary<float, EBasicUnaryOperator::kNeg>>("X", "H"));
   model.AddOperator(std::make_unique<ROperator_Sigmoid<float>>("H", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildReluTwice()
{
   RModel model("StridedReluTwice", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(2), Dim(3), Dim(4)});
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_Relu<float>>("X", "H"));
   model.AddOperator(std::make_unique<ROperator_Relu<float>>("H", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildTanhDynamic()
{
   RModel model("StridedTanhDyn", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim{"N", 8}, Dim(5)});
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_Tanh<float>>("X", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

// deterministic weights shared with the tests (cpu/TestStridedInput.cxx and alpaka/TestAlpakaStridedInput.cxx)
std::vector<float> GemmValues(size_t n, float scale)
{
   std::vector<float> v(n);
   for (size_t i = 0; i < n; i++)
      v[i] = scale * float((i * 7) % 11) - 0.5f;
   return v;
}

RModel BuildGemmBiasRelu()
{
   RModel model("StridedGemmBiasRelu", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("X");
   auto W = GemmValues(5 * 4, 0.1f);
   auto B = GemmValues(4, 0.3f);
   model.AddInitializedTensor("W", ETensorType::FLOAT, std::vector<std::size_t>{5, 4}, W.data());
   model.AddInitializedTensor("B", ETensorType::FLOAT, std::vector<std::size_t>{4}, B.data());
   model.AddOperator(std::make_unique<ROperator_Gemm<float>>(1.0, 1.0, 0, 0, "X", "W", "B", "H"));
   model.AddOperator(std::make_unique<ROperator_Relu<float>>("H", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildGemmTransAB()
{
   RModel model("StridedGemmTransAB", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(5), Dim(3)});
   model.AddInputTensorName("X");
   auto W = GemmValues(4 * 5, 0.1f);
   model.AddInitializedTensor("W", ETensorType::FLOAT, std::vector<std::size_t>{4, 5}, W.data());
   model.AddOperator(std::make_unique<ROperator_Gemm<float>>(1.0, 0.0, 1, 1, "X", "W", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

const std::vector<Dim> kShape3D{Dim(2), Dim(3), Dim(4)};

// a model made of one operator applied to the input X (a float or int32 tensor of shape 2x3x4)
template <class Op, class... Args>
RModel BuildSingleOp(const std::string &name, ETensorType type, Args... args)
{
   RModel model(name, "now");
   model.AddInputTensorInfo("X", type, kShape3D);
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<Op>(args..., "X", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildActivationsCpu()
{
   RModel model("StridedActivationsCpu", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_Erf<float>>("X", "Y0"));
   model.AddOperator(std::make_unique<ROperator_Swish>("X", "Y1"));
   model.AddOperator(std::make_unique<ROperator_Gelu>("X", "Y2", "none"));
   model.AddOperator(std::make_unique<ROperator_HardSigmoid<float>>(0.2f, 0.5f, "X", "Y3"));
   model.AddOperator(std::make_unique<ROperator_HardSwish<float>>("X", "Y4"));
   model.AddOutputTensorNameList({"Y0", "Y1", "Y2", "Y3", "Y4"});
   return model;
}

RModel BuildCast()
{
   RModel model("StridedCast", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_Cast>(ETensorType::INT32, "X", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildIsNaN()
{
   RModel model("StridedIsNaN", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_BasicIs<EBasicIsOperator::kIsNaN>>("X", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildBinary()
{
   // B has the shape 3x1: it is broadcast over the first and last dimensions of A
   RModel model("StridedBinary", "now");
   model.AddInputTensorInfo("A", ETensorType::FLOAT, kShape3D);
   model.AddInputTensorInfo("B", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(1)});
   model.AddInputTensorName("A");
   model.AddInputTensorName("B");
   model.AddOperator(std::make_unique<ROperator_BasicBinary<float, EBasicBinaryOperator::Sub>>("A", "B", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildCompare()
{
   RModel model("StridedCompare", "now");
   model.AddInputTensorInfo("A", ETensorType::FLOAT, kShape3D);
   model.AddInputTensorInfo("B", ETensorType::FLOAT, std::vector<Dim>{Dim(4)});
   model.AddInputTensorName("A");
   model.AddInputTensorName("B");
   model.AddOperator(std::make_unique<ROperator_Comparison<float, EComparisonOperator::Greater>>("A", "B", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildBitwiseAnd()
{
   RModel model("StridedBitwiseAnd", "now");
   model.AddInputTensorInfo("A", ETensorType::INT32, kShape3D);
   model.AddInputTensorInfo("B", ETensorType::INT32, kShape3D);
   model.AddInputTensorName("A");
   model.AddInputTensorName("B");
   model.AddOperator(std::make_unique<ROperator_LogicBinary<int32_t, ELogicBinaryOp::BitwiseAnd>>("A", "B", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildNarySum()
{
   RModel model("StridedNarySum", "now");
   model.AddInputTensorInfo("A", ETensorType::FLOAT, kShape3D);
   model.AddInputTensorInfo("B", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(4)});
   model.AddInputTensorInfo("C", ETensorType::FLOAT, std::vector<Dim>{Dim(4)});
   model.AddInputTensorName("A");
   model.AddInputTensorName("B");
   model.AddInputTensorName("C");
   model.AddOperator(std::make_unique<ROperator_BasicNary<float, EBasicNaryOperator::Sum>>(
      std::vector<std::string>{"A", "B", "C"}, "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildWhere()
{
   RModel model("StridedWhere", "now");
   model.AddInputTensorInfo("C", ETensorType::BOOL, std::vector<Dim>{Dim(3), Dim(1)});
   model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
   model.AddInputTensorInfo("Y", ETensorType::FLOAT, std::vector<Dim>{Dim(4)});
   model.AddInputTensorName("C");
   model.AddInputTensorName("X");
   model.AddInputTensorName("Y");
   model.AddOperator(std::make_unique<ROperator_Where<float>>("C", "X", "Y", "Z"));
   model.AddOutputTensorNameList({"Z"});
   return model;
}

RModel BuildTranspose()
{
   RModel model("StridedTranspose", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_Transpose>(std::vector<int64_t>{2, 0, 1}, "X", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildSlice()
{
   RModel model("StridedSlice", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(4), Dim(6)});
   model.AddInputTensorName("X");
   // X[0:4:2, 1:5:2] (the extents are multiples of the steps)
   std::vector<int64_t> starts{0, 1}, ends{4, 5}, axes{0, 1}, steps{2, 2};
   model.AddInitializedTensor("starts", ETensorType::INT64, std::vector<std::size_t>{2}, starts.data());
   model.AddInitializedTensor("ends", ETensorType::INT64, std::vector<std::size_t>{2}, ends.data());
   model.AddInitializedTensor("axes", ETensorType::INT64, std::vector<std::size_t>{2}, axes.data());
   model.AddInitializedTensor("steps", ETensorType::INT64, std::vector<std::size_t>{2}, steps.data());
   model.AddOperator(std::make_unique<ROperator_Slice<int64_t>>(
      "X", std::vector<std::string>{"starts", "ends", "axes", "steps"}, "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildGather()
{
   RModel model("StridedGather", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(4), Dim(5)});
   model.AddInputTensorInfo("I", ETensorType::INT64, std::vector<Dim>{Dim(3)});
   model.AddInputTensorName("X");
   model.AddInputTensorName("I");
   model.AddOperator(std::make_unique<ROperator_Gather>(0, "X", "I", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildConcat()
{
   RModel model("StridedConcat", "now");
   model.AddInputTensorInfo("A", ETensorType::FLOAT, std::vector<Dim>{Dim(2), Dim(3)});
   model.AddInputTensorInfo("B", ETensorType::FLOAT, std::vector<Dim>{Dim(2), Dim(2)});
   model.AddInputTensorInfo("C", ETensorType::FLOAT, std::vector<Dim>{Dim(2), Dim(1)});
   model.AddInputTensorName("A");
   model.AddInputTensorName("B");
   model.AddInputTensorName("C");
   model.AddOperator(std::make_unique<ROperator_Concat>(std::vector<std::string>{"A", "B", "C"}, 1, 0, "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildSplit()
{
   RModel model("StridedSplit", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(2), Dim(6)});
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_Split>("X", "", 1, std::vector<std::string>{"Y0", "Y1"}));
   model.AddOutputTensorNameList({"Y0", "Y1"});
   return model;
}

RModel BuildPad()
{
   RModel model("StridedPad", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(2), Dim(3)});
   model.AddInputTensorName("X");
   std::vector<int64_t> pads{1, 0, 1, 2};
   model.AddInitializedTensor("P", ETensorType::INT64, std::vector<std::size_t>{4}, pads.data());
   model.AddOperator(std::make_unique<ROperator_Pad<float>>("X", "P", "", "", "Y", "constant"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildTile()
{
   RModel model("StridedTile", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(2), Dim(3)});
   model.AddInputTensorName("X");
   std::vector<int64_t> repeats{2, 3};
   model.AddInitializedTensor("R", ETensorType::INT64, std::vector<std::size_t>{2}, repeats.data());
   model.AddOperator(std::make_unique<ROperator_Tile<float>>("R", "X", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildExpand()
{
   RModel model("StridedExpand", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(1)});
   model.AddInputTensorName("X");
   std::vector<int64_t> shape{2, 3, 4};
   model.AddInitializedTensor("S", ETensorType::INT64, std::vector<std::size_t>{3}, shape.data());
   model.AddOperator(std::make_unique<ROperator_Expand<float>>("X", "S", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

// deterministic values a + b * i shared with the tests (cpu/TestStridedInput.cxx and alpaka/TestAlpakaStridedInput.cxx)
std::vector<float> AffineValues(size_t n, float a, float b)
{
   std::vector<float> v(n);
   for (size_t i = 0; i < n; i++)
      v[i] = a + b * float(i);
   return v;
}

// a model made of a graph input X (float, shape `shape`) and the initialized tensors `weights` (name, values)
// used by the operator `op`
RModel BuildWithWeights(const std::string &name, const std::vector<Dim> &shape,
                        const std::vector<std::pair<std::string, std::vector<float>>> &weights,
                        std::unique_ptr<ROperator> op, const std::string &output = "Y")
{
   RModel model(name, "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, shape);
   model.AddInputTensorName("X");
   for (const auto &w : weights)
      model.AddInitializedTensor(w.first, ETensorType::FLOAT, std::vector<std::size_t>{w.second.size()},
                                 const_cast<float *>(w.second.data()));
   model.AddOperator(std::move(op));
   model.AddOutputTensorNameList({output});
   return model;
}

// an initialized float tensor
struct Weight {
   std::string name;
   std::vector<std::size_t> shape;
   std::vector<float> values;
};

// a model with several float graph inputs (name, shape) and initialized tensors, made of one operator
RModel BuildMulti(const std::string &name, const std::vector<std::pair<std::string, std::vector<size_t>>> &inputs,
                  const std::vector<Weight> &weights, std::unique_ptr<ROperator> op, const std::string &output = "Y")
{
   RModel model(name, "now");
   for (const auto &in : inputs) {
      model.AddInputTensorInfo(in.first, ETensorType::FLOAT, ConvertShapeToDim(in.second));
      model.AddInputTensorName(in.first);
   }
   for (const auto &w : weights)
      model.AddInitializedTensor(w.name, ETensorType::FLOAT, w.shape, const_cast<float *>(w.values.data()));
   model.AddOperator(std::move(op));
   model.AddOutputTensorNameList({output});
   return model;
}

template <class F>
void Emit(F build, const std::string &name)
{
   {
      RModel m = build();
      m.Generate(Options::kStridedInput, -1, 0, false);
      m.OutputGenerated(name + ".hxx");
   }
   {
      RModel m = build();
      m.GenerateGPU_ALPAKA(Options::kStridedInput, -1, false);
      m.OutputGenerated(name + "_GPU_ALPAKA.hxx");
   }
}

} // namespace

// models without a GPU implementation of their operators
template <class F>
void EmitCpu(F build, const std::string &name)
{
   RModel m = build();
   m.Generate(Options::kStridedInput, -1, 0, false);
   m.OutputGenerated(name + ".hxx");
}

int main()
{
   Emit(BuildUnaryChain, "StridedUnaryChain");
   Emit(BuildReluTwice, "StridedReluTwice");
   Emit(BuildTanhDynamic, "StridedTanhDyn");
   Emit([] { return BuildSingleOp<ROperator_LeakyRelu<float>>("StridedLeakyRelu", ETensorType::FLOAT, 0.1f); },
        "StridedLeakyRelu");
   Emit([] { return BuildSingleOp<ROperator_Elu<float>>("StridedElu", ETensorType::FLOAT, 1.5f); }, "StridedElu");
   Emit([] { return BuildSingleOp<ROperator_Selu<float>>("StridedSelu", ETensorType::FLOAT, 1.6f, 1.05f); },
        "StridedSelu");
   Emit(
      [] {
         RModel model("StridedClip", "now");
         model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
         model.AddInputTensorName("X");
         model.AddOperator(std::make_unique<ROperator_Clip<float>>("X", "Y", -0.5f, 1.0f));
         model.AddOutputTensorNameList({"Y"});
         return model;
      },
      "StridedClip");
   Emit([] { return BuildSingleOp<ROperator_Identity<float>>("StridedIdentity", ETensorType::FLOAT); },
        "StridedIdentity");
   Emit([] { return BuildSingleOp<ROperator_Not>("StridedNot", ETensorType::INT32); }, "StridedNot");
   Emit([] { return BuildSingleOp<ROperator_BitwiseNot<int32_t>>("StridedBitwiseNot", ETensorType::INT32); },
        "StridedBitwiseNot");
   EmitCpu(BuildActivationsCpu, "StridedActivationsCpu");
   Emit(BuildCast, "StridedCast");
   Emit(BuildIsNaN, "StridedIsNaN");
   Emit(BuildBinary, "StridedBinary");
   Emit(BuildCompare, "StridedCompare");
   Emit(BuildBitwiseAnd, "StridedBitwiseAnd");
   Emit(BuildNarySum, "StridedNarySum");
   Emit(BuildWhere, "StridedWhere");
   Emit(BuildTranspose, "StridedTranspose");
   Emit(BuildSlice, "StridedSlice");
   Emit(BuildGather, "StridedGather");
   Emit(BuildConcat, "StridedConcat");
   Emit(BuildSplit, "StridedSplit");
   Emit(BuildPad, "StridedPad");
   Emit(BuildTile, "StridedTile");
   Emit(BuildExpand, "StridedExpand");
   Emit([] { return BuildWithWeights("StridedSoftmaxMid", kShape3D, {}, std::make_unique<ROperator_Softmax>(1, "X", "Y")); },
        "StridedSoftmaxMid");
   Emit([] { return BuildWithWeights("StridedSoftmaxLast", kShape3D, {}, std::make_unique<ROperator_Softmax>(-1, "X", "Y")); },
        "StridedSoftmaxLast");
   Emit([] { return BuildWithWeights("StridedLogSoftmax", kShape3D, {}, std::make_unique<ROperator_Softmax>(2, "X", "Y", true)); },
        "StridedLogSoftmax");
   Emit([] {
      return BuildWithWeights("StridedReduceSumMid", kShape3D, {},
                              std::make_unique<ROperator_Reduce<ReduceSum>>(0, std::vector<int64_t>{1}, "X", "", "Y"));
   }, "StridedReduceSumMid");
   Emit([] {
      return BuildWithWeights("StridedReduceMeanLast", kShape3D, {},
                              std::make_unique<ROperator_Reduce<ReduceMean>>(0, std::vector<int64_t>{2}, "X", "", "Y"));
   }, "StridedReduceMeanLast");
   Emit([] {
      return BuildWithWeights("StridedReduceMaxFirst", kShape3D, {},
                              std::make_unique<ROperator_Reduce<ReduceMax>>(0, std::vector<int64_t>{0}, "X", "", "Y"));
   }, "StridedReduceMaxFirst");
   Emit([] {
      // the scale and bias of the normalized dimensions {3, 4} are initialized tensors of that shape
      RModel model("StridedLayerNorm", "now");
      model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
      model.AddInputTensorName("X");
      auto scale = AffineValues(12, 0.5f, 0.1f), bias = AffineValues(12, -1.f, 0.05f);
      model.AddInitializedTensor("S", ETensorType::FLOAT, std::vector<std::size_t>{3, 4}, scale.data());
      model.AddInitializedTensor("B", ETensorType::FLOAT, std::vector<std::size_t>{3, 4}, bias.data());
      model.AddOperator(std::make_unique<ROperator_LayerNormalization<float>>(1, 1.e-5f, 1, "X", "S", "B", "Y", "", ""));
      model.AddOutputTensorNameList({"Y"});
      return model;
   }, "StridedLayerNorm");
   Emit([] {
      RModel model("StridedRMSNorm", "now");
      model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
      model.AddInputTensorName("X");
      auto scale = AffineValues(12, 0.5f, 0.1f);
      model.AddInitializedTensor("S", ETensorType::FLOAT, std::vector<std::size_t>{3, 4}, scale.data());
      model.AddOperator(std::make_unique<ROperator_RMSNorm<float>>(1, 1.e-5f, "X", "S", "Y"));
      model.AddOutputTensorNameList({"Y"});
      return model;
   }, "StridedRMSNorm");
   Emit([] {
      auto scale = AffineValues(3, 0.5f, 0.25f), bias = AffineValues(3, -0.5f, 0.1f), mean = AffineValues(3, 0.1f, 0.2f),
           var = AffineValues(3, 0.5f, 0.3f);
      return BuildWithWeights("StridedBatchNorm", kShape3D,
                              {{"S", scale}, {"B", bias}, {"M", mean}, {"V", var}},
                              std::make_unique<ROperator_BatchNormalization<float>>(1.e-5f, 0.9f, 0, "X", "S", "B", "M", "V",
                                                                                    "Y"));
   }, "StridedBatchNorm");
   Emit([] {
      auto scale = AffineValues(4, 0.5f, 0.25f), bias = AffineValues(4, -0.5f, 0.1f);
      return BuildWithWeights("StridedGroupNorm", std::vector<Dim>{Dim(2), Dim(4), Dim(3)},
                              {{"S", scale}, {"B", bias}},
                              std::make_unique<ROperator_GroupNorm<float>>(2, 1.e-5f, "X", "S", "B", "Y"));
   }, "StridedGroupNorm");
   Emit([] {
      return BuildWithWeights("StridedL2Norm", kShape3D, {},
                              std::make_unique<ROperator_L2Normalization<float>>("X", 1.e-6f, "Y"));
   }, "StridedL2Norm");
   Emit([] {
      return BuildWithWeights("StridedCumSum", kShape3D, {}, std::make_unique<ROperator_CumSum<float>>(1, 0, 0, "X", "Y"));
   }, "StridedCumSum");
   Emit([] {
      return BuildWithWeights("StridedCumSumRev", kShape3D, {}, std::make_unique<ROperator_CumSum<float>>(2, 1, 1, "X", "Y"));
   }, "StridedCumSumRev");
   EmitCpu([] {
      auto scale = AffineValues(3, 0.5f, 0.25f), bias = AffineValues(3, -0.5f, 0.1f);
      return BuildWithWeights("StridedInstanceNorm", kShape3D, {{"S", scale}, {"B", bias}},
                              std::make_unique<ROperator_InstanceNormalization<float>>(1.e-5f, "X", "S", "B", "Y"));
   }, "StridedInstanceNorm");
   for (bool max : {true, false}) {
      Emit([max] {
         RAttributes_Pool attr;
         attr.kernel_shape = {2, 2};
         attr.strides = {2, 2};
         attr.pads = {0, 0, 0, 0};
         return BuildWithWeights(max ? "StridedMaxPool" : "StridedAvgPool",
                                 std::vector<Dim>{Dim(1), Dim(2), Dim(4), Dim(4)}, {},
                                 std::make_unique<ROperator_Pool<float>>(max ? MaxPool : AveragePool, attr, "X", "Y"));
      }, max ? "StridedMaxPool" : "StridedAvgPool");
   }
   Emit([] {
      RModel model("StridedTopK", "now");
      model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
      model.AddInputTensorName("X");
      std::vector<int64_t> k{2};
      model.AddInitializedTensor("K", ETensorType::INT64, std::vector<std::size_t>{1}, k.data());
      model.AddOperator(std::make_unique<ROperator_TopK<float>>(1, 1, 1, "K", "X", "V", "I"));
      model.AddOutputTensorNameList({"V", "I"});
      return model;
   }, "StridedTopK");
   Emit([] {
      std::vector<int64_t> shape{6, 4};
      RModel model("StridedReshape", "now");
      model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
      model.AddInputTensorName("X");
      model.AddInitializedTensor("S", ETensorType::INT64, std::vector<std::size_t>{2}, shape.data());
      model.AddOperator(std::make_unique<ROperator_Reshape>(Reshape, 0, "X", "S", "Y"));
      model.AddOutputTensorNameList({"Y"});
      return model;
   }, "StridedReshape");
   Emit([] {
      RModel model("StridedFlatten", "now");
      model.AddInputTensorInfo("X", ETensorType::FLOAT, kShape3D);
      model.AddInputTensorName("X");
      model.AddOperator(std::make_unique<ROperator_Reshape>(Flatten, 1, "X", "", "Y"));
      model.AddOutputTensorNameList({"Y"});
      return model;
   }, "StridedFlatten");
   Emit([] { return BuildSingleOp<ROperator_Trilu<float>>("StridedTrilu", ETensorType::FLOAT, 1); }, "StridedTrilu");
   Emit([] { return BuildWithWeights("StridedNonZero", kShape3D, {}, std::make_unique<ROperator_NonZero<float>>("X", "Y")); },
        "StridedNonZero");
   for (size_t tuple : {1, 2}) {
      Emit([tuple] {
         RModel model(tuple == 1 ? "StridedGatherND" : "StridedGatherNDElems", "now");
         model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(4), Dim(5)});
         model.AddInputTensorInfo("I", ETensorType::INT64, std::vector<Dim>{Dim(3), Dim(tuple)});
         model.AddInputTensorName("X");
         model.AddInputTensorName("I");
         model.AddOperator(std::make_unique<ROperator_GatherND>(0, "X", "I", "Y"));
         model.AddOutputTensorNameList({"Y"});
         return model;
      }, tuple == 1 ? "StridedGatherND" : "StridedGatherNDElems");
   }
   Emit([] {
      RModel model("StridedScatterND", "now");
      model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(4), Dim(5)});
      model.AddInputTensorInfo("I", ETensorType::INT64, std::vector<Dim>{Dim(2), Dim(1)});
      model.AddInputTensorInfo("U", ETensorType::FLOAT, std::vector<Dim>{Dim(2), Dim(5)});
      model.AddInputTensorName("X");
      model.AddInputTensorName("I");
      model.AddInputTensorName("U");
      model.AddOperator(std::make_unique<ROperator_ScatterND>("X", "I", "U", "Y", "none"));
      model.AddOutputTensorNameList({"Y"});
      return model;
   }, "StridedScatterND");
   Emit([] {
      RModel model("StridedScatterElements", "now");
      model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(4)});
      model.AddInputTensorInfo("I", ETensorType::INT64, std::vector<Dim>{Dim(2), Dim(4)});
      model.AddInputTensorInfo("U", ETensorType::FLOAT, std::vector<Dim>{Dim(2), Dim(4)});
      model.AddInputTensorName("X");
      model.AddInputTensorName("I");
      model.AddInputTensorName("U");
      model.AddOperator(std::make_unique<ROperator_ScatterElements>("X", "I", "U", "Y", 0, "none"));
      model.AddOutputTensorNameList({"Y"});
      return model;
   }, "StridedScatterElements");
   Emit([] {
      return BuildMulti("StridedConv", {{"X", {2, 3, 6, 6}}},
                        {{"W", {4, 3, 3, 3}, AffineValues(108, -1.f, 0.02f)}, {"B", {4}, AffineValues(4, 0.1f, 0.2f)}},
                        std::make_unique<ROperator_Conv<float>>("NOTSET", std::vector<size_t>{1, 1}, 1,
                                                                std::vector<size_t>{3, 3}, std::vector<size_t>{0, 0, 0, 0},
                                                                std::vector<size_t>{1, 1}, "X", "W", "B", "Y"));
   }, "StridedConv");
   EmitCpu([] {
      return BuildMulti("StridedConvTranspose", {{"X", {1, 2, 3, 3}}},
                        {{"W", {2, 3, 2, 2}, AffineValues(24, -0.5f, 0.05f)}, {"B", {3}, AffineValues(3, 0.1f, 0.2f)}},
                        std::make_unique<ROperator_ConvTranspose<float>>(
                           "NOTSET", std::vector<size_t>{1, 1}, 1, std::vector<size_t>{2, 2}, std::vector<size_t>{0, 0},
                           std::vector<size_t>{}, std::vector<size_t>{0, 0, 0, 0}, std::vector<size_t>{1, 1}, "X", "W", "B",
                           "Y"));
   }, "StridedConvTranspose");
   EmitCpu([] {
      return BuildMulti("StridedEinsum", {{"A", {3, 4}}, {"B", {4, 5}}}, {},
                        std::make_unique<ROperator_Einsum<float>>("ij,jk->ik", std::vector<std::string>{"A", "B"}, "Y"));
   }, "StridedEinsum");
   Emit([] {
      return BuildMulti("StridedSDPA", {{"Q", {1, 2, 4, 3}}, {"K", {1, 2, 4, 3}}, {"V", {1, 2, 4, 3}}}, {},
                        std::make_unique<ROperator_SDPA<float>>("Q", "K", "V", "Y"));
   }, "StridedSDPA");
   EmitCpu([] {
      return BuildMulti("StridedRNN", {{"X", {3, 2, 2}}},
                        {{"W", {1, 3, 2}, AffineValues(6, -0.5f, 0.2f)}, {"R", {1, 3, 3}, AffineValues(9, -0.4f, 0.1f)}},
                        std::make_unique<ROperator_RNN<float>>(std::vector<float>{}, std::vector<float>{},
                                                               std::vector<std::string>{}, 0.f, "forward", 3, 0, "X", "W",
                                                               "R", "", "", "", "Y", ""));
   }, "StridedRNN");
   EmitCpu([] {
      return BuildMulti("StridedLSTM", {{"X", {3, 2, 2}}},
                        {{"W", {1, 12, 2}, AffineValues(24, -0.5f, 0.04f)}, {"R", {1, 12, 3}, AffineValues(36, -0.4f, 0.025f)}},
                        std::make_unique<ROperator_LSTM<float>>(std::vector<float>{}, std::vector<float>{},
                                                                std::vector<std::string>{}, 0.f, "forward", 3, 0, 0, "X",
                                                                "W", "R", "", "", "", "", "", "Y", "", ""));
   }, "StridedLSTM");
   EmitCpu([] {
      return BuildMulti("StridedGRU", {{"X", {3, 2, 2}}},
                        {{"W", {1, 9, 2}, AffineValues(18, -0.5f, 0.05f)}, {"R", {1, 9, 3}, AffineValues(27, -0.4f, 0.03f)}},
                        std::make_unique<ROperator_GRU<float>>(std::vector<float>{}, std::vector<float>{},
                                                               std::vector<std::string>{}, 0.f, "forward", 3, 0, 0, "X",
                                                               "W", "R", "", "", "", "Y", ""));
   }, "StridedGRU");
   Emit([] {
      return BuildMulti("StridedMamba",
                        {{"u", {1, 2, 4}}, {"delta", {1, 2, 4}}, {"A", {2, 3}}, {"B", {1, 3, 4}}, {"C", {1, 3, 4}}, {"Dbias", {2}}},
                        {}, std::make_unique<ROperator_MambaScan<float>>("u", "delta", "A", "B", "C", "Dbias", "Y"));
   }, "StridedMamba");
   Emit([] {
      return BuildMulti("StridedRWKV",
                        {{"r", {1, 2, 3, 2}}, {"k", {1, 2, 3, 2}}, {"v", {1, 2, 3, 2}}, {"w", {1, 2, 3, 2}}, {"u", {2, 2}}}, {},
                        std::make_unique<ROperator_RWKV_WKV6<float>>("r", "k", "v", "w", "u", "Y"));
   }, "StridedRWKV");
   Emit([] {
      return BuildMulti("StridedGriffin", {{"X", {1, 3, 4}}, {"A", {1, 3, 4}}}, {},
                        std::make_unique<ROperator_GriffinRGLRU<float>>("X", "A", "Y"));
   }, "StridedGriffin");
   // both operands of a Gemm are graph inputs
   Emit([] {
      return BuildMulti("StridedGemmAB", {{"A", {3, 5}}, {"B", {5, 4}}}, {},
                        std::make_unique<ROperator_Gemm<float>>(1.0, 0.0, 0, 0, "A", "B", "Y"));
   }, "StridedGemmAB");
   Emit([] {
      return BuildMulti("StridedGemmABt", {{"A", {5, 3}}, {"B", {4, 5}}}, {},
                        std::make_unique<ROperator_Gemm<float>>(1.0, 0.0, 1, 1, "A", "B", "Y"));
   }, "StridedGemmABt");
   Emit(BuildGemmBiasRelu, "StridedGemmBiasRelu");
   Emit(BuildGemmTransAB, "StridedGemmTransAB");
   std::cout << "StridedInputModelGenerator: emitted CPU and GPU strided input models\n";
   return 0;
}
