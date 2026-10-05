#include "TestCustomModelsFromONNX_models.hxx"

constexpr auto modelDataSuffix = "_FromONNX.dat";
#include "common/test_helpers.h"

#include "gtest/gtest.h"

TEST(ONNX, Linear16)
{
   SofieReference ref = readReference("Linear_16");

   ASSERT_RUN(std::vector<float>, Linear_16, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Linear32)
{
   SofieReference ref = readReference("Linear_32");

   ASSERT_RUN(std::vector<float>, Linear_32, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Sub)
{
   SofieReference ref = readReference("Sub");

   ASSERT_RUN(std::vector<float>, Sub, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Add)
{
   SofieReference ref = readReference("Add");

   ASSERT_RUN(std::vector<float>, Add, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Mul)
{
   SofieReference ref = readReference("Mul");

   ASSERT_RUN(std::vector<float>, Mul, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Div)
{
   SofieReference ref = readReference("Div");

   ASSERT_RUN(std::vector<float>, Div, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Neg)
{
   SofieReference ref = readReference("Neg");

   ASSERT_RUN(std::vector<float>, Neg, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Elu)
{
   SofieReference ref = readReference("Elu");

   ASSERT_RUN(std::vector<float>, Elu, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}
TEST(ONNX, EluAlpha)
{
   SofieReference ref = readReference("EluAlpha");

   ASSERT_RUN(std::vector<float>, EluAlpha, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Constant)
{
   SofieReference ref = readReference("Constant");

   ASSERT_RUN_0(std::vector<float>, Constant);

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ComplexTopK)
{
   SofieReference ref = readReference("ComplexTopK");

   ASSERT_RUN(TupleFloatInt64_t, ComplexTopK, ref.f32("input0"));

   expectNear(std::get<0>(output), ref.f32("output0"), DEFAULT_TOLERANCE);
   expectEqual(std::get<1>(output), ref.i64("output1"));
}
TEST(ONNX, TopK)
{
   SofieReference ref = readReference("TopK");

   ASSERT_RUN(TupleFloatInt64_t, TopK, ref.f32("input0"));

   expectNear(std::get<0>(output), ref.f32("output0"), DEFAULT_TOLERANCE);
   expectEqual(std::get<1>(output), ref.i64("output1"));
}
   TEST(ONNX, EyeLike)
{
   SofieReference ref = readReference("EyeLike");

   ASSERT_RUN(std::vector<float>, EyeLike, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Cast)
{
   SofieReference ref = readReference("Cast");

   ASSERT_RUN(std::vector<double>, Cast, ref.i64("input0"));

   expectNear(output, ref.f64("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Linear64)
{
   SofieReference ref = readReference("Linear_64");

   ASSERT_RUN(std::vector<float>, Linear_64, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LinearWithSelu)
{
   SofieReference ref = readReference("LinearWithSelu");

   ASSERT_RUN(std::vector<float>, LinearWithSelu, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Tanh)
{
   SofieReference ref = readReference("Tanh");

   ASSERT_RUN(std::vector<float>, Tanh, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Erf)
{
   SofieReference ref = readReference("Erf");

   ASSERT_RUN(std::vector<float>, Erf, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Log)
{
   SofieReference ref = readReference("Log");

   ASSERT_RUN(std::vector<float>, Log, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LinearWithLeakyRelu)
{
   SofieReference ref = readReference("LinearWithLeakyRelu");

   ASSERT_RUN(std::vector<float>, LinearWithLeakyRelu, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LinearWithSigmoid)
{
   SofieReference ref = readReference("LinearWithSigmoid");

   ASSERT_RUN(std::vector<float>, LinearWithSigmoid, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithPadding)
{
   SofieReference ref = readReference("ConvWithPadding");

   ASSERT_RUN(std::vector<float>, ConvWithPadding, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithoutPadding)
{
   SofieReference ref = readReference("ConvWithoutPadding");

   ASSERT_RUN(std::vector<float>, ConvWithoutPadding, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithAutopadSameLower)
{
   SofieReference ref = readReference("ConvWithAutopadSameLower");

   ASSERT_RUN(std::vector<float>, ConvWithAutopadSameLower, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithAutopadSameUpper)
{
   SofieReference ref = readReference("ConvWithAutopadSameUpper");

   ASSERT_RUN(std::vector<float>, ConvWithAutopadSameUpper, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithStridesPadding)
{
   SofieReference ref = readReference("ConvWithStridesPadding");

   ASSERT_RUN(std::vector<float>, ConvWithStridesPadding, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithDilation)
{
   SofieReference ref = readReference("ConvWithDilation");

   ASSERT_RUN(std::vector<float>, ConvWithDilation, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithStridesNoPadding)
{
   SofieReference ref = readReference("ConvWithStridesNoPadding");

   ASSERT_RUN(std::vector<float>, ConvWithStridesNoPadding, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvAddRelu)
{
   SofieReference ref = readReference("ConvAddRelu");

   ASSERT_RUN(std::vector<float>, ConvAddRelu, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithDynShapeStride)
{

   std::vector<float> input = {0, 1, 2, 3, 4, 5, 6};
   std::vector<float> correct_output = {3, 9, 15};

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, ConvWithDynShapeStride, ("ConvWithDynShapeStride_FromONNX.dat", 7), 7, input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithAsymmetricPadding)
{
   SofieReference ref = readReference("ConvWithAsymmetricPadding");

   ASSERT_RUN(std::vector<float>, ConvWithAsymmetricPadding, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvSameUpperEvenKernel)
{
   SofieReference ref = readReference("ConvSameUpperEvenKernel");

   ASSERT_RUN(std::vector<float>, ConvSameUpperEvenKernel, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvSameLowerEvenKernel)
{
   SofieReference ref = readReference("ConvSameLowerEvenKernel");

   ASSERT_RUN(std::vector<float>, ConvSameLowerEvenKernel, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvAsymmetricPads1d)
{
   SofieReference ref = readReference("ConvAsymmetricPads1d");

   ASSERT_RUN(std::vector<float>, ConvAsymmetricPads1d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvAsymmetricPads2d)
{
   SofieReference ref = readReference("ConvAsymmetricPads2d");

   ASSERT_RUN(std::vector<float>, ConvAsymmetricPads2d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvAsymmetricPads3d)
{
   SofieReference ref = readReference("ConvAsymmetricPads3d");

   ASSERT_RUN(std::vector<float>, ConvAsymmetricPads3d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvAsymmetricPadsGrouped)
{
   SofieReference ref = readReference("ConvAsymmetricPadsGrouped");

   ASSERT_RUN(std::vector<float>, ConvAsymmetricPadsGrouped, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool1d)
{
   SofieReference ref = readReference("MaxPool1d");

   ASSERT_RUN(std::vector<float>, MaxPool1d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool2d)
{
   SofieReference ref = readReference("MaxPool2d");

   ASSERT_RUN(std::vector<float>, MaxPool2d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool2d_AsymPad)
{
   SofieReference ref = readReference("MaxPool2d_AsymPad");

   ASSERT_RUN(std::vector<float>, MaxPool2d_AsymPad, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool2d_CeilMode)
{
   SofieReference ref = readReference("MaxPool2d_CeilMode");

   ASSERT_RUN(std::vector<float>, MaxPool2d_CeilMode, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool1d_CeilMode_Overhang)
{
   SofieReference ref = readReference("MaxPool1d_CeilMode_Overhang");

   ASSERT_RUN(std::vector<float>, MaxPool1d_CeilMode_Overhang, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool2d_CeilMode_Pads)
{
   SofieReference ref = readReference("MaxPool2d_CeilMode_Pads");

   ASSERT_RUN(std::vector<float>, MaxPool2d_CeilMode_Pads, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MaxPool3d)
{
   SofieReference ref = readReference("MaxPool3d");

   ASSERT_RUN(std::vector<float>, MaxPool3d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool1d_CeilMode)
{
   SofieReference ref = readReference("AveragePool1d_CeilMode");

   ASSERT_RUN(std::vector<float>, AveragePool1d_CeilMode, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool1d_CeilMode_Overhang)
{
   SofieReference ref = readReference("AveragePool1d_CeilMode_Overhang");

   ASSERT_RUN(std::vector<float>, AveragePool1d_CeilMode_Overhang, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool2d_CeilMode)
{
   SofieReference ref = readReference("AveragePool2d_CeilMode");

   ASSERT_RUN(std::vector<float>, AveragePool2d_CeilMode, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool2d_CeilMode_Pads)
{
   SofieReference ref = readReference("AveragePool2d_CeilMode_Pads");

   ASSERT_RUN(std::vector<float>, AveragePool2d_CeilMode_Pads, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool2d_CeilMode_CountIncludePad)
{
   SofieReference ref = readReference("AveragePool2d_CeilMode_CountIncludePad");

   ASSERT_RUN(std::vector<float>, AveragePool2d_CeilMode_CountIncludePad, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool2d_Pads_CountIncludePad)
{
   SofieReference ref = readReference("AveragePool2d_Pads_CountIncludePad");

   ASSERT_RUN(std::vector<float>, AveragePool2d_Pads_CountIncludePad, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AveragePool3d_CeilMode)
{
   SofieReference ref = readReference("AveragePool3d_CeilMode");

   ASSERT_RUN(std::vector<float>, AveragePool3d_CeilMode, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AvgPool)
{
   SofieReference ref = readReference("AvgPool");

   ASSERT_RUN(std::vector<float>, AvgPool, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Pow)
{
   SofieReference ref = readReference("Pow");

   ASSERT_RUN(std::vector<float>, Pow, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Pow_broadcast)
{
   SofieReference ref = readReference("Pow_broadcast");

   ASSERT_RUN(std::vector<float>, Pow_broadcast, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, FMod_ConstantFolding)
{
   std::vector<float> correct_output = {1, 1, 2};
   ASSERT_RUN_0(std::vector<float>, FMod_ConstantFolding);
   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Mod_ConstantFolding)
{
   std::vector<int64_t> correct_output = {1, 1, 2};
   ASSERT_RUN_0(std::vector<int64_t>, Mod_ConstantFolding);
   expectEqual(output, correct_output);
}

TEST(ONNX, Gemm_ConstantFolding)
{
   std::vector<float> correct_output = {59, 65, 140, 155};
   ASSERT_RUN_0(std::vector<float>, Gemm_ConstantFolding);
   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Gemm_ConstantFolding_Shared)
{
   std::vector<float> A2 = {5, 6, 7, 8};
   std::vector<float> correct_Y1 = {2, 3, 4, 5};
   std::vector<float> correct_Y2 = {15, 16, 17, 18};

   ASSERT_RUN(std::vector<std::vector<float>>, Gemm_ConstantFolding_Shared, A2);

   ASSERT_EQ(output.size(), 2u);
   expectNear(output[0], correct_Y1, DEFAULT_TOLERANCE);
   expectNear(output[1], correct_Y2, DEFAULT_TOLERANCE);
}

   TEST(ONNX, ReduceMean)
{
   SofieReference ref = readReference("ReduceMean");

   ASSERT_RUN(std::vector<float>, ReduceMean, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ReduceMean_kFirst)
{
   std::vector<float> input(12);
   std::iota(input.begin(), input.end(), 0.0f);
   std::vector<float> correct_output = {4, 5, 6, 7};

   ASSERT_RUN(std::vector<float>, ReduceMean_kFirst, input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ReduceMax)
{
   std::vector<float> input({5, 2, 3, 5, 5, 4});
   std::vector<float> correct_output({5, 5, 4});

   ASSERT_RUN(std::vector<float>, ReduceMax, input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ReduceMin)
{
   std::vector<float> input({5, 2, 3, 5, 5, 4});
   std::vector<float> correct_output({5, 2, 3});

   ASSERT_RUN(std::vector<float>, ReduceMin, input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, EluDynShape)
{
   std::vector<float> input({-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, -1.0, 3.0});
   std::vector<float> correct_output;
   for (float x : input)
      correct_output.push_back(x >= 0 ? x : std::exp(x) - 1);

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, EluDynShape, ("EluDynShape_FromONNX.dat", 2), 2, input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, TopKWithDynShapeK)
{
   std::vector<float> input({5, 1, 9, 2, 8, 3, 7, 4, 6, 0, 5, 5, 3, 3, 3});
   std::vector<float> correct_values({7, 8, 9, 5, 5, 6, 3, 4, 5, 2, 3, 3});
   std::vector<int64_t> correct_indices({2, 1, 0, 0, 3, 2, 4, 2, 3, 1, 4, 1});

   ASSERT_RUN_SESSION_ARGS(TupleFloatInt64_t, TopKWithDynShapeK, ("TopKWithDynShapeK_FromONNX.dat", 5),
                                       5, input);

   expectNear(std::get<0>(output), correct_values, DEFAULT_TOLERANCE);
   expectEqual(std::get<1>(output), correct_indices);
}

TEST(ONNX, ReduceMean_kMiddle_DynShape)
{
   std::vector<float> input(24);
   std::iota(input.begin(), input.end(), 0.0f);
   std::vector<float> correct_output = {4, 5, 6, 7, 16, 17, 18, 19};

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, ReduceMean_kMiddle_DynShape, ("ReduceMean_kMiddle_DynShape_FromONNX.dat", 2), 2, input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, TopKLargestUnsorted)
{
   std::vector<float> input({1, 6, 3, 2, 5, 4, 10, 40, 20, 60, 30, 50});
   std::vector<float> correct_values({6, 5, 4, 60, 50, 40});
   std::vector<int64_t> correct_indices({1, 4, 5, 3, 5, 1});

   ASSERT_RUN(TupleFloatInt64_t, TopKLargestUnsorted, input);

   expectNear(std::get<0>(output), correct_values, DEFAULT_TOLERANCE);
   expectEqual(std::get<1>(output), correct_indices);
}

   TEST(ONNX, ReduceProd)
{
   SofieReference ref = readReference("ReduceProd");

   ASSERT_RUN(std::vector<float>, ReduceProd, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ReduceSum){
   std::vector<float> input({
      5, 2, 3,
      5, 5, 4
   });

   ASSERT_RUN(std::vector<float>, ReduceSum, input);
   std::vector<float> correct{24};

   expectNear(output, correct, DEFAULT_TOLERANCE);
}

TEST(ONNX, ReduceSumSquare){
   std::vector<float> input({
      5, 2, 3,
      5, 5, 4
   });

   ASSERT_RUN(std::vector<float>, ReduceSumSquare, input);
   std::vector<float> correct{38, 66};

   expectNear(output, correct, DEFAULT_TOLERANCE);
}

TEST(ONNX, Max)
{
   SofieReference ref = readReference("Max");

   ASSERT_RUN(std::vector<float>, Max, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MinInt64)
{
   std::vector<int64_t> a({1, -7, 3, 100, 0});
   std::vector<int64_t> b({2, -2, -3, 50, 0});
   std::vector<int64_t> c({0, 5, 9, 75, 1});
   std::vector<int64_t> correct_output({0, -7, -3, 50, 0});

   ASSERT_RUN(std::vector<int64_t>, MinInt64, a, b, c);

   expectEqual(output, correct_output);
}

TEST(ONNX, MaxInt64)
{
   std::vector<int64_t> a({1, -7, 3, 100, 0});
   std::vector<int64_t> b({2, -2, -3, 50, 0});
   std::vector<int64_t> c({0, 5, 9, 75, 1});
   std::vector<int64_t> correct_output({2, 5, 9, 100, 1});

   ASSERT_RUN(std::vector<int64_t>, MaxInt64, a, b, c);

   expectEqual(output, correct_output);
}

TEST(ONNX, MaxMultidirectionalBroadcast)
{
   SofieReference ref = readReference("MaxMultidirectionalBroadcast");

   ASSERT_RUN(
      std::vector<float>, MaxMultidirectionalBroadcast,
      ref.f32("input0"),
      ref.f32("input1"),
      ref.f32("input2"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MinMultidirectionalBroadcast)
{
   SofieReference ref = readReference("MinMultidirectionalBroadcast");

   ASSERT_RUN(
      std::vector<float>, MinMultidirectionalBroadcast,
      ref.f32("input0"),
      ref.f32("input1"),
      ref.f32("input2"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, MeanMultidirectionalBroadcast)
{
   SofieReference ref = readReference("MeanMultidirectionalBroadcast");

   ASSERT_RUN(
      std::vector<float>, MeanMultidirectionalBroadcast,
      ref.f32("input0"),
      ref.f32("input1"),
      ref.f32("input2"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, SumMultidirectionalBroadcast)
{
   SofieReference ref = readReference("SumMultidirectionalBroadcast");

   ASSERT_RUN(
      std::vector<float>, SumMultidirectionalBroadcast,
      ref.f32("input0"),
      ref.f32("input1"),
      ref.f32("input2"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Shape)
{
   SofieReference ref = readReference("Shape");

   ASSERT_RUN(std::vector<float>, Shape, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNBatchwise)
{
   SofieReference ref = readReference("RNNBatchwise");

   ASSERT_RUN(std::vector<std::vector<float>>, RNNBatchwise, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNBidirectional)
{
   SofieReference ref = readReference("RNNBidirectional");

   ASSERT_RUN(std::vector<std::vector<float>>, RNNBidirectional, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNBidirectionalBatchwise)
{
   SofieReference ref = readReference("RNNBidirectionalBatchwise");

   ASSERT_RUN(std::vector<std::vector<float>>, RNNBidirectionalBatchwise, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNDefaults)
{
   SofieReference ref = readReference("RNNDefaults");

   ASSERT_RUN(std::vector<std::vector<float>>, RNNDefaults, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNClip)
{
   SofieReference ref = readReference("RNNClip");

   ASSERT_RUN(std::vector<std::vector<float>>, RNNClip, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNSeqLength)
{
   SofieReference ref = readReference("RNNSeqLength");

   ASSERT_RUN(std::vector<std::vector<float>>, RNNSeqLength, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNSequence)
{
   SofieReference ref = readReference("RNNSequence");

   ASSERT_RUN(std::vector<std::vector<float>>, RNNSequence, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNSequenceBatchwise)
{
   SofieReference ref = readReference("RNNSequenceBatchwise");

   ASSERT_RUN(std::vector<std::vector<float>>, RNNSequenceBatchwise, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMBatchwise)
{
   SofieReference ref = readReference("LSTMBatchwise");

   ASSERT_RUN(std::vector<std::vector<float>>, LSTMBatchwise, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMBidirectional)
{
   SofieReference ref = readReference("LSTMBidirectional");

   ASSERT_RUN(std::vector<std::vector<float>>, LSTMBidirectional, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
   expectNear(output[2], ref.f32("output2"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMDefaults)
{
   SofieReference ref = readReference("LSTMDefaults");

   ASSERT_RUN(std::vector<std::vector<float>>, LSTMDefaults, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMInitialBias)
{
   SofieReference ref = readReference("LSTMInitialBias");

   ASSERT_RUN(std::vector<std::vector<float>>, LSTMInitialBias, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMPeepholes)
{
   SofieReference ref = readReference("LSTMPeepholes");

   ASSERT_RUN(std::vector<std::vector<float>>, LSTMPeepholes, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUBatchwise)
{
   SofieReference ref = readReference("GRUBatchwise");

   ASSERT_RUN(std::vector<std::vector<float>>, GRUBatchwise, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUBidirectional)
{
   SofieReference ref = readReference("GRUBidirectional");

   ASSERT_RUN(std::vector<std::vector<float>>, GRUBidirectional, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUDefaults)
{
   SofieReference ref = readReference("GRUDefaults");

   ASSERT_RUN(std::vector<std::vector<float>>, GRUDefaults, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUInitialBias)
{
   SofieReference ref = readReference("GRUInitialBias");

   ASSERT_RUN(std::vector<std::vector<float>>, GRUInitialBias, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUSeqLength)
{
   SofieReference ref = readReference("GRUSeqLength");

   ASSERT_RUN(std::vector<std::vector<float>>, GRUSeqLength, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Softmax1d)
{
   SofieReference ref = readReference("Softmax1d");

   ASSERT_RUN(std::vector<float>, Softmax1d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Softmax2d)
{
   SofieReference ref = readReference("Softmax2d");

   ASSERT_RUN(std::vector<float>, Softmax2d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Softmax3d)
{
   SofieReference ref = readReference("Softmax3d");

   ASSERT_RUN(std::vector<float>, Softmax3d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Softmax4d)
{
   SofieReference ref = readReference("Softmax4d");

   ASSERT_RUN(std::vector<float>, Softmax4d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LogSoftmaxLargeRange)
{
   SofieReference ref = readReference("LogSoftmaxLargeRange");

   ASSERT_RUN(std::vector<float>, LogSoftmaxLargeRange, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LogSoftmaxLargeRangeAxis0)
{
   SofieReference ref = readReference("LogSoftmaxLargeRangeAxis0");

   ASSERT_RUN(std::vector<float>, LogSoftmaxLargeRangeAxis0, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvTranspose1d)
{
   SofieReference ref = readReference("ConvTranspose1d");

   ASSERT_RUN(std::vector<float>, ConvTranspose1d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvTranspose2d)
{
   SofieReference ref = readReference("ConvTranspose2d");

   ASSERT_RUN(std::vector<float>, ConvTranspose2d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvTranspose2dOutputShape)
{
   SofieReference ref = readReference("ConvTranspose2dOutputShape");

   ASSERT_RUN(std::vector<float>, ConvTranspose2dOutputShape, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvTransposeBias2d)
{
   SofieReference ref = readReference("ConvTransposeBias2d");

   ASSERT_RUN(std::vector<float>, ConvTransposeBias2d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvTransposeBias2dBatched)
{
   SofieReference ref = readReference("ConvTransposeBias2dBatched");

   ASSERT_RUN(std::vector<float>, ConvTransposeBias2dBatched, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Sqrt)
{
   SofieReference ref = readReference("Sqrt");

   ASSERT_RUN(std::vector<float>, Sqrt, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Reciprocal)
{
   SofieReference ref = readReference("Reciprocal");

   ASSERT_RUN(std::vector<float>, Reciprocal, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Exp)
{
   SofieReference ref = readReference("Exp");

   ASSERT_RUN(std::vector<float>, Exp, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast1)
{
   SofieReference ref = readReference("AddBroadcast1");

   ASSERT_RUN(std::vector<float>, AddBroadcast1, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast2)
{
   SofieReference ref = readReference("AddBroadcast2");

   ASSERT_RUN(std::vector<float>, AddBroadcast2, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast3)
{
   SofieReference ref = readReference("AddBroadcast3");

   ASSERT_RUN(std::vector<float>, AddBroadcast3, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast4)
{
   SofieReference ref = readReference("AddBroadcast4");

   ASSERT_RUN(std::vector<float>, AddBroadcast4, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast5)
{
   SofieReference ref = readReference("AddBroadcast5");

   ASSERT_RUN(std::vector<float>, AddBroadcast5, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast6)
{
   SofieReference ref = readReference("AddBroadcast6");

   ASSERT_RUN(std::vector<float>, AddBroadcast6, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AddBroadcast7)
{
   SofieReference ref = readReference("AddBroadcast7");

   ASSERT_RUN(std::vector<float>, AddBroadcast7, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Concat0D) {
   std::vector<float> input({1.40519865e+00, -2.87660856e-01});
   std::vector<float> expected_output({1.40519865e+00, -2.87660856e-01, 1.40519865e+00, -2.87660856e-01});
   ASSERT_RUN(std::vector<float>, Concat_0D, input);

   expectNear(output, expected_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, LayerNormalization2d)
{
   SofieReference ref = readReference("LayerNormalization2d");

   ASSERT_RUN(std::vector<float>, LayerNormalization2d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LayerNormalization4d)
{
   SofieReference ref = readReference("LayerNormalization4d");

   ASSERT_RUN(std::vector<float>, LayerNormalization4d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, InstanceNormalization)
{
   SofieReference ref = readReference("InstanceNormalization");

   ASSERT_RUN(std::vector<float>, InstanceNormalization, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, InstanceNormalization3d)
{
   SofieReference ref = readReference("InstanceNormalization3d");

   ASSERT_RUN(std::vector<float>, InstanceNormalization3d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, InstanceNormalizationEpsilon)
{
   SofieReference ref = readReference("InstanceNormalizationEpsilon");

   ASSERT_RUN(std::vector<float>, InstanceNormalizationEpsilon, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Equal)
{
   SofieReference ref = readReference("Equal");

   ASSERT_RUN(std::vector<std::uint8_t>, Equal, ref.f32("input0"), ref.f32("input1"));

   expectEqual(output, ref.u8("output0"));
}

TEST(ONNX, LessOrEqual)
{
   SofieReference ref = readReference("LessOrEqual");

   ASSERT_RUN(std::vector<std::uint8_t>, LessOrEqual, ref.f32("input0"), ref.f32("input1"));

   expectEqual(output, ref.u8("output0"));
}

TEST(ONNX, GreaterOrEqual)
{
   SofieReference ref = readReference("GreaterOrEqual");

   ASSERT_RUN(std::vector<std::uint8_t>, GreaterOrEqual, ref.f32("input0"), ref.f32("input1"));

   expectEqual(output, ref.u8("output0"));
}

TEST(ONNX, Greater)
{
   SofieReference ref = readReference("Greater");

   ASSERT_RUN(std::vector<std::uint8_t>, Greater, ref.f32("input0"), ref.f32("input1"));

   expectEqual(output, ref.u8("output0"));
}

TEST(ONNX, Less)
{
   SofieReference ref = readReference("Less");

   ASSERT_RUN(std::vector<std::uint8_t>, Less, ref.f32("input0"), ref.f32("input1"));

   expectEqual(output, ref.u8("output0"));
}

TEST(ONNX, ExpandSameSize)
{
   SofieReference ref = readReference("ExpandSameSize");

   ASSERT_RUN(std::vector<float>, ExpandSameSize, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ExpandDiffSize)
{
   SofieReference ref = readReference("ExpandDiffSize");

   ASSERT_RUN(std::vector<float>, ExpandDiffSize, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherAxis0)
{
   SofieReference ref = readReference("GatherAxis0");

   ASSERT_RUN(std::vector<float>, GatherAxis0, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherAxis1)
{
   SofieReference ref = readReference("GatherAxis1");

   ASSERT_RUN(std::vector<float>, GatherAxis1, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherAxis2)
{
   SofieReference ref = readReference("GatherAxis2");

   ASSERT_RUN(std::vector<float>, GatherAxis2, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherAxis3)
{
   SofieReference ref = readReference("GatherAxis3");

   ASSERT_RUN(std::vector<float>, GatherAxis3, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Gather2d)
{
   SofieReference ref = readReference("Gather2d");

   ASSERT_RUN(std::vector<float>, Gather2d, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherNegativeIndices)
{
   SofieReference ref = readReference("GatherNegativeIndices");

   ASSERT_RUN(std::vector<float>, GatherNegativeIndices, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherRuntimeNegativeIndices)
{
   SofieReference ref = readReference("GatherRuntimeNegativeIndices");

   ASSERT_RUN(std::vector<float>, GatherRuntimeNegativeIndices, ref.f32("input0"), ref.i64("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Slice)
{
   SofieReference ref = readReference("Slice");

   ASSERT_RUN(std::vector<float>, Slice, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Slice_Default_Axis)
{
   SofieReference ref = readReference("Slice_Default_Axis");

   ASSERT_RUN(std::vector<float>, Slice_Default_Axis, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Slice_Default_Steps)
{
   SofieReference ref = readReference("Slice_Default_Steps");

   ASSERT_RUN(std::vector<float>, Slice_Default_Steps, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Slice_Neg)
{
   SofieReference ref = readReference("Slice_Neg");

   ASSERT_RUN(std::vector<float>, Slice_Neg, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}
TEST(ONNX, RangeFloat)
{
   SofieReference ref = readReference("RangeFloat");

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, RangeFloat, ("RangeFloat_FromONNX.dat", 5),
      ref.f32("input0"),
      ref.f32("input1"),
      ref.f32("input2"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RangeInt)
{
   SofieReference ref = readReference("RangeInt");

   ASSERT_RUN_SESSION_ARGS(std::vector<int64_t>, RangeInt, ("RangeInt_FromONNX.dat", 5),
      ref.i64("input0"),
      ref.i64("input1"),
      ref.i64("input2"));

   expectEqual(output, ref.i64("output0"));
}

TEST(ONNX, Range_ConstantFolding)
{
   std::vector<int64_t> correct_output = {0, 2, 4};
   ASSERT_RUN_0(std::vector<int64_t>, Range_ConstantFolding);
   expectEqual(output, correct_output);
}

TEST(ONNX, RangeWithDynShapeStart)
{
   std::vector<float> input(5);
   std::vector<int64_t> correct_output = {1, 2, 3, 4};

   ASSERT_RUN_SESSION_ARGS(std::vector<int64_t>, RangeWithDynShapeStart, ("RangeWithDynShapeStart_FromONNX.dat", 5), 5, input);

   expectEqual(output, correct_output);
}

TEST(ONNX, RangeWithDynShapeDelta)
{
   std::vector<float> input(5);
   std::vector<int64_t> correct_output = {0, 2, 4};

   ASSERT_RUN_SESSION_ARGS(std::vector<int64_t>, RangeWithDynShapeDelta, ("RangeWithDynShapeDelta_FromONNX.dat", 5), 5, input);

   expectEqual(output, correct_output);
}

TEST(ONNX, RangeWithDynShapeStartDelta)
{
   std::vector<float> input(5);
   std::vector<int64_t> correct_output = {1, 3};

   ASSERT_RUN_SESSION_ARGS(std::vector<int64_t>, RangeWithDynShapeStartDelta, ("RangeWithDynShapeStartDelta_FromONNX.dat", 5), 5, input);

   expectEqual(output, correct_output);
}
TEST(ONNX, Tile5D)
{
   SofieReference ref = readReference("Tile5D");

   ASSERT_RUN(std::vector<float>, Tile5D, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}
TEST(ONNX, Pad) {
   std::vector<float> input = {1,2,3,4};
   std::vector<float> correct = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 3,
       4, 0, 0, 0, 0, 0, 0, 0};
   ASSERT_RUN(std::vector<float>, Pad, input);

   expectEqual(output, correct);
}
TEST(ONNX, Where) {
   std::vector<float> input1 = {1,2};
   std::vector<float> input2 = {3,4,5,6};
   std::vector<uint8_t> cond = {true, false, true};
   std::vector<float> correct = {1,2,5,6,1,2};
   ASSERT_RUN(std::vector<float>, Where, input1, input2, cond);

   expectEqual(output, correct);
}
TEST(ONNX, WhereMultidirectionalBroadcast)
{
   std::vector<float> input1 = {1, 2, 3};
   std::vector<float> input2 = {-1};
   std::vector<uint8_t> cond = {true, false};
   std::vector<float> correct = {1, 2, 3, -1, -1, -1};
   ASSERT_RUN(std::vector<float>, WhereMultidirectionalBroadcast, input1, input2, cond);

   expectEqual(output, correct);
}
TEST(ONNX, WhereBroadcastHighRankCond)
{
   std::vector<float> input1 = {10, 20, 30, 40};
   std::vector<float> input2 = {-1};
   std::vector<uint8_t> cond = {true, false};
   std::vector<float> correct = {10, 20, 30, 40, -1, -1, -1, -1};
   ASSERT_RUN(std::vector<float>, WhereBroadcastHighRankCond, input1, input2, cond);

   expectEqual(output, correct);
}
TEST(ONNX, WhereBroadcastEqualElementCount)
{
   std::vector<float> input1 = {1, 2};
   std::vector<float> input2 = {-1, -2};
   std::vector<uint8_t> cond = {true, false};
   std::vector<float> correct = {1, 2, -1, -2};
   ASSERT_RUN(std::vector<float>, WhereBroadcastEqualElementCount, input1, input2, cond);

   expectEqual(output, correct);
}

// If operator (subgraphs): both branches read the outer-scope input X
TEST(ONNX, IfSimpleThen)
{
   std::vector<uint8_t> cond = {true};
   std::vector<float> x = {1, -2, 3, -4, 5, -6};
   std::vector<float> correct = {1, 4, 9, 16, 25, 36};
   ASSERT_RUN(std::vector<float>, IfSimple, cond, x);

   expectEqual(output, correct);
}
TEST(ONNX, IfSimpleElse)
{
   std::vector<uint8_t> cond = {false};
   std::vector<float> x = {1, -2, 3, -4, 5, -6};
   std::vector<float> correct = {-1, 2, -3, 4, -5, 6};
   ASSERT_RUN(std::vector<float>, IfSimple, cond, x);

   expectEqual(output, correct);
}
// two outputs per branch, branch-local initializers and an operator reading the If outputs
TEST(ONNX, IfTwoOutputsThen)
{
   std::vector<uint8_t> cond = {true};
   std::vector<float> x = {1, -2, 3, -4};
   std::vector<float> correct = {4, -3, 10, -6};
   ASSERT_RUN(std::vector<float>, IfTwoOutputs, cond, x);

   expectEqual(output, correct);
}
TEST(ONNX, IfTwoOutputsElse)
{
   std::vector<uint8_t> cond = {false};
   std::vector<float> x = {1, -2, 3, -4};
   std::vector<float> correct = {1, 0, 5, 0};
   ASSERT_RUN(std::vector<float>, IfTwoOutputs, cond, x);

   expectEqual(output, correct);
}

TEST(ONNX, Sin)
{
   std::vector<float> input({
     -0.786738,-0.197796,-0.187787,0.142758,0.876096,-0.653239,0.145444,-1.107658,2.259171,-0.947054,-0.506689,1.801250
   });

   ASSERT_RUN(std::vector<float>, Sin, input);

   std::vector<float> correct_output;
   for (float x : input)
      correct_output.push_back(std::sin(x));

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Asinh)
{
   std::vector<float> input({
     -0.786738,-0.197796,-0.187787,0.142758,0.876096,-0.653239,0.145444,-1.107658,2.259171,-0.947054,-0.506689,1.801250
   });

   ASSERT_RUN(std::vector<float>, Asinh, input);

   std::vector<float> correct_output;
   for (float x : input)
      correct_output.push_back(std::asinh(x));
   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Acosh)
{
   std::vector<float> input({
     1.0, 1.001, 1.5, 2.0, 3.789, 5.234, 10.0, 1.234, 7.891, 2.345, 1.999, 100.0
   });

   ASSERT_RUN(std::vector<float>, Acosh, input);

   std::vector<float> correct_output;
   for (float x : input)
      correct_output.push_back(std::acosh(x));
   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Atanh)
{
   std::vector<float> input({
     -0.99,-0.786738,-0.5,-0.197796,0.0,0.142758,0.5,0.876096,-0.653239,0.3,0.99,-0.142758
   });

   ASSERT_RUN(std::vector<float>, Atanh, input);

   std::vector<float> correct_output;
   for (float x : input)
      correct_output.push_back(std::atanh(x));
   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Cos)
{
   std::vector<float> input({
     1.152504,-1.459324,0.691594,0.347690,-1.307323,1.832516,-1.261772,0.014224,1.311477,1.147405,-0.567206,-0.530606
   });

   ASSERT_RUN(std::vector<float>, Cos, input);

   std::vector<float> correct_output;
   for (float x : input)
      correct_output.push_back(std::cos(x));

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Abs)
{
   std::vector<float> input({1.,-2.,-3,4,-5.,6});

   ASSERT_RUN(std::vector<float>, Abs, input);

   std::vector<float> correct_output;
   for (float x : input)
      correct_output.push_back(std::abs(x));

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Softplus)
{
   std::vector<float> input({0.1f, -0.2f, 100.0f, 89.0f, 0.0f, 50.0f});

   ASSERT_RUN(std::vector<float>, Softplus, input);

   ASSERT_EQ(output.size(), input.size());

   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_FALSE(std::isinf(output[i])) << "Inf at input=" << input[i];
      EXPECT_FALSE(std::isnan(output[i])) << "NaN at input=" << input[i];
      if (input[i] >= 20.0f) {
         EXPECT_NEAR(output[i], input[i], DEFAULT_TOLERANCE);
      } else {
         float exp_value = std::log1p(std::exp(input[i]));
         EXPECT_LE(std::abs(output[i] - exp_value), DEFAULT_TOLERANCE);
      }
   }
}
TEST(ONNX, Einsum_matmul)
{
   std::vector<float> input1{1, 2, 3, 4};
   std::vector<float> input2{5, 6, 7, 8};
   std::vector<float> correct_output = {19, 22, 43, 50};

   ASSERT_RUN(std::vector<float>, Einsum_matmul, input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}
TEST(ONNX, Einsum_dotprod)
{
   std::vector<float> input1{1, 2, 3};
   std::vector<float> input2{5, 6, 7};
   std::vector<float> correct_output {5 +  12 + 21};

   ASSERT_RUN(std::vector<float>, Einsum_dotprod, input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}
TEST(ONNX, Einsum_3)
{
   std::vector<float> input1 {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<float> input2 {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<float> correct_output {66. , 87. , 108., 498.,  555., 612. };

   ASSERT_RUN(std::vector<float>, Einsum_3, input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}
TEST(ONNX, Einsum_4)
{
   std::vector<float> input1 {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<float> input2 {1.,2.,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
   std::vector<float> correct_output { 14., 32.,  50., 32.,  77.,  122.,
                                      266., 338., 410., 365., 464., 563. };

   ASSERT_RUN(std::vector<float>, Einsum_4, input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}


TEST(ONNX, Split_0)
{
   std::vector<float> input {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<std::vector<float>> correct_output ={ {1,2,3,4,5,6}, {7,8,9,10,11,12} };

   ASSERT_RUN(std::vector<std::vector<float>>, Split_0, input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Split_1)
{
   std::vector<float> input {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<std::vector<float>> correct_output ={ {1,2,3,7,8,9}, {4,5,6,10,11,12} };

   ASSERT_RUN(std::vector<std::vector<float>>, Split_1, input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Split_2)
{
   std::vector<float> input {1.,2.,3,4,5,6,7,8,9,10,11,12};
   std::vector<std::vector<float>> correct_output ={ {1,2,4,5,7,8,10,11}, {3,6,9,12} };

   ASSERT_RUN(std::vector<std::vector<float>>, Split_2, input);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ScatterElements)
{
   std::vector<float> input(9, 0.);
   std::vector<int64_t> indices = { 1, 0, 2, 0, 2, 1};
   std::vector<float> updates = { 1, 1.1, 1.2, 2, 2.1, 2.2};
   std::vector<float> correct_output = {2, 1.1, 0., 1., 0., 2.2, 0., 2.1, 1.2 };

   ASSERT_RUN(std::vector<float>, ScatterElements, input, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, MatMul_1D_Constant)
{
   std::vector<float> correct_output = {22, 28};
   ASSERT_RUN_0(std::vector<float>, MatMul_1D_Constant);
   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, MatMul_Stacked)
{
   std::vector<float> input1 = {1,2,3,4,5,6,7,8};
   std::vector<float> input2 = {2,3};

   std::vector<float> correct_output = {8,18, 28,38};

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, MatMul_Stacked, ("MatMul_Stacked_FromONNX.dat", 2), 2, input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, MatMul_Stacked2)
{
   std::vector<float> input1 = {1,2,3,4,5,6,7,8};
   std::vector<float> input2 = {2,3,3,2};

   std::vector<float> correct_output = {8,18, 27,37};

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, MatMul_Stacked2, ("MatMul_Stacked2_FromONNX.dat", 2), 2, input1, input2);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, GatherND_1)
{
   std::vector<float> input(18, 0.);
   std::iota(input.begin(), input.end(), 1.);
   std::vector<int64_t> indices = { 1, 0, 2,   0, 2, 1};
   std::vector<float> correct_output = {12, 8};

   ASSERT_RUN(std::vector<float>, GatherND_1, input, indices);

   expectEqual(output, correct_output);
}

TEST(ONNX, GatherND_2)
{
   std::vector<float> input(18, 0.);
   std::iota(input.begin(), input.end(), 1.);
   std::vector<int64_t> indices = { 1, 1, 0, 2};
   std::vector<float> correct_output = {13,14,15, 7,8,9};

   ASSERT_RUN(std::vector<float>, GatherND_2, input, indices);

   expectEqual(output, correct_output);
}

TEST(ONNX, GatherND_3)
{
   std::vector<float> input(24, 0.);
   std::iota(input.begin(), input.end(), 1.);
   std::vector<int64_t> indices = { 2, 0, 0, 1};
   std::vector<float> correct_output = {9,10,11,12, 1,2,3,4, 13,14,15,16, 17,18,19,20};

   ASSERT_RUN(std::vector<float>, GatherND_3, input, indices);

   expectEqual(output, correct_output);
}

TEST(ONNX, NonZero)
{
   std::vector<uint8_t> input = {0,1,0, 1,1,0, 0,0,1, 0,1,1 };
   std::vector<int64_t> correct_output = { 0,0,0,1,1,1 ,   0,1,1,0,1,1 ,    1,0,1,2,1,2 };

   ASSERT_RUN(std::vector<int64_t>, NonZero, input);

   expectEqual(output, correct_output);
}

TEST(ONNX, NonZero_Constant)
{
   std::vector<int64_t> correct_output = { 0,0,0,1,1,1 ,   0,1,1,0,1,1 ,    1,0,1,2,1,2 };

   ASSERT_RUN_0(std::vector<int64_t>, NonZero_Constant);

   expectEqual(output, correct_output);
}
TEST(ONNX, IsInf)
{
   std::vector<float> input = { 1, static_cast<float>(1./0.), 2.};
   std::vector<uint8_t> correct_output = { 0,1,0 };

   ASSERT_RUN_SESSION_ARGS(std::vector<uint8_t>, IsInf, ("", input.size()), input.size(),input);

   expectEqual(output, correct_output);
}

TEST(ONNX, NotIsNaN)
{
   std::vector<float> input = { 1, static_cast<float>(0./0.), 2.};
   std::vector<uint8_t> correct_output = { 1,0,1 };

   ASSERT_RUN_SESSION_ARGS(std::vector<uint8_t>, NotIsNaN, ("", input.size()), input.size(),input);

   expectEqual(output, correct_output);
}

TEST(ONNX, ScatterND_1)
{
   std::vector<float> input = {1.,2.,3.,4.,5.};
   std::vector<int64_t> indices = { 0, 2, 4};
   std::vector<float> updates = { 10.,30.,50.};
   std::vector<float> correct_output = {10., 2., 30., 4., 50.};

   ASSERT_RUN(std::vector<float>, ScatterND_1, input, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ScatterND_2)
{
   std::vector<float> input = {1.,1.,2.,2.,3.,3.};
   std::vector<int64_t> indices = { 0, 1};
   std::vector<float> updates = { 10.,10.,20.,20.};
   std::vector<float> correct_output = {11., 11., 22., 22., 3., 3.};

   ASSERT_RUN(std::vector<float>, ScatterND_2, input, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ScatterND_3)
{
   std::vector<float> input = {1.,2.,3.,4.};
   std::vector<int64_t> indices = { 0,0, 1,1};
   std::vector<float> updates = { 11.,22.};
   std::vector<float> correct_output = {11., 2., 3., 88.};

   ASSERT_RUN(std::vector<float>, ScatterND_3, input, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, Clip)
{
   std::vector<float> input = {-2.0,  0.5, 1.5, -0.3, 0.0,  3.0, -1.5,  0.8};
   std::vector<float> correct_output1 = {-1, 0.5, 1., -0.3, 0., 1.0, -1, 0.8};
   std::vector<float> correct_output2 = {-1, 0.5, 1.5, -0.3, 0., 3.0, -1, 0.8};

   ASSERT_RUN_SESSION_ARGS(std::vector<std::vector<float>>, Clip, ("Clip_FromONNX.dat", 2), 2, input);

   ASSERT_EQ(output.size(), 2u);
   expectNear(output[0], correct_output1, DEFAULT_TOLERANCE);
   expectNear(output[1], correct_output2, DEFAULT_TOLERANCE);
}

TEST(ONNX, Gelu)
{
   SofieReference ref = readReference("Gelu");

   ASSERT_RUN(std::vector<float>, Gelu, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, Swish)
{
   SofieReference ref = readReference("Swish");

   ASSERT_RUN(std::vector<float>, Swish, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, HardSigmoid)
{
   SofieReference ref = readReference("HardSigmoid");

   ASSERT_RUN(std::vector<float>, HardSigmoid, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, HardSwish)
{
   SofieReference ref = readReference("HardSwish");

   ASSERT_RUN(std::vector<float>, HardSwish, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ComparisonBroadcast)
{
   std::vector<float> input_A = {0.0f, 1.0f, 2.0f, 3.0f};

   std::vector<float> input_B = {4.0f, 4.0f, 2.0f, 2.0f};

   std::vector<uint8_t> expected_output_less = {1, 1, 0, 0};

   ASSERT_RUN(std::vector<std::vector<uint8_t>>, Comparison_broadcast, input_A, input_B);

   ASSERT_EQ(output.size(), 3u);
   const std::vector<uint8_t> &output_less = output[2];

   expectEqual(output_less, expected_output_less);
}

TEST(ONNX, ComparisonBroadcast3d)
{
   std::vector<float> input_A = {1.0f, 6.0f, 2.0f, 9.0f, 0.0f, 5.0f, 3.0f, 1.0f,
                                 2.0f, 4.0f, 4.0f, 2.0f, 1.0f, 7.0f, 0.0f, 3.0f};

   std::vector<float> input_B = {1.0f, 5.0f, 3.0f, 2.0f};

   std::vector<uint8_t> expected_greater = {0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1};

   std::vector<uint8_t> expected_equal = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0};

   std::vector<uint8_t> expected_less = {0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0};

   ASSERT_RUN(std::vector<std::vector<uint8_t>>, Comparison_broadcast_3d, input_A, input_B);

   ASSERT_EQ(output.size(), 3);

   const std::vector<uint8_t> &output_greater = output[0];
   const std::vector<uint8_t> &output_equal = output[1];
   const std::vector<uint8_t> &output_less = output[2];

   ASSERT_EQ(output_greater, expected_greater);
   ASSERT_EQ(output_equal, expected_equal);
   ASSERT_EQ(output_less, expected_less);
}

TEST(ONNX, ConvSharedInput)
{
   SofieReference ref = readReference("ConvSharedInput");

   ASSERT_RUN(std::vector<float>, ConvSharedInput, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvTransposeSharedInput)
{
   SofieReference ref = readReference("ConvTransposeSharedInput");

   ASSERT_RUN(std::vector<float>, ConvTransposeSharedInput, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvResidualAdd)
{
   SofieReference ref = readReference("ConvResidualAdd");

   ASSERT_RUN(std::vector<float>, ConvResidualAdd, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, NonZeroTwice)
{
   std::vector<uint8_t> input = {0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1};
   std::vector<int64_t> correct_output = {0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1, 2};

   ASSERT_RUN(std::vector<std::vector<int64_t>>, NonZeroTwice, input);

   ASSERT_EQ(output.size(), 2u);
   expectEqual(output[0], correct_output);
   expectEqual(output[1], correct_output);
}

TEST(ONNX, GatherNDNegativeIndicesTwice)
{
   SofieReference ref = readReference("GatherNDNegativeIndicesTwice");

   ASSERT_RUN(std::vector<float>, GatherNDNegativeIndicesTwice, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, IdentityWeightBatchNorm)
{
   SofieReference ref = readReference("IdentityWeightBatchNorm");

   ASSERT_RUN(std::vector<float>, IdentityWeightBatchNorm, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, BatchNormEpsilon)
{
   SofieReference ref = readReference("BatchNormEpsilon");

   ASSERT_RUN(std::vector<float>, BatchNormEpsilon, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, BatchNormReluEpsilon)
{
   SofieReference ref = readReference("BatchNormReluEpsilon");

   ASSERT_RUN(std::vector<float>, BatchNormReluEpsilon, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ClipInt)
{
   std::vector<int> input = {-5, -2, 0, 3, 5, 9};
   std::vector<int> correct_output = {-2, -2, 0, 3, 5, 5};

   ASSERT_RUN(std::vector<int>, ClipInt, input);

   expectEqual(output, correct_output);
}

TEST(ONNX, TanhDynShape)
{
   SofieReference ref = readReference("TanhDynShape");

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, TanhDynShape, ("TanhDynShape_FromONNX.dat", 2), 2,
                                       ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RangeCleanName)
{
   std::vector<float> start = {0.0};
   std::vector<float> limit = {5.0};
   std::vector<float> delta = {1.0};
   std::vector<float> correct_output = {0.0, 1.0, 2.0, 3.0, 4.0};

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, RangeCleanName, ("", 5), start, limit, delta);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, GemmDynBias)
{
   SofieReference ref = readReference("GemmDynBias");

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, GemmDynBias, ("GemmDynBias_FromONNX.dat", 2), 2,
                                       ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, IdentityWeightOutput)
{
   SofieReference ref = readReference("IdentityWeightOutput");

   ASSERT_RUN(std::vector<std::vector<float>>, IdentityWeightOutput, ref.f32("input0"));

   ASSERT_EQ(output.size(), 2u);
   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ReshapeAlias)
{
   SofieReference ref = readReference("ReshapeAlias");

   ASSERT_RUN(std::vector<float>, ReshapeAlias, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ReshapeAliasGraphOutput)
{
   SofieReference ref = readReference("ReshapeAliasGraphOutput");

   ASSERT_RUN(std::vector<float>, ReshapeAliasGraphOutput, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, SliceIdentityAlias)
{
   SofieReference ref = readReference("SliceIdentityAlias");

   ASSERT_RUN(std::vector<float>, SliceIdentityAlias, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, IdentityAlias)
{
   SofieReference ref = readReference("IdentityAlias");

   ASSERT_RUN(std::vector<float>, IdentityAlias, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AliasAcrossNewTensor)
{
   SofieReference ref = readReference("AliasAcrossNewTensor");

   ASSERT_RUN(std::vector<float>, AliasAcrossNewTensor, ref.f32("input0"), ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AliasOwnerReadAfterAlias)
{
   SofieReference ref = readReference("AliasOwnerReadAfterAlias");

   ASSERT_RUN(std::vector<std::vector<float>>, AliasOwnerReadAfterAlias, ref.f32("input0"),
                          ref.f32("input1"));

   ASSERT_EQ(output.size(), 2u);
   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AliasChain)
{
   SofieReference ref = readReference("AliasChain");

   ASSERT_RUN(std::vector<float>, AliasChain, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AliasChainSingle)
{
   SofieReference ref = readReference("AliasChainSingle");

   ASSERT_RUN(std::vector<float>, AliasChainSingle, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AliasDynShape)
{
   SofieReference ref = readReference("AliasDynShape");

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, AliasDynShape, ("AliasDynShape_FromONNX.dat", 2), 2,
                                       ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, AliasDynShapeAcrossNewTensor)
{
   SofieReference ref = readReference("AliasDynShapeAcrossNewTensor");

   ASSERT_RUN_SESSION_ARGS(std::vector<float>, AliasDynShapeAcrossNewTensor, ("AliasDynShapeAcrossNewTensor_FromONNX.dat", 2), 2, ref.f32("input0"),
                                       ref.f32("input1"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ScatterND_Ex1)
{
   std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
   std::vector<int64_t> indices = {4, 3, 1, 7};
   std::vector<float> updates = {9, 10, 11, 12};
   std::vector<float> correct_output = {1, 11, 3, 10, 9, 6, 7, 12};

   ASSERT_RUN(std::vector<float>, ScatterND_Ex1, data, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ScatterND_Ex2)
{
   std::vector<float> data(64, 0.f);
   std::vector<int64_t> indices = {0, 2};
   std::vector<float> updates = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};

   std::vector<float> correct_output(64, 0.f);
   for (int j = 0; j < 16; ++j)
      correct_output[j] = updates[j];
   for (int j = 0; j < 16; ++j)
      correct_output[32 + j] = updates[16 + j];

   ASSERT_RUN(std::vector<float>, ScatterND_Ex2, data, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ScatterND_NegativeIndices)
{
   std::vector<float> data = {0, 0, 0, 0, 0};
   std::vector<int64_t> indices = {-1, -3};
   std::vector<float> updates = {99, 88};
   std::vector<float> correct_output = {0, 0, 88, 0, 99};

   ASSERT_RUN(std::vector<float>, ScatterND_NegativeIndices, data, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ScatterND_2D)
{
   std::vector<float> data = {0, 0, 0, 0, 0, 0, 0, 0, 0};
   std::vector<int64_t> indices = {0, 2, 1, 0, 2, 1};
   std::vector<float> updates = {5, 6, 7};
   std::vector<float> correct_output = {0, 0, 5, 6, 0, 0, 0, 7, 0};

   ASSERT_RUN(std::vector<float>, ScatterND_2D, data, indices, updates);

   expectNear(output, correct_output, DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithPadding_NoSession)
{
   SofieReference ref = readReference("ConvWithPadding");

   ASSERT_RUN_NO_SESSION(std::vector<float>, ConvWithPadding_NoSession, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithoutPadding_NoSession)
{
   SofieReference ref = readReference("ConvWithoutPadding");

   ASSERT_RUN_NO_SESSION(std::vector<float>, ConvWithoutPadding_NoSession, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithAutopadSameLower_NoSession)
{
   SofieReference ref = readReference("ConvWithAutopadSameLower");

   ASSERT_RUN_NO_SESSION(std::vector<float>, ConvWithAutopadSameLower_NoSession, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithStridesPadding_NoSession)
{
   SofieReference ref = readReference("ConvWithStridesPadding");

   ASSERT_RUN_NO_SESSION(std::vector<float>, ConvWithStridesPadding_NoSession, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, ConvWithStridesNoPadding_NoSession)
{
   SofieReference ref = readReference("ConvWithStridesNoPadding");

   ASSERT_RUN_NO_SESSION(std::vector<float>, ConvWithStridesNoPadding_NoSession, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNBatchwise_NoSession)
{
   SofieReference ref = readReference("RNNBatchwise");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, RNNBatchwise_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNBidirectional_NoSession)
{
   SofieReference ref = readReference("RNNBidirectional");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, RNNBidirectional_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNBidirectionalBatchwise_NoSession)
{
   SofieReference ref = readReference("RNNBidirectionalBatchwise");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, RNNBidirectionalBatchwise_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNDefaults_NoSession)
{
   SofieReference ref = readReference("RNNDefaults");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, RNNDefaults_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNSeqLength_NoSession)
{
   SofieReference ref = readReference("RNNSeqLength");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, RNNSeqLength_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNSequence_NoSession)
{
   SofieReference ref = readReference("RNNSequence");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, RNNSequence_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, RNNSequenceBatchwise_NoSession)
{
   SofieReference ref = readReference("RNNSequenceBatchwise");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, RNNSequenceBatchwise_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMBatchwise_NoSession)
{
   SofieReference ref = readReference("LSTMBatchwise");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, LSTMBatchwise_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMBidirectional_NoSession)
{
   SofieReference ref = readReference("LSTMBidirectional");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, LSTMBidirectional_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
   expectNear(output[2], ref.f32("output2"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMDefaults_NoSession)
{
   SofieReference ref = readReference("LSTMDefaults");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, LSTMDefaults_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMInitialBias_NoSession)
{
   SofieReference ref = readReference("LSTMInitialBias");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, LSTMInitialBias_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LSTMPeepholes_NoSession)
{
   SofieReference ref = readReference("LSTMPeepholes");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, LSTMPeepholes_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUBatchwise_NoSession)
{
   SofieReference ref = readReference("GRUBatchwise");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, GRUBatchwise_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUBidirectional_NoSession)
{
   SofieReference ref = readReference("GRUBidirectional");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, GRUBidirectional_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUDefaults_NoSession)
{
   SofieReference ref = readReference("GRUDefaults");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, GRUDefaults_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUInitialBias_NoSession)
{
   SofieReference ref = readReference("GRUInitialBias");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, GRUInitialBias_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, GRUSeqLength_NoSession)
{
   SofieReference ref = readReference("GRUSeqLength");

   ASSERT_RUN_NO_SESSION(std::vector<std::vector<float>>, GRUSeqLength_NoSession, ref.f32("input0"));

   expectNear(output[0], ref.f32("output0"), DEFAULT_TOLERANCE);
   expectNear(output[1], ref.f32("output1"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LinearWithSelu_NoWeightFile)
{
   SofieReference ref = readReference("LinearWithSelu");

   ASSERT_RUN(std::vector<float>, LinearWithSelu_NoWeightFile, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}

TEST(ONNX, LinearWithSigmoid_NoWeightFile)
{
   SofieReference ref = readReference("LinearWithSigmoid");

   ASSERT_RUN(std::vector<float>, LinearWithSigmoid_NoWeightFile, ref.f32("input0"));

   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);
}
