#include "SOFIE/RFloat16.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_Reduce.hxx"
#include "SOFIE/ROperator_Softmax.hxx"
#include <gtest/gtest.h>
#include <cmath>
#include <array>

TEST(SOFIEFloat16, RepresentationAndRounding)
{
   using SOFIE::Float16;
   for (std::uint32_t bits = 0; bits <= 65535; ++bits) {
      const auto value = Float16::FromBits(static_cast<std::uint16_t>(bits));
      const float expanded = value;
      if ((bits & 0x7c00) == 0x7c00 && (bits & 1023))
         EXPECT_TRUE(std::isnan(expanded));
      else
         EXPECT_EQ(Float16(expanded).bits, bits);
   }
   EXPECT_EQ(Float16(1.f + std::ldexp(1.f, -11)).bits, 0x3c00);
   EXPECT_EQ(Float16(1.f + 3 * std::ldexp(1.f, -11)).bits, 0x3c02);
   EXPECT_EQ(Float16(std::ldexp(1.f, -25)).bits, 0);
   EXPECT_EQ(Float16(65520.f).bits, 0x7c00);
   EXPECT_EQ(Float16(-0.f).bits, 0x8000);
   EXPECT_EQ(sizeof(Float16), 2);
   EXPECT_EQ(SOFIE::GetTypeSize(SOFIE::ETensorType::FLOAT16), 2);
}

TEST(SOFIEFloat16, GeneratesTypedAlpakaAndAllocationFreeReductions)
{
   using namespace SOFIE;
   RModel model("half_pipeline", "test\n");
   model.SetExternalWorkspace();
   model.AddInputTensorInfo("input", ETensorType::FLOAT16, std::vector<Dim>{Dim("batch"), Dim(3)});
   model.AddInputTensorName("input");
   std::array<Float16, 9> weights{1, 0, 0, 0, 1, 0, 0, 0, 1};
   std::array<Float16, 3> bias{1, -2, 3};
   model.AddInitializedTensor("weights", ETensorType::FLOAT16, {3, 3}, weights.data());
   model.AddInitializedTensor("bias", ETensorType::FLOAT16, {3}, bias.data());
   model.AddOperator(std::make_unique<ROperator_Gemm<Float16>>(0.5f, 2.f, 0, 1,
      "input", "weights", "bias", "hidden", EActivationType::RELU));
   model.AddOperator(std::make_unique<ROperator_Softmax>(-1, "hidden", "probabilities"));
   model.AddOperator(std::make_unique<ROperator_Reduce<ReduceSum>>(1,
      std::vector<int64_t>{1}, "probabilities", "", "output"));
   model.AddOutputTensorNameList({"output"});
   model.GenerateGPU_ALPAKA(Options::kNoWeightFile);
   const auto source = model.ReturnGenerated();
   EXPECT_NE(source.find("ViewConstH1D"), std::string::npos);
   EXPECT_NE(source.find("SOFIE::Float16::FromBits("), std::string::npos);
   EXPECT_NE(source.find("nativeBlas.Gemm"), std::string::npos);
   EXPECT_NE(source.find("static_cast<const SOFIE::Float16*>(alpaka::getPtrNative(deviceBuf_bias))"),
             std::string::npos);
   EXPECT_EQ(source.find("static_cast<const float*>(alpaka::getPtrNative(deviceBuf_bias))"),
             std::string::npos);
   EXPECT_NE(source.find("SOFIE::AccumulationType<T>"), std::string::npos);
   EXPECT_EQ(source.find("reduceScratch_"), std::string::npos);
   EXPECT_EQ(source.find("blas.addOperationConfig"), std::string::npos);
}
