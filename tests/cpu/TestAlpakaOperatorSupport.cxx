// Operators without an alpaka implementation are rejected by the generation
// for the GPU, with an error naming the operator.

#include "SOFIE/RModel.hxx"
#include "SOFIE/operators/ROperator_BasicUnary.hxx"
#include "SOFIE/operators/ROperator_Erf.hxx"
#include "SOFIE/operators/ROperator_Relu.hxx"
#include "SOFIE/operators/ROperator_SubGraph.hxx"
#include "SOFIE/operators/ROperator_Tile.hxx"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

using namespace SOFIE;

namespace {

RModel BuildModel(const std::string &name, std::unique_ptr<ROperator> op)
{
   RModel model(name, "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("X");
   model.AddOperator(std::move(op));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

// the message of the exception thrown by the GPU generation, empty if it succeeds
std::string GpuGenerationError(RModel &model)
{
   try {
      model.GenerateGPU_ALPAKA(Options::kNoWeightFile);
   } catch (const std::exception &e) {
      return e.what();
   }
   return "";
}

} // namespace

TEST(AlpakaOperatorSupport, OperatorWithAlpakaImplementationIsGenerated)
{
   auto model = BuildModel("AlpakaRelu", std::make_unique<ROperator_Relu<float>>("X", "Y"));
   EXPECT_EQ(GpuGenerationError(model), "");
   EXPECT_NE(model.ReturnGenerated().find("struct Session"), std::string::npos);
}

TEST(AlpakaOperatorSupport, OperatorWithoutAlpakaImplementationIsRejected)
{
   auto model = BuildModel("AlpakaErf", std::make_unique<ROperator_Erf<float>>("X", "Y"));
   const std::string error = GpuGenerationError(model);
   EXPECT_NE(error.find("has no alpaka implementation"), std::string::npos) << error;
   EXPECT_NE(error.find("ROperator_Erf"), std::string::npos) << error;
   EXPECT_NE(error.find("AlpakaErf"), std::string::npos) << error;
}

TEST(AlpakaOperatorSupport, OperatorWithoutAlpakaImplementationIsGeneratedForTheCpu)
{
   auto model = BuildModel("CpuErf", std::make_unique<ROperator_Erf<float>>("X", "Y"));
   EXPECT_NO_THROW(model.Generate(Options::kNoWeightFile));
}

// the operators of the branches of an If are checked
TEST(AlpakaOperatorSupport, OperatorOfASubGraphIsChecked)
{
   auto thenGraph = std::make_unique<RModel>("then_branch", "now");
   thenGraph->AddOperator(std::make_unique<ROperator_Erf<float>>("X", "then_out"));
   thenGraph->AddOutputTensorNameList({"then_out"});
   auto elseGraph = std::make_unique<RModel>("else_branch", "now");
   elseGraph->AddOperator(std::make_unique<ROperator_Relu<float>>("X", "else_out"));
   elseGraph->AddOutputTensorNameList({"else_out"});

   RModel model("AlpakaIfErf", "now");
   model.AddInputTensorInfo("cond", ETensorType::BOOL, std::vector<Dim>{Dim(1)});
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("cond");
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_If>("cond", std::vector<std::string>{"Y"}, std::move(thenGraph),
                                                    std::move(elseGraph)));
   model.AddOutputTensorNameList({"Y"});

   const std::string error = GpuGenerationError(model);
   EXPECT_NE(error.find("has no alpaka implementation"), std::string::npos) << error;
   EXPECT_NE(error.find("ROperator_Erf"), std::string::npos) << error;
}

TEST(AlpakaOperatorSupport, CheckGPUOperatorsAcceptsSupportedModel)
{
   auto model = BuildModel("CheckRelu", std::make_unique<ROperator_Relu<float>>("X", "Y"));
   testing::internal::CaptureStdout();
   EXPECT_TRUE(model.CheckGPUOperators());
   EXPECT_NE(testing::internal::GetCapturedStdout().find("All operators"), std::string::npos);
}

TEST(AlpakaOperatorSupport, CheckGPUOperatorsListsAllMissingOperators)
{
   RModel model("CheckTwoErf", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_Erf<float>>("X", "H"));
   model.AddOperator(std::make_unique<ROperator_Relu<float>>("H", "R"));
   model.AddOperator(std::make_unique<ROperator_Erf<float>>("R", "Y"));
   model.AddOutputTensorNameList({"Y"});
   testing::internal::CaptureStdout();
   EXPECT_FALSE(model.CheckGPUOperators());
   const std::string report = testing::internal::GetCapturedStdout();
   EXPECT_NE(report.find("operator 0"), std::string::npos) << report;
   EXPECT_EQ(report.find("operator 1"), std::string::npos) << report;
   EXPECT_NE(report.find("operator 2"), std::string::npos) << report;
   EXPECT_NE(report.find("ROperator_Erf"), std::string::npos) << report;
}

TEST(AlpakaOperatorSupport, CheckGPUOperatorsLooksIntoSubGraphs)
{
   auto thenGraph = std::make_unique<RModel>("then_branch", "now");
   thenGraph->AddOperator(std::make_unique<ROperator_Erf<float>>("X", "then_out"));
   thenGraph->AddOutputTensorNameList({"then_out"});
   auto elseGraph = std::make_unique<RModel>("else_branch", "now");
   elseGraph->AddOperator(std::make_unique<ROperator_Relu<float>>("X", "else_out"));
   elseGraph->AddOutputTensorNameList({"else_out"});

   RModel model("CheckIfErf", "now");
   model.AddInputTensorInfo("cond", ETensorType::BOOL, std::vector<Dim>{Dim(1)});
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("cond");
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_If>("cond", std::vector<std::string>{"Y"}, std::move(thenGraph),
                                                    std::move(elseGraph)));
   model.AddOutputTensorNameList({"Y"});
   testing::internal::CaptureStdout();
   EXPECT_FALSE(model.CheckGPUOperators());
   const std::string report = testing::internal::GetCapturedStdout();
   EXPECT_NE(report.find("then_branch"), std::string::npos) << report;
}

TEST(AlpakaOperatorSupport, DebugModeChecksTheGeneratedCode)
{
   auto model = BuildModel("DebugRelu", std::make_unique<ROperator_Relu<float>>("X", "Y"));
   EXPECT_NO_THROW(model.GenerateGPU_ALPAKA(Options::kNoWeightFile | Options::kCustomMemory | Options::kDebug));
   EXPECT_NO_THROW(model.CheckNoDeviceAllocation_GPU_ALPAKA());
   const std::string generated = model.ReturnGenerated();
   EXPECT_NE(generated.find("GetMemoryReport"), std::string::npos);
   EXPECT_NE(generated.find("ValidateMemoryRegions"), std::string::npos);
   EXPECT_NE(model.GetDeviceRegionsReport_GPU_ALPAKA().find("DebugRelu"), std::string::npos);
}

TEST(AlpakaOperatorSupport, DebugModeTracesTheGeneration)
{
   auto model = BuildModel("TraceRelu", std::make_unique<ROperator_Relu<float>>("X", "Y"));
   testing::internal::CaptureStdout();
   model.GenerateGPU_ALPAKA(Options::kNoWeightFile | Options::kCustomMemory | Options::kDebug);
   const std::string trace = testing::internal::GetCapturedStdout();
   for (const char *expected : {"[SOFIE generator] TraceRelu: generating the GPU (alpaka) code", "initializing operator 0",
                                "model initialized", "all the operators have an alpaka implementation", "fusion:",
                                "pooled tensor Y", "generating its kernel", "init code", "launch code",
                                "kernel objects of the session", "session code generated"})
      EXPECT_NE(trace.find(expected), std::string::npos) << expected << "\n" << trace;
}

TEST(AlpakaOperatorSupport, DebugModeIsIndependentOfTheOtherOptions)
{
   struct Case {
      const char *name;
      std::underlying_type_t<Options> options;
      bool session;
   };
   const auto dbg = static_cast<std::underlying_type_t<Options>>(Options::kDebug);
   const auto noWeights = static_cast<std::underlying_type_t<Options>>(Options::kNoWeightFile);
   auto opt = [](Options o) { return static_cast<std::underlying_type_t<Options>>(o); };
   const std::vector<Case> cases = {
      {"alone", dbg | noWeights, true},
      {"nosession", dbg | noWeights | opt(Options::kNoSession), false},
      {"kernelonly", dbg | opt(Options::kKernelOnly), false},
      {"profile", dbg | noWeights | opt(Options::kProfile), true},
      {"stream", dbg | noWeights | opt(Options::kExternalStream), true},
      {"custom", dbg | noWeights | opt(Options::kCustomMemory), true},
      {"strided", dbg | noWeights | opt(Options::kStridedInput), true},
      {"all", dbg | noWeights | opt(Options::kExternalStream) | opt(Options::kCustomMemory) | opt(Options::kStridedInput), true},
   };
   for (const auto &c : cases) {
      auto model = BuildModel(std::string("DebugCombo_") + c.name, std::make_unique<ROperator_Relu<float>>("X", "Y"));
      testing::internal::CaptureStdout();
      EXPECT_NO_THROW(model.GenerateGPU_ALPAKA(c.options)) << c.name;
      const std::string trace = testing::internal::GetCapturedStdout();
      EXPECT_NE(trace.find("[SOFIE generator]"), std::string::npos) << c.name;
      const std::string generated = model.ReturnGenerated();
      EXPECT_NE(generated.find("DebugLog"), std::string::npos) << c.name;
      EXPECT_EQ(generated.find("GetMemoryReport") != std::string::npos, c.session) << c.name;
   }
}

TEST(AlpakaOperatorSupport, NoTraceWithoutDebugMode)
{
   auto model = BuildModel("QuietRelu", std::make_unique<ROperator_Relu<float>>("X", "Y"));
   testing::internal::CaptureStdout();
   model.GenerateGPU_ALPAKA(Options::kNoWeightFile);
   EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
   EXPECT_EQ(model.ReturnGenerated().find("DebugLog"), std::string::npos);
}

TEST(AlpakaOperatorSupport, DebugMethodsAreNotGeneratedByDefault)
{
   auto model = BuildModel("PlainRelu", std::make_unique<ROperator_Relu<float>>("X", "Y"));
   model.GenerateGPU_ALPAKA(Options::kNoWeightFile | Options::kCustomMemory);
   const std::string generated = model.ReturnGenerated();
   EXPECT_EQ(generated.find("GetMemoryReport"), std::string::npos);
   EXPECT_EQ(generated.find("MemoryRegionInfo"), std::string::npos);
}

TEST(AlpakaOperatorSupport, GetDeviceRegionsReportIsOnlyForDebugMode)
{
   auto model = BuildModel("NoDebugRegions", std::make_unique<ROperator_Relu<float>>("X", "Y"));
   model.GenerateGPU_ALPAKA(Options::kNoWeightFile | Options::kCustomMemory);
   EXPECT_THROW(model.GetDeviceRegionsReport_GPU_ALPAKA(), std::runtime_error);
   auto debugModel = BuildModel("DebugRegions", std::make_unique<ROperator_Relu<float>>("X", "Y"));
   debugModel.GenerateGPU_ALPAKA(Options::kNoWeightFile | Options::kDebug);
   EXPECT_NO_THROW(debugModel.GetDeviceRegionsReport_GPU_ALPAKA());
}

TEST(AlpakaOperatorSupport, CheckNoDeviceAllocationIsOnlyForDebugMode)
{
   const std::vector<std::underlying_type_t<Options>> optionSets = {static_cast<std::underlying_type_t<Options>>(Options::kNoWeightFile),
                                                                    Options::kNoWeightFile | Options::kCustomMemory};
   for (auto options : optionSets) {
      auto model = BuildModel("NoDebugRelu", std::make_unique<ROperator_Relu<float>>("X", "Y"));
      model.GenerateGPU_ALPAKA(options);
      try {
         model.CheckNoDeviceAllocation_GPU_ALPAKA();
         FAIL() << "expected an exception";
      } catch (const std::runtime_error &e) {
         EXPECT_NE(std::string(e.what()).find("kDebug"), std::string::npos) << e.what();
      }
   }
}

TEST(AlpakaOperatorSupport, CheckNoDeviceAllocationIsOnlyForCustomMemory)
{
   auto model = BuildModel("PlainReluDefault", std::make_unique<ROperator_Relu<float>>("X", "Y"));
   model.GenerateGPU_ALPAKA(Options::kNoWeightFile | Options::kDebug);
   try {
      model.CheckNoDeviceAllocation_GPU_ALPAKA();
      FAIL() << "expected an exception";
   } catch (const std::runtime_error &e) {
      EXPECT_NE(std::string(e.what()).find("kCustomMemory"), std::string::npos) << e.what();
   }
}

TEST(AlpakaOperatorSupport, TileFusionDataInputIndicesAreValidInputs)
{
   ROperator_Tile<float> tile("repeats", "X", "Y");
   const auto inputs = tile.GetOpInputTensors();
   const auto indices = tile.GetFusionDataInputIndices();
   ASSERT_FALSE(indices.empty());
   EXPECT_TRUE(std::all_of(indices.begin(), indices.end(), [&](size_t i) { return i < inputs.size(); }));
}
