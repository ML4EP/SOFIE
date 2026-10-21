// Operators without an alpaka implementation are rejected by the generation
// for the GPU, with an error naming the operator.

#include "SOFIE/model/RModel.hxx"
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

TEST(AlpakaOperatorSupport, TileFusionDataInputIndicesAreValidInputs)
{
   ROperator_Tile<float> tile("repeats", "X", "Y");
   const auto inputs = tile.GetOpInputTensors();
   const auto indices = tile.GetFusionDataInputIndices();
   ASSERT_FALSE(indices.empty());
   EXPECT_TRUE(std::all_of(indices.begin(), indices.end(), [&](size_t i) { return i < inputs.size(); }));
}
