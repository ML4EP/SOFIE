// Operators without an Alpaka implementation (they do not derive from ROperator_ALPAKA) are rejected by the generation
// for the GPU, with an error naming the operator, instead of generating a model which silently skips them. The
// generation only needs SOFIE_core: the Alpaka code is generated but not compiled here.

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_BasicUnary.hxx"
#include "SOFIE/ROperator_Erf.hxx"
#include "SOFIE/ROperator_Relu.hxx"
#include "SOFIE/ROperator_SubGraph.hxx"

#include <gtest/gtest.h>

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
   EXPECT_NE(error.find("has no Alpaka implementation"), std::string::npos) << error;
   EXPECT_NE(error.find("ROperator_Erf"), std::string::npos) << error;
   EXPECT_NE(error.find("AlpakaErf"), std::string::npos) << error;
}

// the CPU generation of the same model is not affected
TEST(AlpakaOperatorSupport, OperatorWithoutAlpakaImplementationIsGeneratedForTheCpu)
{
   auto model = BuildModel("CpuErf", std::make_unique<ROperator_Erf<float>>("X", "Y"));
   EXPECT_NO_THROW(model.Generate(Options::kNoWeightFile));
}

// the operators of the branches of an If are checked too
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
   EXPECT_NE(error.find("has no Alpaka implementation"), std::string::npos) << error;
   EXPECT_NE(error.find("ROperator_Erf"), std::string::npos) << error;
}
