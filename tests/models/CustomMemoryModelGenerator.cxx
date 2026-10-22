#include "SOFIE/RModel.hxx"
#include "SOFIE/operators/ROperator_BasicBinary.hxx"
#include "SOFIE/operators/ROperator_BasicUnary.hxx"
#include "SOFIE/operators/ROperator_Gemm.hxx"
#include "SOFIE/operators/ROperator_Relu.hxx"
#include "SOFIE/operators/ROperator_Sigmoid.hxx"
#include "SOFIE/operators/ROperator_SubGraph.hxx"
#include "SOFIE/operators/ROperator_Tanh.hxx"

using namespace SOFIE;

namespace {

RModel BuildChain(const std::string &name)
{
   RModel model(name, "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("X");
   std::vector<float> W(5 * 4, 0.5f);
   std::vector<float> B(4, 1.0f);
   model.AddInitializedTensor("W", ETensorType::FLOAT, std::vector<std::size_t>{5, 4}, W.data());
   model.AddInitializedTensor("B", ETensorType::FLOAT, std::vector<std::size_t>{4}, B.data());
   model.AddOperator(std::make_unique<ROperator_Gemm<float>>(1.0, 1.0, 0, 0, "X", "W", "B", "H1"));
   model.AddOperator(std::make_unique<ROperator_Relu<float>>("H1", "H2"));
   model.AddOperator(std::make_unique<ROperator_Sigmoid<float>>("H2", "H3"));
   model.AddOperator(std::make_unique<ROperator_Tanh<float>>("H3", "Y"));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

RModel BuildIf()
{
   auto thenGraph = std::make_unique<RModel>("then_branch", "now");
   thenGraph->AddOperator(std::make_unique<ROperator_BasicBinary<float, EBasicBinaryOperator::Mul>>("X", "X", "then_out"));
   thenGraph->AddOutputTensorNameList({"then_out"});

   auto elseGraph = std::make_unique<RModel>("else_branch", "now");
   elseGraph->AddOperator(std::make_unique<ROperator_BasicUnary<float, EBasicUnaryOperator::kNeg>>("X", "else_neg"));
   elseGraph->AddOperator(std::make_unique<ROperator_Sigmoid<float>>("else_neg", "else_out"));
   elseGraph->AddOutputTensorNameList({"else_out"});

   RModel model("CustomMemoryIf", "now");
   model.AddInputTensorInfo("cond", ETensorType::BOOL, std::vector<Dim>{Dim(1)});
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("cond");
   model.AddInputTensorName("X");
   model.AddOperator(std::make_unique<ROperator_If>("cond", std::vector<std::string>{"Y"}, std::move(thenGraph),
                                                    std::move(elseGraph)));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

void Emit(RModel model, int options, const std::string &name)
{
   model.GenerateGPU_ALPAKA(options, -1, false);
   model.OutputGenerated(name + "_GPU_ALPAKA.hxx");
}

} // namespace

int main()
{
   Emit(BuildChain("CustomMemoryChain"), static_cast<int>(Options::kCustomMemory), "CustomMemoryChain");
   Emit(BuildChain("CustomMemoryStreamChain"),
        static_cast<int>(Options::kCustomMemory) | static_cast<int>(Options::kExternalStream), "CustomMemoryStreamChain");
   Emit(BuildIf(), static_cast<int>(Options::kCustomMemory), "CustomMemoryIf");
   return 0;
}
