#include "SOFIE/RModel.hxx"
#include "SOFIE/operators/ROperator_Gemm.hxx"
#include "SOFIE/operators/ROperator_Relu.hxx"

using namespace SOFIE;

int main()
{
   RModel model("ExternalStreamGemmRelu", "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(3), Dim(5)});
   model.AddInputTensorName("X");
   std::vector<float> W(5 * 4, 0.5f);
   std::vector<float> B(4, 1.0f);
   model.AddInitializedTensor("W", ETensorType::FLOAT, std::vector<std::size_t>{5, 4}, W.data());
   model.AddInitializedTensor("B", ETensorType::FLOAT, std::vector<std::size_t>{4}, B.data());
   model.AddOperator(std::make_unique<ROperator_Gemm<float>>(1.0, 1.0, 0, 0, "X", "W", "B", "H"));
   model.AddOperator(std::make_unique<ROperator_Relu<float>>("H", "Y"));
   model.AddOutputTensorNameList({"Y"});
   model.GenerateGPU_ALPAKA(Options::kExternalStream, -1, false);
   model.OutputGenerated("ExternalStreamGemmRelu_GPU_ALPAKA.hxx");
   return 0;
}
