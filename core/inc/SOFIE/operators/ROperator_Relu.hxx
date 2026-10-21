#ifndef SOFIE_ROPERATOR_RELU
#define SOFIE_ROPERATOR_RELU

#include "SOFIE/utils/SOFIE_common.hxx"
#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename T>
class ROperator_Relu final : public ROperator_ALPAKA
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_Relu(){}
   ROperator_Relu(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fKind = OperatorKind::RELU;
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE Relu Op Input Tensor " + fNX + " is not found in model");
      }

      fShape = model.GetDimTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShape.empty();

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      if (model.Verbose()) {
         std::cout << "Relu : " << fNX << " -> " << fNY << " " << ConvertDimShapeToString(fShape) << std::endl;
      }
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator Relu called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShape);
      out << "\n//------ RELU\n";
      if (fHasStridedInput) {
         out << GenerateStridedUnaryLoop(OpName, fNX, fNY, fShape,
                                         [](const std::string &v) { return "((" + v + " > 0 )? " + v + " : 0)"; });
         return out.str();
      }
      out << "for (int id = 0; id < " << length << " ; id++){\n";
      out << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] > 0 )? tensor_" << fNX << "[id] : 0);\n";
      out << "}\n";
      return out.str();
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) {
      if (fHasStridedInput)
         return GenerateStridedUnaryKernel("ReluStridedKernel", "RELU",
                                           [this](const std::string &v) { return GetElementwiseExpr(v); });
      std::string op;
      op = "\n//------ RELU_KERNEL_ALPAKA\n";

      op = "\n//------ RELU_KERNEL_ALPAKA\n";
      op += "struct ReluKernel {\n";
      op += "template<typename TAcc, typename T>\n";
      op += "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements) const {\n";
      op += "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += "if (idx < numElements) {\n";
      op += "out[idx] = data[idx] >= T(0) ? data[idx] : 0;\n";
      op += "}\n";
      op += "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      if (fHasStridedInput)
         return "ReluStridedKernel reluStridedKernel;\n";
      return "ReluKernel reluKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator Relu called to Generate without being initialized first");
      }

      if (fHasStridedInput)
         return GenerateStridedUnaryLaunch(OpName, "reluStridedKernel", "RELU", fNX, fNY, fShape);

      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShape);
      out << "\n//------ RELU_GPU_ALPAKA\n";
      out << "auto const elementsPerThread_"<<fNY<<" = Vec::all(static_cast<Idx>(1));\n";
      out << "auto const elementsPerGrid_"<<fNY<<" = Vec::all(Idx{"<< length << "});\n";
      out << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
         << ", reluKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "));\n";
      out << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }

   std::string GetFusableOutputTensorName() override {
         return fNY;
   }

   bool IsElementwise() const override { return !fHasStridedInput; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return "(" + v + ") >= T(0) ? (" + v + ") : T(0)";
   }

   void UpdateFusableTensorName(std::string fusable_tensor_name, const std::function<void(const std::string&)>& removal_func){
      removal_func(fNX);
      removal_func(fNY);
      fNX = fusable_tensor_name;
      fNY = fusable_tensor_name;
      fInputTensorNames[0] =  fNX;
      fOutputTensorNames[0] = fNY;
   }
};

}//SOFIE

#endif //SOFIE_ROPERATOR_RELU
