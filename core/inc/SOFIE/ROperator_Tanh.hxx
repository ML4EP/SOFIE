#ifndef SOFIE_ROPERATOR_Tanh
#define SOFIE_ROPERATOR_Tanh

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename T>
class ROperator_Tanh final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_Tanh(){}
   ROperator_Tanh(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fKind = OperatorKind::TANH;
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("SOFIE Tanh Op Input Tensor is not found in model");
      }
      fShape = model.GetDimTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShape.empty();
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);

   }

   bool SupportsStridedInput() const override { return true; }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Tanh operator called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShape);
      out << "\n//------ TANH\n";
      if (fHasStridedInput) {
         out << GenerateStridedUnaryLoop(OpName, fNX, fNY, fShape,
                                         [](const std::string &v) { return "std::tanh(" + v + ")"; });
         return out.str();
      }
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = std::tanh(tensor_" << fNX << "[id]);\n";
      out << SP << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") };}

   bool IsElementwise() const override { return !fHasStridedInput; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return "tanh(" + v + ")";
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fHasStridedInput)
         return GenerateStridedUnaryKernel("TanhStridedKernel", "TANH",
                                           [this](const std::string &v) { return GetElementwiseExpr(v); });
      std::string op;
      op = "\n//------ TANH_KERNEL_ALPAKA\n";
      op += "struct TanhKernel {\n";
      op += SP + "template<typename TAcc, typename T>\n";
      op += SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements) const {\n";
      op += SP + SP + "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + "if (idx < numElements) { out[idx] = tanh(data[idx]); }\n";
      op += SP + "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      if (fHasStridedInput)
         return SP + "TanhStridedKernel tanhStridedKernel;\n";
      return SP + "TanhKernel tanhKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Tanh called to Generate_GPU_ALPAKA without being initialized");
      }
      if (fHasStridedInput)
         return GenerateStridedUnaryLaunch(OpName, "tanhStridedKernel", "TANH", fNX, fNY, fShape);

      std::stringstream out;
      std::string length = ConvertDimShapeToLength(fShape);
      out << "\n//------ TANH_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_"<<fNX<<" = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"<<fNX<<" = Vec::all(Idx{"<< length << "});\n";
      out << SP << "auto const workDiv_" << fNX << " = sofie_workdiv(elementsPerGrid_" << fNX << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNX
         << ", tanhKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }
};

}//SOFIE


#endif //SOFIE_ROPERATOR_Tanh
