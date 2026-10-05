#ifndef SOFIE_ROPERATOR_Elu
#define SOFIE_ROPERATOR_Elu

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
namespace SOFIE {

template <typename T>
class ROperator_Elu final : public ROperator {

private:
   /* Attributes*/
   float falpha = 1.0;
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;
   std::string fType;

public:
   ROperator_Elu() {}
   ROperator_Elu(float alpha, std::string nameX, std::string nameY)
      : falpha(alpha), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::ELU;
      fInputTensorNames = { fNX };
      fOutputTensorNames = { fNY };
      
      if(std::is_same<T, float>::value){
         fType = "float";
      } else {
         throw std::runtime_error("SOFIE Encountered unsupported type parsing a Elu operator");
      }
   }

   void Initialize(RModel &model) override
   {
      if (model.CheckIfTensorAlreadyExist(fNX) ==
          false) {
         throw std::runtime_error("SOFIE Elu Op Input Tensor is not found in model");
      }
      fShape = model.GetDimTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShape.empty();
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      if (model.Verbose()) {
         std::cout << "Elu : " << fNX << " -> " << fNY << " " << ConvertDimShapeToString(fShape) << std::endl;
      }
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator Elu called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShape);

      out << SP << "float " << OpName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10)
          << falpha << ";\n";

      out << "\n//------ ELU \n";
      if (fHasStridedInput) {
         out << GenerateStridedUnaryLoop(OpName, fNX, fNY, fShape, [&](const std::string &v) {
            return "((" + v + " >= 0 )? " + v + " : " + OpName + "_alpha * (std::exp(" + v + ") - 1))";
         });
         return out.str();
      }
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] >= 0 )? tensor_" << fNX << "[id] : "<< OpName << "_alpha * (std::exp(tensor_"<< fNX<<"[id]) - 1));\n";
      out << SP << "}\n";
      return out.str();
   }
   
   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }

   // elu gpu kernel
   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fHasStridedInput)
         return GenerateStridedUnaryKernel(
            "EluStridedKernel", "ELU",
            [](const std::string &v) { return v + " >= T(0) ? " + v + " : alpha * (exp(" + v + ") - T(1))"; },
            ", T alpha");
      std::string op;
      op = "\n//------ ELU_KERNEL_ALPAKA\n";
      op += "struct EluKernel {\n";
      op += SP + "template<typename TAcc, typename T>\n";
      op += SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements, T alpha) const {\n";
      op += SP + SP + "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + "if (idx < numElements) { out[idx] = data[idx] >= T(0) ? data[idx]:alpha * (exp(data[idx]) - T(1)); }\n";
      op += SP + "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      if (fHasStridedInput)
         return SP + "EluStridedKernel eluStridedKernel;\n";
      return SP + "EluKernel eluKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Elu called to Generate_GPU_ALPAKA without being initialized");
      }
      std::stringstream out;
      std::string length = ConvertDimShapeToLength(fShape);
      out << "\n//------ ELU_GPU_ALPAKA\n";
      if (fHasStridedInput)
         return GenerateStridedUnaryLaunch(OpName, "eluStridedKernel", "ELU", fNX, fNY, fShape,
                                           ", static_cast<float>(" +
                                              [&] {
                                                 std::stringstream s;
                                                 s << std::setprecision(std::numeric_limits<float>::max_digits10) << falpha;
                                                 return s.str();
                                              }() + ")");
      out << SP << "auto const elementsPerThread_"<<fNX<<" = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"<<fNX<<" = Vec::all(Idx{"<< length << "});\n";
      out << SP << "auto const workDiv_" << fNX << " = sofie_workdiv(elementsPerGrid_" << fNX << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNX
         << ", eluKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "), static_cast<float>("
         << std::setprecision(std::numeric_limits<float>::max_digits10) << falpha << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }
};

}

#endif
