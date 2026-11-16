#ifndef SOFIE_ROPERATOR_GELU
#define SOFIE_ROPERATOR_GELU

#include "SOFIE/utils/SOFIE_common.hxx"
#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
namespace SOFIE{

class ROperator_Gelu final : public ROperator_ALPAKA
{

private:

   std::string fNX;
   std::string fNY;
   std::string fApproximate;
   std::vector<size_t> fShape;

public:
   ROperator_Gelu(){}
   ROperator_Gelu(std::string nameX, std::string nameY, std::string approximate = "none"):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)), fApproximate(approximate){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }


   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
         throw std::runtime_error("SOFIE Gelu Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShape.empty();
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }

   bool SupportsStridedInput() const override { return true; }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("SOFIE Operator Gelu called to Generate without being initialized first");
      }
      std::stringstream out;
      int length = 1;
      for(auto& i: fShape){
         length *= i;
      }
      if (fHasStridedInput) {
         const bool tanhApprox = fApproximate == "tanh";
         out << GenerateStridedUnaryLoop(OpName, fNX, fNY, ConvertShapeToDim(fShape), [&](const std::string &v) {
            if (tanhApprox)
               return "0.5f * " + v + " * (1.0f + std::tanh(0.7978845608028654f * (" + v + " + 0.044715f * " + v + " * " + v +
                      " * " + v + ")))";
            return "0.5f * " + v + " * (1.0f + std::erf(" + v + " * 0.7071067811865475f))";
         });
         return out.str();
      }
      out << "for (int id = 0; id < " << length << " ; id++){\n";
      if (fApproximate == "tanh") {
         out << "float x = tensor_" << fNX << "[id];\n";
         out << "tensor_" << fNY << "[id] = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));\n";
      } else {
         out << "tensor_" << fNY << "[id] = 0.5f * tensor_" << fNX << "[id] * (1.0f + std::erf(tensor_" << fNX << "[id] * 0.7071067811865475f));\n";
      }
      out << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") };}

   /* The output element as an expression of the input element `v`, in the form the
      approximate attribute selects: the exact erf form or the tanh approximation. The
      choice is made here, at generation time, so the kernel holds a single formula. */
   std::string GpuExpr(const std::string &v) const {
      if (fApproximate == "tanh")
         return "T(0.5) * " + v + " * (T(1) + tanh(T(0.7978845608028654) * (" + v + " + T(0.044715) * " + v + " * " +
                v + " * " + v + ")))";
      return "T(0.5) * " + v + " * (T(1) + erf(" + v + " * T(0.7071067811865475)))";
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fHasStridedInput)
         return GenerateStridedUnaryKernel("GeluStridedKernel", "GELU", [this](const std::string &v) { return GpuExpr(v); });
      std::string op;
      op = "\n//------ GELU_KERNEL_ALPAKA\n";
      op += "struct GeluKernel {\n";
      op += "template<typename TAcc, typename T>\n";
      op += "ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements) const {\n";
      op += "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += "if (idx < numElements) { T const x = data[idx]; out[idx] = " + GpuExpr("x") + "; }\n";
      op += "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      if (fHasStridedInput)
         return "GeluStridedKernel geluStridedKernel;\n";
      return "GeluKernel geluKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Gelu called to Generate_GPU_ALPAKA without being initialized");
      }
      std::stringstream out;
      std::string length = std::to_string(ConvertShapeToLength(fShape));
      out << "\n//------ GELU_GPU_ALPAKA\n";
      if (fHasStridedInput)
         return GenerateStridedUnaryLaunch(OpName, "geluStridedKernel", "GELU", fNX, fNY, ConvertShapeToDim(fShape));
      out << "auto const elementsPerGrid_" << fNX << " = Vec::all(Idx{" << length << "});\n";
      out << "auto const workDiv_" << fNX << " = sofie_workdiv(elementsPerGrid_" << fNX << ");\n";
      out << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNX
          << ", geluKernel, alpaka::getPtrNative(deviceBuf_" << fNX
          << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "));\n";
      out << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }
};

}


#endif
