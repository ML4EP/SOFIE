#ifndef SOFIE_ROPERATOR_NOT
#define SOFIE_ROPERATOR_NOT

#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include <SOFIE/RModel.hxx>
#include <SOFIE/utils/SOFIE_common.hxx>
namespace SOFIE {


class ROperator_Not final : public ROperator_ALPAKA {
private:
   std::string fNX;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

public:
   ROperator_Not() {}

   ROperator_Not(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
         fKind = OperatorKind::NOT;
         fInputTensorNames =  { fNX };
         fOutputTensorNames = { fNY };
   }


   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("SOFIE - Tensor " + fNX + " not found.");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      fShapeY = fShapeX;
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShapeX.empty();
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
   }

   std::string Generate(std::string opName) override
   {
      opName = "op_" + opName;
      std::stringstream out;

      out << "\n//---- Operator Not  " << opName << "\n";
      if (fHasStridedInput) {
         out << GenerateStridedUnaryLoop(opName, fNX, fNY, fShapeX, [](const std::string &v) { return "!" + v; });
         return out.str();
      }
      auto length = ConvertDimShapeToLength(fShapeX);
      out << "for (size_t i = 0; i < " << length << "; i++) {\n";
      out << "tensor_" << fNY << "[i] = !tensor_" + fNX + "[i];\n";
      out << "}\n";
      return out.str();
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override
   {
      if (fIsOutputConstant)
         return "";
      if (fHasStridedInput)
         return GenerateStridedUnaryKernel("NotStridedKernel", "NOT", [](const std::string &v) { return "!" + v; });

      std::string op;
      op  = "\n//------  NOT_KERNEL_ALPAKA\n";
      op += "struct NotKernel {\n";
      op += "template<typename TAcc, typename T>\n";
      op += "ALPAKA_FN_ACC void operator()(\n";
      op += "TAcc const & acc,\n";
      op += "T const * data,\n";
      op += "T * output,\n";
      op += "std::size_t const length) const\n";
      op += "{\n";
      op += "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += "if (idx < length) {\n";
      op += "output[idx] = !data[idx];\n";
      op += "}\n";
      op += "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override
   {
      if (fHasStridedInput)
         return "NotStridedKernel notStridedKernel;\n";
      return "NotKernel notKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShapeX);

      if (fHasStridedInput)
         return GenerateStridedUnaryLaunch(opName, "notStridedKernel", "NOT", fNX, fNY, fShapeX);

      out << "\n//------ " << opName << "_ALPAKA\n";
      out << "auto const elementsPerThread_" << fNY << " = Vec::all(static_cast<Idx>(1));\n";
      out << "auto const elementsPerGrid_"   << fNY << " = Vec::all(Idx{" << length << "});\n";
      out << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << "auto task_" << opName
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
          << ", " << "notKernel"
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", " << length << ");\n";
      out << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   bool IsElementwise() const override { return !fIsOutputConstant && !fHasStridedInput; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return "!" + v;
   }

};

} // namespace SOFIE

#endif
