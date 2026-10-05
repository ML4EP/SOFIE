#ifndef SOFIE_ROPERATOR_HARDSWISH
#define SOFIE_ROPERATOR_HARDSWISH

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
namespace SOFIE{

template <typename T>
class ROperator_HardSwish final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_HardSwish(){}
   ROperator_HardSwish(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
         throw std::runtime_error("SOFIE HardSwish Op Input Tensor " + fNX + " is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShape.empty();
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }

   bool SupportsStridedInput() const override { return true; }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("SOFIE HardSwish operator called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      out << "\n//------ HardSwish\n";
      if (fHasStridedInput) {
         out << GenerateStridedUnaryLoop(OpName, fNX, fNY, ConvertShapeToDim(fShape), [](const std::string &v) {
            return v + " * std::fmax(0x0p+0f, std::fmin(0x1p+0f, 0x1.5555555555555p-3f * " + v + " + 0x1p-1f))";
         });
         return out.str();
      }
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "float h = 0x1.5555555555555p-3f * tensor_" << fNX << "[id] + 0x1p-1f;\n";
      out << SP << SP << "tensor_" << fNY << "[id] = tensor_" << fNX
          << "[id] * std::fmax(0x0p+0f, std::fmin(0x1p+0f, h));\n";
      out << SP << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") };}
};

}


#endif
