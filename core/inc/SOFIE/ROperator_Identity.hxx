#ifndef SOFIE_ROPERATOR_IDENTITY
#define SOFIE_ROPERATOR_IDENTITY

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename T>
class ROperator_Identity final : public ROperator
{

private:
   bool fIsOutputInitialized = false;
   bool fIsAlias = false;
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_Identity(){}
   ROperator_Identity(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("SOFIE Identity Op Input Tensor is not found in model");
      }
      fShape = model.GetDimTensorShape(fNX);
      if (model.IsInitializedTensor(fNX)) {
         if (model.IsConstantTensor(fNX)) {
            auto inputData = static_cast<T*>(model.GetInitializedTensorData(fNX).get());
            model.AddConstantTensor<T>(fNY, model.GetTensorShape(fNX), inputData);
            fIsOutputConstant = true;
         } else {
            fIsOutputInitialized = true;
            model.AddInitializedTensor(fNY, model.GetTensorType(fNX), model.GetTensorShape(fNX),
                                       model.GetInitializedTensorData(fNX));
         }
      } else {
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
         fIsAlias = model.AddAliasTensor(fNY, fNX);
      }
   }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant || fIsOutputInitialized)
         return "";
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator Identity called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ IDENTITY\n";
      if (fIsAlias) {
         out << SP << "auto * tensor_" << fNY << " = tensor_" << fNX << ";\n";
      } else {
         out << SP << "std::copy(tensor_" << fNX << ", tensor_" << fNX << " + " << ConvertDimShapeToLength(fShape)
             << ", tensor_" << fNY << ");\n";
      }
      return out.str();
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      // Constant outputs and already-initialised tensors need no runtime work.
      if (fIsOutputConstant || fIsOutputInitialized) return "";
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator Identity called to Generate_GPU_ALPAKA without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ IDENTITY\n";
      out << SP << "alpaka::memcpy(queue, deviceBuf_" << fNY << ", deviceBuf_" << fNX << ");\n";
      return out.str();
   }

   bool IsElementwise() const override {
      return !fIsOutputConstant && !fIsOutputInitialized;
   }

   std::string GetElementwiseExpr(const std::string &inputVar) const override {
      return inputVar;
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_IDENTITY
