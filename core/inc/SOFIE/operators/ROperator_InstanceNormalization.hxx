#ifndef SOFIE_ROPERATOR_INSTANCENORMALIZATION
#define SOFIE_ROPERATOR_INSTANCENORMALIZATION

#include "SOFIE/operators/ROperator.hxx"
#include "SOFIE/model/RModel.hxx"
#include "SOFIE/utils/SOFIE_common.hxx"

#include <sstream>
#include <stdexcept>
namespace SOFIE {

template <typename T>
class ROperator_InstanceNormalization final : public ROperator {
private:
   float fEpsilon;
   std::string fNInput;
   std::string fNScale;
   std::string fNBias;
   std::string fNOutput;
   std::vector<size_t> fShape;
   std::string fType;

public:
   ROperator_InstanceNormalization() : fEpsilon(1e-5) {}

   ROperator_InstanceNormalization(float epsilon, std::string nameInput, std::string nameScale, std::string nameBias,
                                   std::string nameOutput)
      : fEpsilon(epsilon),
        fNInput(UTILITY::Clean_name(nameInput)),
        fNScale(UTILITY::Clean_name(nameScale)),
        fNBias(UTILITY::Clean_name(nameBias)),
        fNOutput(UTILITY::Clean_name(nameOutput))
   {
      fInputTensorNames = {fNInput, fNScale, fNBias};
      fOutputTensorNames = {fNOutput};
   }

   void Initialize(RModel &model) override
   {
      for (const std::string &name : {fNInput, fNScale, fNBias}) {
         if (!model.CheckIfTensorAlreadyExist(name)) {
            throw std::runtime_error("SOFIE - InstanceNormalization - Tensor " + name + " not found.");
         }
      }

      fShape = model.GetTensorShape(fNInput);
      if (fShape.size() < 3) {
         throw std::runtime_error("SOFIE - InstanceNormalization - Input tensor " + fNInput + " has rank " +
                                  std::to_string(fShape.size()) + " but at least rank 3 (N, C, D1, ...) is required.");
      }

      const size_t channels = fShape[1];
      for (const std::string &name : {fNScale, fNBias}) {
         auto shape = model.GetTensorShape(name);
         if (shape.size() != 1 || shape[0] != channels) {
            throw std::runtime_error("SOFIE - InstanceNormalization - Tensor " + name + " has invalid shape " +
                                     ConvertShapeToString(shape) + ", expected " +
                                     ConvertShapeToString(std::vector<size_t>{channels}) + ".");
         }
      }

      fType = ConvertTypeToString(model.GetTensorType(fNInput));
      fHasStridedInput = model.IsStridedInputTensor(fNInput);
      if (model.IsStridedInputTensor(fNScale) || model.IsStridedInputTensor(fNBias))
         throw std::runtime_error("SOFIE - InstanceNormalization - strided input is only supported for the data tensor");
      model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNInput), fShape);
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string) override
   {
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE - InstanceNormalization called to generate without being initialized.");
      }

      const size_t batchSize = fShape[0];
      const size_t channels = fShape[1];
      size_t spatialSize = 1;
      for (size_t i = 2; i < fShape.size(); ++i)
         spatialSize *= fShape[i];

      std::stringstream out;
      out << "\n" << "//---- InstanceNormalization " << fNOutput << "\n";
      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      const std::string xAt = "xoff_" + fNOutput;
      auto readX = [&](const std::string &index) {
         return fHasStridedInput ? "tensor_" + fNInput + "[" + xAt + "(" + index + ")]"
                                 : "tensor_" + fNInput + "[" + index + "]";
      };
      if (fHasStridedInput)
         out << GenerateStridedOffsetLambda(fNOutput, fNInput, ConvertShapeToDim(fShape));
      out << "for (size_t n = 0; n < " << batchSize << "; n++) {\n";
      out << "for (size_t c = 0; c < " << channels << "; c++) {\n";
      out << "const size_t offset = n * " << channels * spatialSize << " + c * " << spatialSize
          << ";\n";

      out << fType << " mean = 0.;\n";
      out << "for (size_t i = 0; i < " << spatialSize << "; i++) {\n";
      out << "mean += " << readX("offset + i") << ";\n";
      out << "}\n";
      out << "mean /= " << fType << "(" << spatialSize << ");\n";

      out << fType << " sum = 0.;\n";
      out << "for (size_t i = 0; i < " << spatialSize << "; i++) {\n";
      out << fType << " tmp = " << readX("offset + i") << " - mean;\n";
      out << "sum += tmp * tmp;\n";
      out << "}\n";
      out << fType << " invStdDev = 1 / std::sqrt(sum / " << fType << "(" << spatialSize << ") + "
          << std::to_string(fEpsilon) << ");\n";

      out << fType << " scale = tensor_" << fNScale << "[c];\n";
      out << fType << " bias = tensor_" << fNBias << "[c];\n";
      out << "for (size_t i = 0; i < " << spatialSize << "; i++) {\n";
      out << "tensor_" << fNOutput << "[offset + i] = scale * (" << readX("offset + i")
          << " - mean) * invStdDev + bias;\n";
      out << "}\n";

      out << "}\n";
      out << "}\n";
      return out.str();
   }
};

}
#endif
