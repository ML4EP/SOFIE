#ifndef SOFIE_ROPERATOR_QONNXQUANT
#define SOFIE_ROPERATOR_QONNXQUANT

// Defines qonnx.custom_op.general::Quant as parsed. It holds the node's tensors and attributes
// only: the QONNX frontend replaces it with a canonical QuantEncode / QuantDecode pair before the
// model initializes.

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/SOFIE_common.hxx"

#include <stdexcept>
#include <string>
#include <vector>

namespace SOFIE {

class ROperator_QONNXQuant final : public ROperator {
   std::string fNX, fNScale, fNZeroPoint, fNBitWidth, fNY;
   bool fIsSigned = false;
   bool fNarrow = false;
   EQuantizationRoundingMode fRounding = EQuantizationRoundingMode::UNDEFINED;

   [[noreturn]] void ThrowUntranslated() const
   {
      throw std::runtime_error("SOFIE QONNX Quant producing " + fNY +
                               " reached Initialize untranslated; quantized models need the SOFIE_quantization "
                               "library (link SOFIE_parsers or SOFIE_quantization)");
   }

public:
   ROperator_QONNXQuant(std::string nameX, std::string nameScale, std::string nameZeroPoint,
                        std::string nameBitWidth, std::string nameY, bool isSigned, bool narrow,
                        EQuantizationRoundingMode rounding)
      : fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
        fNZeroPoint(UTILITY::Clean_name(nameZeroPoint)), fNBitWidth(UTILITY::Clean_name(nameBitWidth)),
        fNY(UTILITY::Clean_name(nameY)), fIsSigned(isSigned), fNarrow(narrow), fRounding(rounding)
   {
      fInputTensorNames = {fNX, fNScale, fNZeroPoint, fNBitWidth};
      fOutputTensorNames = {fNY};
   }

   bool IsQuantizationBoundary() const override { return true; }
   std::string GetQuantizationSourceTensor() const override { return fNX; }

   const std::string &GetInputTensor() const { return fNX; }
   const std::string &GetScaleTensor() const { return fNScale; }
   const std::string &GetZeroPointTensor() const { return fNZeroPoint; }
   const std::string &GetBitWidthTensor() const { return fNBitWidth; }
   const std::string &GetOutputTensor() const { return fNY; }
   bool IsSigned() const { return fIsSigned; }
   bool IsNarrow() const { return fNarrow; }
   EQuantizationRoundingMode GetRoundingMode() const { return fRounding; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   {
      return input.empty() ? std::vector<ETensorType>{} : std::vector<ETensorType>{input.front()};
   }
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      return input.empty() ? std::vector<std::vector<size_t>>{} : std::vector<std::vector<size_t>>{input.front()};
   }
   void Initialize(RModel &) override { ThrowUntranslated(); }
   std::string Generate(std::string) override { ThrowUntranslated(); }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QONNXQUANT
