#include "SOFIE/quantization/frontend/RQuantization_Frontend.hxx"

#include "SOFIE/ROperator_QONNXQuant.hxx"

namespace SOFIE {

std::optional<std::vector<std::unique_ptr<ROperator>>> QonnxFrontend::Translate(const ROperator &op, RModel &) const
{
   const auto *quant = dynamic_cast<const ROperator_QONNXQuant *>(&op);
   if (quant == nullptr)
      return std::nullopt;

   QuantBoundaryParameters params;
   params.scaleTensor = quant->GetScaleTensor();
   params.zeroPointTensor = quant->GetZeroPointTensor();
   params.bitWidthTensor = quant->GetBitWidthTensor();
   params.encoding = EQuantEncoding::Integer;
   params.isSigned = quant->IsSigned();
   params.narrow = quant->IsNarrow();
   switch (quant->GetRoundingMode()) {
   case EQuantizationRoundingMode::FLOOR: params.rounding = EQuantRounding::Floor; break;
   case EQuantizationRoundingMode::TRUNCATE: params.rounding = EQuantRounding::Truncate; break;
   default: params.rounding = EQuantRounding::NearestEven; break;
   }
   params.origin = "QONNX Quant";

   // A Quant is a fake quant (float in, float on the grid out), expressed as an encode onto the
   // grid followed by its decode through an explicit codes tensor.
   const std::string codes = quant->GetOutputTensor() + "_codes";
   std::vector<std::unique_ptr<ROperator>> replacement;
   replacement.push_back(std::make_unique<ROperator_QuantEncode>(quant->GetInputTensor(), codes, params));
   replacement.push_back(std::make_unique<ROperator_QuantDecode>(codes, quant->GetOutputTensor(), std::move(params)));
   return replacement;
}

} // namespace SOFIE
