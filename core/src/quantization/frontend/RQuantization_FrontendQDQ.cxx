#include "SOFIE/quantization/frontend/RQuantization_Frontend.hxx"

#include "SOFIE/ROperator_ONNXQuantizeLinear.hxx"

#include <stdexcept>

namespace SOFIE {

namespace {

// Returns the boundary parameters for an ONNX Q/DQ carrier type (the zero point's dtype, uint8
// by default), which fixes the grid.
QuantBoundaryParameters ParametersForCarrier(ETensorType carrier, const std::string &scale,
                                             const std::string &zeroPoint, int axis, const char *origin)
{
   QuantBoundaryParameters params;
   params.scaleTensor = scale;
   params.zeroPointTensor = zeroPoint;
   params.axis = axis;
   params.origin = origin;
   params.rounding = EQuantRounding::NearestEven;
   params.narrow = false;
   switch (carrier) {
   case ETensorType::FLOAT8E4M3FN: params.encoding = EQuantEncoding::FP8E4M3; break;
   case ETensorType::FLOAT8E5M2: params.encoding = EQuantEncoding::FP8E5M2; break;
   case ETensorType::INT8: params.bitWidth = 8; params.isSigned = true; break;
   case ETensorType::UINT8: params.bitWidth = 8; params.isSigned = false; break;
   case ETensorType::INT16: params.bitWidth = 16; params.isSigned = true; break;
   case ETensorType::UINT16: params.bitWidth = 16; params.isSigned = false; break;
   case ETensorType::INT32: params.bitWidth = 32; params.isSigned = true; break;
   case ETensorType::UINT32: params.bitWidth = 32; params.isSigned = false; break;
   default:
      throw std::runtime_error(std::string("SOFIE ") + origin + ": unsupported carrier type " +
                               ConvertTypeToString(carrier));
   }
   return params;
}

} // namespace

std::optional<std::vector<std::unique_ptr<ROperator>>> QdqFrontend::Translate(const ROperator &op, RModel &) const
{
   std::vector<std::unique_ptr<ROperator>> replacement;
   if (const auto *quantize = dynamic_cast<const ROperator_ONNXQuantizeLinear *>(&op)) {
      auto params = ParametersForCarrier(quantize->GetOutputType(), quantize->GetScaleTensor(),
                                         quantize->GetZeroPointTensor(), quantize->GetAxis(), "ONNX QuantizeLinear");
      replacement.push_back(std::make_unique<ROperator_QuantEncode>(quantize->GetInputTensor(),
                                                                    quantize->GetOutputTensor(), std::move(params)));
      return replacement;
   }
   if (const auto *dequantize = dynamic_cast<const ROperator_ONNXDequantizeLinear *>(&op)) {
      auto params = ParametersForCarrier(dequantize->GetInputType(), dequantize->GetScaleTensor(),
                                         dequantize->GetZeroPointTensor(), dequantize->GetAxis(),
                                         "ONNX DequantizeLinear");
      replacement.push_back(std::make_unique<ROperator_QuantDecode>(dequantize->GetInputTensor(),
                                                                    dequantize->GetOutputTensor(), std::move(params)));
      return replacement;
   }
   return std::nullopt;
}

} // namespace SOFIE
