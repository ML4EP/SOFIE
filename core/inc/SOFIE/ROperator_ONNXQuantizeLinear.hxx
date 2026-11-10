#ifndef SOFIE_ROPERATOR_ONNXQUANTIZELINEAR
#define SOFIE_ROPERATOR_ONNXQUANTIZELINEAR

// Defines ONNX QuantizeLinear / DequantizeLinear as parsed. They hold the node's tensors and
// attributes only: the Q/DQ frontend replaces them with canonical QuantEncode / QuantDecode
// boundaries before the model initializes, so they are never initialized or code-generated.

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/SOFIE_common.hxx"

#include <stdexcept>
#include <string>
#include <vector>

namespace SOFIE {

namespace DETAIL {
// Throws when a parsed Q/DQ node reaches Initialize without frontend translation.
[[noreturn]] inline void ThrowUntranslatedQDQ(const char *op, const std::string &tensor)
{
   throw std::runtime_error(std::string("SOFIE ") + op + " producing " + tensor +
                            " reached Initialize untranslated; quantized models need the SOFIE_quantization "
                            "library (link SOFIE_parsers or SOFIE_quantization)");
}
} // namespace DETAIL

class ROperator_ONNXQuantizeLinear final : public ROperator {
   std::string fNX, fNScale, fNZeroPoint, fNY;
   ETensorType fOutputType = ETensorType::UNDEFINED;
   int fAxis = -1;

public:
   ROperator_ONNXQuantizeLinear(std::string nameX, std::string nameScale, std::string nameZeroPoint,
                                std::string nameY, ETensorType outputType, int axis)
      : fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
        fNZeroPoint(UTILITY::Clean_name(nameZeroPoint)), fNY(UTILITY::Clean_name(nameY)),
        fOutputType(outputType), fAxis(axis)
   {
      fInputTensorNames = fNZeroPoint.empty() ? std::vector<std::string>{fNX, fNScale}
                                             : std::vector<std::string>{fNX, fNScale, fNZeroPoint};
      fOutputTensorNames = {fNY};
   }

   bool IsQuantizationBoundary() const override { return true; }
   std::string GetQuantizationSourceTensor() const override { return fNX; }

   const std::string &GetInputTensor() const { return fNX; }
   const std::string &GetOutputTensor() const { return fNY; }
   const std::string &GetScaleTensor() const { return fNScale; }
   const std::string &GetZeroPointTensor() const { return fNZeroPoint; }
   ETensorType GetOutputType() const { return fOutputType; }
   int GetAxis() const { return fAxis; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType>) override { return {fOutputType}; }
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      return input.empty() ? std::vector<std::vector<size_t>>{} : std::vector<std::vector<size_t>>{input.front()};
   }
   void Initialize(RModel &) override { DETAIL::ThrowUntranslatedQDQ("ONNX QuantizeLinear", fNY); }
   std::string Generate(std::string) override { DETAIL::ThrowUntranslatedQDQ("ONNX QuantizeLinear", fNY); }
};

class ROperator_ONNXDequantizeLinear final : public ROperator {
   std::string fNX, fNScale, fNZeroPoint, fNY;
   ETensorType fInputType = ETensorType::UNDEFINED;
   int fAxis = -1;

public:
   ROperator_ONNXDequantizeLinear(std::string nameX, std::string nameScale, std::string nameZeroPoint,
                                  std::string nameY, ETensorType inputType, int axis)
      : fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
        fNZeroPoint(UTILITY::Clean_name(nameZeroPoint)), fNY(UTILITY::Clean_name(nameY)),
        fInputType(inputType), fAxis(axis)
   {
      fInputTensorNames = fNZeroPoint.empty() ? std::vector<std::string>{fNX, fNScale}
                                             : std::vector<std::string>{fNX, fNScale, fNZeroPoint};
      fOutputTensorNames = {fNY};
   }

   bool IsQuantizationBoundary() const override { return true; }
   std::string GetQuantizationSourceTensor() const override { return fNX; }

   const std::string &GetInputTensor() const { return fNX; }
   const std::string &GetOutputTensor() const { return fNY; }
   const std::string &GetScaleTensor() const { return fNScale; }
   const std::string &GetZeroPointTensor() const { return fNZeroPoint; }
   ETensorType GetInputType() const { return fInputType; }
   int GetAxis() const { return fAxis; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType>) override { return {ETensorType::FLOAT}; }
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      return input.empty() ? std::vector<std::vector<size_t>>{} : std::vector<std::vector<size_t>>{input.front()};
   }
   void Initialize(RModel &) override { DETAIL::ThrowUntranslatedQDQ("ONNX DequantizeLinear", fNY); }
   std::string Generate(std::string) override { DETAIL::ThrowUntranslatedQDQ("ONNX DequantizeLinear", fNY); }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_ONNXQUANTIZELINEAR
