#include "SOFIE/quantization/frontend/RQuantization_Frontend.hxx"

#include "SOFIE/RModel.hxx"
#include "SOFIE/quantization/graph/RQuantization_Graph.hxx"
#include "SOFIE/quantization/spec/RQuantization_CodeFormat.hxx"
#include "SOFIE/quantization/utils/RQuantization_Utils.hxx"

#include <cmath>
#include <stdexcept>

namespace SOFIE {

QuantSpec ResolveQuantSpec(RModel &model, const QuantBoundaryParameters &params, const std::vector<std::size_t> &shape)
{
   const std::string context = "SOFIE " + params.origin;
   const auto scales = params.scaleValue ? std::vector<double>{*params.scaleValue}
                                         : ReadInitializerAsDouble(model, params.scaleTensor, context + " scale");
   for (double scale : scales)
      if (!(scale > 0.0) || !std::isfinite(scale))
         throw std::runtime_error(context + ": scales must be positive and finite");

   // E5M2 parses (QONNX / native Cast can name it) but has no codec, so it is rejected here
   if (params.encoding == EQuantEncoding::FP8E5M2)
      throw std::runtime_error(context + ": only the FLOAT8E4M3FN float8 format has a codec");
   if (params.encoding == EQuantEncoding::FP8E4M3) {
      if (scales.size() != 1)
         throw std::runtime_error(context + ": float8 quantization supports a per-tensor scale only");
      if (!params.zeroPointTensor.empty())
         for (double zp : ReadInitializerAsDouble(model, params.zeroPointTensor, context + " zero point"))
            if (zp != 0.0)
               throw std::runtime_error(context + ": float8 zero points must be zero");
      return MakeFloat8QuantSpec(params.encoding, scales[0]);
   }

   std::vector<std::int64_t> zeroPoints{0};
   if (!params.zeroPointTensor.empty()) {
      zeroPoints.clear();
      for (double zp : ReadInitializerAsDouble(model, params.zeroPointTensor, context + " zero point")) {
         if (!std::isfinite(zp) || std::round(zp) != zp)
            throw std::runtime_error(context + ": zero points must be finite integers");
         zeroPoints.push_back(static_cast<std::int64_t>(zp));
      }
   }
   unsigned bitWidth = params.bitWidth;
   if (!params.bitWidthTensor.empty()) {
      const auto bits = ReadInitializerAsDouble(model, params.bitWidthTensor, context + " bit width");
      if (bits.size() != 1 || std::round(bits[0]) != bits[0] || bits[0] < 1.0 || bits[0] > 32.0)
         throw std::runtime_error(context + ": bit width must be an integer scalar in [1, 32]");
      bitWidth = static_cast<unsigned>(bits[0]);
   }
   const std::size_t parameterCount = std::max(scales.size(), zeroPoints.size());
   if (scales.size() != 1 && zeroPoints.size() != 1 && scales.size() != zeroPoints.size())
      throw std::runtime_error(context + ": scale and zero-point vectors differ in length");
   const int axis = ResolvePerChannelAxis(params.axis, shape, parameterCount, context);

   // Broadcasts a scalar side so per-channel specs carry one entry per channel.
   std::vector<double> fullScales = scales;
   std::vector<std::int64_t> fullZeroPoints = zeroPoints;
   if (parameterCount > 1) {
      if (fullScales.size() == 1)
         fullScales.assign(parameterCount, scales[0]);
      if (fullZeroPoints.size() == 1)
         fullZeroPoints.assign(parameterCount, zeroPoints[0]);
   }
   return MakeIntegerQuantSpec(bitWidth, params.isSigned, params.narrow, std::move(fullScales),
                               std::move(fullZeroPoints), axis, params.rounding);
}

// ---- constant folding of the canonical boundaries ----------------------------------------

void ROperator_QuantEncode::FoldConstant(RModel &model)
{
   const auto type = fSpec.CodeType();
   if (!IsBakeableCodeType(type)) {
      // Storage SOFIE cannot bake (wider integers) encodes at run time.
      model.AddIntermediateTensor(fNY, type, fShape);
      return;
   }
   const auto values = model.GetTensorData<float>(fNX);
   const auto stride = QuantSpecChannelStride(fSpec, fShape);
   const auto channels = std::max<std::size_t>(fSpec.ChannelCount(), 1);
   std::vector<double> codes(values.size());
   for (std::size_t i = 0; i < codes.size(); ++i) {
      const std::size_t c = fSpec.IsPerTensor() ? 0 : (i / stride) % channels;
      codes[i] = QuantEncodeValue(fSpec, static_cast<double>(values[i]), c);
   }
   AddFoldedConstant(model, fNY, type, fShape, PackCodes(type, codes));
   fIsOutputConstant = true;
}

void ROperator_QuantDecode::FoldConstant(RModel &model)
{
   const auto codes = ReadInitializerAsDouble(model, fNX, "SOFIE " + fParams.origin + " constant carrier");
   const auto stride = QuantSpecChannelStride(fSpec, fShape);
   const auto channels = std::max<std::size_t>(fSpec.ChannelCount(), 1);
   std::vector<float> values(codes.size());
   for (std::size_t i = 0; i < values.size(); ++i) {
      const std::size_t c = fSpec.IsPerTensor() ? 0 : (i / stride) % channels;
      values[i] = static_cast<float>(QuantDecodeCode(fSpec, codes[i], c));
   }
   AddFoldedConstant(model, fNY, ETensorType::FLOAT, fShape, ToBuffer(values));
   fIsOutputConstant = true;
}

// ---- registry and driver -----------------------------------------------------------------

const std::vector<std::unique_ptr<IQuantFrontend>> &QuantFrontends()
{
   static const std::vector<std::unique_ptr<IQuantFrontend>> frontends = [] {
      std::vector<std::unique_ptr<IQuantFrontend>> list;
      list.push_back(std::make_unique<QdqFrontend>());
      list.push_back(std::make_unique<QonnxFrontend>());
      list.push_back(std::make_unique<NativeLowPrecisionFrontend>());
      return list;
   }();
   return frontends;
}

std::size_t NormalizeQuantizationFrontends(RModel &model)
{
   QuantModelAccess access(model);
   std::vector<std::unique_ptr<ROperator>> canonical;
   std::size_t translated = 0;
   // Graph-level rewrites run first, while the parsed list is in the model;
   for (const auto &frontend : QuantFrontends()) {
      auto prepended = frontend->Prepare(model, access);
      translated += prepended.size();
      for (auto &node : prepended)
         canonical.push_back(std::move(node));
   }
   access.UpdateOperators([&](RModel::OperatorList parsed) {
      canonical.reserve(canonical.size() + parsed.size());
      for (auto &op : parsed) {
         std::optional<std::vector<std::unique_ptr<ROperator>>> replacement;
         for (const auto &frontend : QuantFrontends()) {
            replacement = frontend->Translate(*op, model);
            if (replacement)
               break;
         }
         if (!replacement) {
            canonical.push_back(std::move(op));
            continue;
         }
         ++translated;
         for (std::size_t i = 0; i < replacement->size(); ++i) {
            auto &node = (*replacement)[i];
            node->fName = replacement->size() == 1 ? op->fName : op->fName + "_" + std::to_string(i);
            canonical.push_back(std::move(node));
         }
      }
      return std::move(canonical);
   });
   return translated;
}

} // namespace SOFIE
