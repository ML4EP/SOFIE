#include "SOFIE/quantization/frontend/RQuantization_Frontend.hxx"

#include "SOFIE/RModel.hxx"
#include "SOFIE/quantization/graph/RQuantization_Graph.hxx"
#include "SOFIE/quantization/spec/RQuantization_Codec.hxx"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace SOFIE {

namespace {

std::vector<double> ReadInitializerAsDouble(RModel &model, const std::string &name, const std::string &context)
{
   if (name.empty() || !model.IsInitializedTensor(name))
      throw std::runtime_error(context + ": " + (name.empty() ? std::string("missing") : name) +
                               " must be an initialized tensor");
   std::vector<double> values;
   auto append = [&values](const auto &typed) { values.assign(typed.begin(), typed.end()); };
   switch (model.GetTensorType(name)) {
   case ETensorType::FLOAT: append(model.GetTensorData<float>(name)); break;
   case ETensorType::DOUBLE: append(model.GetTensorData<double>(name)); break;
   case ETensorType::INT8: append(model.GetTensorData<std::int8_t>(name)); break;
   case ETensorType::UINT8:
   case ETensorType::FLOAT8E4M3FN: append(model.GetTensorData<std::uint8_t>(name)); break;
   case ETensorType::INT16: append(model.GetTensorData<std::int16_t>(name)); break;
   case ETensorType::UINT16: append(model.GetTensorData<std::uint16_t>(name)); break;
   case ETensorType::INT32: append(model.GetTensorData<std::int32_t>(name)); break;
   case ETensorType::UINT32: append(model.GetTensorData<std::uint32_t>(name)); break;
   case ETensorType::INT64: append(model.GetTensorData<std::int64_t>(name)); break;
   default:
      throw std::runtime_error(context + ": unsupported initializer type for " + name);
   }
   if (values.empty())
      throw std::runtime_error(context + ": " + name + " is empty");
   return values;
}

// Resolves the axis a parameter vector runs along: the explicit axis, else the first dimension
// whose extent matches the vector length. ONNX's default of axis 1 is not applied.
int ResolveAxis(const QuantBoundaryParameters &params, const std::vector<std::size_t> &shape,
                std::size_t parameterCount, const std::string &context)
{
   if (parameterCount <= 1)
      return -1;
   const int rank = static_cast<int>(shape.size());
   if (params.axis) {
      int axis = *params.axis < 0 ? *params.axis + rank : *params.axis;
      if (axis < 0 || axis >= rank)
         throw std::runtime_error(context + ": per-channel axis is outside the tensor rank");
      if (shape[axis] != parameterCount)
         throw std::runtime_error(context + ": per-channel parameter count does not match the axis extent");
      return axis;
   }
   for (int i = 0; i < rank; ++i)
      if (shape[i] == parameterCount)
         return i;
   throw std::runtime_error(context + ": cannot infer the per-channel axis");
}

template <typename T>
std::shared_ptr<void> CopyToBuffer(const std::vector<T> &values)
{
   std::shared_ptr<void> data(malloc(values.size() * sizeof(T)), free);
   std::memcpy(data.get(), values.data(), values.size() * sizeof(T));
   return data;
}

// Registers a folded constant: small ones are embedded, large ones go to the weight file.
void RegisterFoldedTensor(RModel &model, const std::string &name, ETensorType type,
                          const std::vector<std::size_t> &shape, std::shared_ptr<void> data)
{
   if (ConvertShapeToLength(shape) > kQuantizedFoldedConstantEmbedMaxLength)
      model.AddInitializedTensor(name, type, shape, std::move(data));
   else
      model.AddConstantTensor(name, type, shape, std::move(data));
}

} // namespace

QuantSpec ResolveQuantSpec(RModel &model, const QuantBoundaryParameters &params, const std::vector<std::size_t> &shape)
{
   const std::string context = "SOFIE " + params.origin;
   const auto scales = params.scaleValue ? std::vector<double>{*params.scaleValue}
                                         : ReadInitializerAsDouble(model, params.scaleTensor, context + " scale");
   for (double scale : scales)
      if (!(scale > 0.0) || !std::isfinite(scale))
         throw std::runtime_error(context + ": scales must be positive and finite");

   if (params.encoding == EQuantEncoding::FP8E4M3 || params.encoding == EQuantEncoding::FP8E5M2) {
      if (params.encoding != EQuantEncoding::FP8E4M3)
         throw std::runtime_error(context + ": only the FLOAT8E4M3FN float8 format has a codec");
      if (scales.size() != 1)
         throw std::runtime_error(context + ": float8 quantization supports a per-tensor scale only");
      if (!params.zeroPointTensor.empty())
         for (double zp : ReadInitializerAsDouble(model, params.zeroPointTensor, context + " zero point"))
            if (zp != 0.0)
               throw std::runtime_error(context + ": float8 zero points must be zero");
      auto spec = MakeFloat8QuantSpec(params.encoding, scales[0]);
      spec.scaleTensor = params.scaleTensor;
      return spec;
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
   const int axis = ResolveAxis(params, shape, parameterCount, context);

   // Broadcasts a scalar side so per-channel specs carry one entry per channel.
   std::vector<double> fullScales = scales;
   std::vector<std::int64_t> fullZeroPoints = zeroPoints;
   if (parameterCount > 1) {
      if (fullScales.size() == 1)
         fullScales.assign(parameterCount, scales[0]);
      if (fullZeroPoints.size() == 1)
         fullZeroPoints.assign(parameterCount, zeroPoints[0]);
   }
   auto spec = MakeIntegerQuantSpec(bitWidth, params.isSigned, params.narrow, std::move(fullScales),
                                    std::move(fullZeroPoints), axis, params.rounding);
   spec.scaleTensor = params.scaleTensor;
   spec.zeroPointTensor = params.zeroPointTensor;
   return spec;
}

std::size_t QuantSpecChannelStride(const QuantSpec &spec, const std::vector<std::size_t> &shape)
{
   if (spec.IsPerTensor() || spec.axis < 0)
      return 1;
   std::size_t stride = 1;
   for (std::size_t i = static_cast<std::size_t>(spec.axis) + 1; i < shape.size(); ++i)
      stride *= shape[i];
   return stride;
}

// ---- constant folding of the canonical boundaries ----------------------------------------

void ROperator_QuantEncode::FoldConstant(RModel &model)
{
   const auto values = model.GetTensorData<float>(fNX);
   const auto stride = QuantSpecChannelStride(fSpec, fShape);
   const auto channels = std::max<std::size_t>(fSpec.ChannelCount(), 1);
   auto codeAt = [&](std::size_t i) {
      const std::size_t c = fSpec.IsPerTensor() ? 0 : (i / stride) % channels;
      if (fSpec.IsFloatingPoint())
         return static_cast<double>(QuantCodec::EncodeE4M3Scaled(static_cast<double>(values[i]), fSpec.Scale(c)));
      return QuantCodec::EncodeInteger(values[i], fSpec.Scale(c), static_cast<double>(fSpec.ZeroPoint(c)),
                                       fSpec.codeMin, fSpec.codeMax, RoundingCode());
   };
   const auto type = fSpec.CodeType();
   std::shared_ptr<void> data;
   switch (type) {
   case ETensorType::INT8: {
      std::vector<std::int8_t> codes(values.size());
      for (std::size_t i = 0; i < codes.size(); ++i)
         codes[i] = static_cast<std::int8_t>(codeAt(i));
      data = CopyToBuffer(codes);
      break;
   }
   case ETensorType::UINT8:
   case ETensorType::FLOAT8E4M3FN: {
      std::vector<std::uint8_t> codes(values.size());
      for (std::size_t i = 0; i < codes.size(); ++i)
         codes[i] = static_cast<std::uint8_t>(codeAt(i));
      data = CopyToBuffer(codes);
      break;
   }
   default:
      // Wider integer carriers have no initializer codegen, so they encode at run time.
      model.AddIntermediateTensor(fNY, type, fShape);
      return;
   }
   RegisterFoldedTensor(model, fNY, type, fShape, std::move(data));
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
      values[i] = static_cast<float>(
         fSpec.IsFloatingPoint()
            ? QuantCodec::DecodeE4M3(static_cast<std::uint8_t>(codes[i])) * fSpec.Scale(c)
            : QuantCodec::DecodeInteger(codes[i], fSpec.Scale(c), static_cast<double>(fSpec.ZeroPoint(c))));
   }
   RegisterFoldedTensor(model, fNY, ETensorType::FLOAT, fShape, CopyToBuffer(values));
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
   // Graph-level rewrites run first, while the parsed list is in the model (e.g. moving
   // low-precision initializers behind a decode); the operators they create run before every node.
   for (const auto &frontend : QuantFrontends()) {
      auto prepended = frontend->Prepare(model, access);
      translated += prepended.size();
      for (auto &node : prepended)
         canonical.push_back(std::move(node));
   }
   auto parsed = access.TakeOperators();
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
   access.SetOperators(std::move(canonical));
   return translated;
}

} // namespace SOFIE
