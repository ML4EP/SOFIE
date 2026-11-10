#include "SOFIE/quantization/frontend/RQuantization_Frontend.hxx"

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Cast.hxx"
#include "SOFIE/quantization/graph/RQuantization_Graph.hxx"

#include <set>
#include <stdexcept>

namespace SOFIE {

namespace {

bool IsFloat8(ETensorType type)
{
   return type == ETensorType::FLOAT8E4M3FN || type == ETensorType::FLOAT8E4M3FNUZ ||
          type == ETensorType::FLOAT8E5M2 || type == ETensorType::FLOAT8E5M2FNUZ ||
          type == ETensorType::FLOAT8E8M0;
}

// Returns the parameters of a native float8 tensor, which holds values directly: the format's
// own grid with scale 1.
QuantBoundaryParameters NativeFloat8Parameters(ETensorType type, const std::string &tensor)
{
   if (type != ETensorType::FLOAT8E4M3FN)
      throw std::runtime_error("SOFIE native low-precision tensor " + tensor + " is " + ConvertTypeToString(type) +
                               "; only FLOAT8E4M3FN has a codec");
   QuantBoundaryParameters params;
   params.encoding = EQuantEncoding::FP8E4M3;
   params.scaleValue = 1.0;
   params.origin = "native FP8 tensor " + tensor;
   return params;
}

} // namespace

std::vector<std::unique_ptr<ROperator>> NativeLowPrecisionFrontend::Prepare(RModel &model,
                                                                            QuantModelAccess &access) const
{
   // Moves every float8 initializer an operator reads behind a decode under its own name,
   // except those a Q/DQ or QONNX boundary reads: those are the boundary's codes (or zero
   // point), which its own frontend decodes on the boundary's grid.
   std::set<std::string> read, readByBoundary;
   for (std::size_t i = 0; i < access.OperatorCount(); ++i) {
      const auto &op = access.Operator(i);
      for (const auto &input : op.GetOpInputTensors())
         (op.IsQuantizationBoundary() ? readByBoundary : read).insert(std::string(input));
   }

   std::vector<std::unique_ptr<ROperator>> decodes;
   for (const auto &name : read) {
      if (access.IsGraphInput(name) && IsFloat8(model.GetTensorType(name)))
         throw std::runtime_error("SOFIE native FP8 graph input " + name +
                                  " is not supported; feed float and Cast to FLOAT8E4M3FN in the graph");
      if (!model.IsInitializedTensor(name) || !IsFloat8(model.GetTensorType(name)))
         continue;
      if (readByBoundary.count(name) != 0)
         throw std::runtime_error("SOFIE float8 initializer " + name +
                                  " is read both as quantization codes and as a native value");
      auto params = NativeFloat8Parameters(model.GetTensorType(name), name);
      const std::string codes = name + "_codes";
      access.RenameInitializer(name, codes);
      decodes.push_back(std::make_unique<ROperator_QuantDecode>(codes, name, std::move(params)));
      decodes.back()->fName = "native_fp8_decode_" + name;
   }
   return decodes;
}

std::optional<std::vector<std::unique_ptr<ROperator>>> NativeLowPrecisionFrontend::Translate(const ROperator &op,
                                                                                             RModel &) const
{
   const auto *cast = dynamic_cast<const ROperator_Cast *>(&op);
   if (cast == nullptr)
      return std::nullopt;
   std::vector<std::unique_ptr<ROperator>> replacement;
   const auto &in = cast->GetInputTensor();
   const auto &out = cast->GetOutputTensor();
   if (IsFloat8(cast->GetTargetType())) {
      // float -> float8: encodes onto the format, then decodes for every reader.
      auto params = NativeFloat8Parameters(cast->GetTargetType(), out);
      replacement.push_back(std::make_unique<ROperator_QuantEncode>(in, out + "_codes", params));
      replacement.push_back(std::make_unique<ROperator_QuantDecode>(out + "_codes", out, std::move(params)));
      return replacement;
   }
   // A Cast to float stays an ordinary float cast: every float8 value is already decoded to float.
   return std::nullopt;
}

} // namespace SOFIE
