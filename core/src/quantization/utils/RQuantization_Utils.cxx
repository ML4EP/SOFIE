#include "SOFIE/quantization/utils/RQuantization_Utils.hxx"

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/quantization/spec/RQuantization_CodeFormat.hxx"

#include <stdexcept>

namespace SOFIE {

std::size_t NormalizeAxis(std::int64_t axis, std::size_t rank, const std::string &context)
{
   const auto r = static_cast<std::int64_t>(rank);
   const std::int64_t normalized = axis < 0 ? axis + r : axis;
   if (normalized < 0 || normalized >= r)
      throw std::runtime_error(context + ": axis " + std::to_string(axis) + " is outside the tensor rank " +
                               std::to_string(rank));
   return static_cast<std::size_t>(normalized);
}

int ResolvePerChannelAxis(std::optional<int> axis, const std::vector<std::size_t> &shape,
                          std::size_t parameterCount, const std::string &context)
{
   if (parameterCount <= 1)
      return -1;
   if (axis) {
      const std::size_t resolved = NormalizeAxis(*axis, shape.size(), context + " per-channel");
      if (shape[resolved] != parameterCount)
         throw std::runtime_error(context + ": per-channel parameter count does not match the axis extent");
      return static_cast<int>(resolved);
   }
   for (std::size_t i = 0; i < shape.size(); ++i)
      if (shape[i] == parameterCount)
         return static_cast<int>(i);
   throw std::runtime_error(context + ": cannot infer the per-channel axis");
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

std::vector<double> ReadCodes(RModel &model, const std::string &tensor)
{
   const auto data = model.GetInitializedTensorData(tensor);
   return UnpackCodes(model.GetTensorType(tensor), data.get(), ConvertShapeToLength(model.GetTensorShape(tensor)));
}

void AddFoldedConstant(RModel &model, const std::string &name, ETensorType type,
                       const std::vector<std::size_t> &shape, std::shared_ptr<void> data)
{
   if (ConvertShapeToLength(shape) > kQuantizedFoldedConstantEmbedMaxLength)
      model.AddInitializedTensor(name, type, shape, std::move(data));
   else
      model.AddConstantTensor(name, type, shape, std::move(data));
}

} // namespace SOFIE
