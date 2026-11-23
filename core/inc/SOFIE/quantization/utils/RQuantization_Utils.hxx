#ifndef SOFIE_RQUANTIZATION_UTILS
#define SOFIE_RQUANTIZATION_UTILS

// Helpers shared across the quantization modules: host buffers, axis arithmetic,
// reading initializers and registering folded constants. They depend only on RModel and the
// spec module, so every other module (frontend, graph, handlers, lowered) can use them.

#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/quantization/spec/RQuantization_Spec.hxx"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace SOFIE {

// ---- host buffers -----------------------------------------------------------------------

// Copies `values` into a malloc'ed buffer owned by the returned pointer, as RModel stores tensors.
template <typename T>
std::shared_ptr<void> ToBuffer(const std::vector<T> &values)
{
   std::shared_ptr<void> data(std::malloc(values.size() * sizeof(T)), std::free);
   std::memcpy(data.get(), values.data(), values.size() * sizeof(T));
   return data;
}

// ---- axes -------------------------------------------------------------------------------

// Maps an ONNX axis in [-rank, rank) to [0, rank). Throws, naming `context`, when it is out of range.
std::size_t NormalizeAxis(std::int64_t axis, std::size_t rank, const std::string &context);

// Resolves the axis a per-channel parameter vector of `parameterCount` entries runs along in a
// tensor of `shape`: `axis` when given, else the first dimension
// whose extent matches. Returns -1 for a single parameter. ONNX's default of axis 1 is not applied.
int ResolvePerChannelAxis(std::optional<int> axis, const std::vector<std::size_t> &shape,
                          std::size_t parameterCount, const std::string &context);

// Returns the stride for the per-element channel of a per-channel spec: (index / stride) % channels.
// Returns 1 for a per-tensor spec.
std::size_t QuantSpecChannelStride(const QuantSpec &spec, const std::vector<std::size_t> &shape);

// ---- model tensors ----------------------------------------------------------------------

// Returns an initializer's values as doubles, whatever its numeric type. Throws, naming
// `context`, when `name` is not an initialized tensor, has an unsupported type or is empty.
std::vector<double> ReadInitializerAsDouble(RModel &model, const std::string &name, const std::string &context);

// Returns a constant's codes as whole numbers or raw float8 bytes. Throws when its storage is
// not a bakeable code type.
std::vector<double> ReadCodes(RModel &model, const std::string &tensor);

// Registers a constant produced by the workflow: up to kQuantizedFoldedConstantEmbedMaxLength
// elements are embedded in the generated code, larger ones go to the weight file.
void AddFoldedConstant(RModel &model, const std::string &name, ETensorType type,
                       const std::vector<std::size_t> &shape, std::shared_ptr<void> data);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_UTILS
