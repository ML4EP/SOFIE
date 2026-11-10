#ifndef SOFIE_RQUANTIZATION_FRONTEND
#define SOFIE_RQUANTIZATION_FRONTEND

// Provides the frontend adapters. Each translates its format's boundary nodes into the canonical
// ROperator_QuantEncode/QuantDecode, so after NormalizeQuantizationFrontends no other component
// distinguishes the formats. A new format is one IQuantFrontend registered in QuantFrontends().

#include "SOFIE/ROperator.hxx"
#include "SOFIE/quantization/frontend/ROperator_QuantBoundary.hxx"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace SOFIE {

class RModel;
class QuantModelAccess;

class IQuantFrontend {
public:
   virtual ~IQuantFrontend() = default;
   virtual const char *Name() const = 0;
   // Applies graph-level rewrites before node translation; returns operators that run first.
   virtual std::vector<std::unique_ptr<ROperator>> Prepare(RModel &, QuantModelAccess &) const { return {}; }
   // Returns the canonical replacement for `op`, or nullopt when `op` belongs to another frontend.
   // Runs before Initialize: initializers are readable, intermediate shapes are not.
   virtual std::optional<std::vector<std::unique_ptr<ROperator>>> Translate(const ROperator &op,
                                                                              RModel &model) const = 0;
};

// Translates ONNX QuantizeLinear / DequantizeLinear to Encode / Decode.
class QdqFrontend final : public IQuantFrontend {
public:
   const char *Name() const override { return "ONNX Q/DQ"; }
   std::optional<std::vector<std::unique_ptr<ROperator>>> Translate(const ROperator &op,
                                                                      RModel &model) const override;
};

// Translates QONNX Quant (qonnx.custom_op.general) to Decode(Encode(x)), the fake quant it denotes.
class QonnxFrontend final : public IQuantFrontend {
public:
   const char *Name() const override { return "QONNX Quant"; }
   std::optional<std::vector<std::unique_ptr<ROperator>>> Translate(const ROperator &op,
                                                                      RModel &model) const override;
};

// Translates native low-precision tensors: ONNX Cast to FLOAT8E4M3FN and E4M3 initializers read
// directly by arithmetic. A Cast maps to Encode(scale 1) + Decode under the Cast's output name; an
// E4M3 initializer moves to "<name>_codes" behind a Decode, so its readers stay unchanged.
class NativeLowPrecisionFrontend final : public IQuantFrontend {
public:
   const char *Name() const override { return "native FP8"; }
   std::vector<std::unique_ptr<ROperator>> Prepare(RModel &model, QuantModelAccess &access) const override;
   std::optional<std::vector<std::unique_ptr<ROperator>>> Translate(const ROperator &op,
                                                                      RModel &model) const override;
};

// Returns the registered frontends in the order they are queried.
const std::vector<std::unique_ptr<IQuantFrontend>> &QuantFrontends();

// Rewrites every frontend node of the model into canonical boundaries and returns the number of
// translated nodes. Runs before RModel::Initialize.
std::size_t NormalizeQuantizationFrontends(RModel &model);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_FRONTEND
