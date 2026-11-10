#ifndef SOFIE_RQUANTIZATION_HANDLERSUPPORT
#define SOFIE_RQUANTIZATION_HANDLERSUPPORT

// Helpers shared by the family handlers: kernel-grid predicates, constant readers, and the
// absorption walk after an arithmetic operator.

#include "SOFIE/RModel.hxx"
#include "SOFIE/quantization/handlers/RQuantization_Handler.hxx"
#include "SOFIE/quantization/lowered/ROperator_QuantLowered.hxx"

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace SOFIE {

// Returns whether a generated kernel can decode or encode codes on `spec`: per-tensor integer
// of at most 8 bits (int8 / uint8 storage) or per-tensor E4M3.
inline bool IsKernelGrid(const QuantSpec &spec)
{
   if (!spec.IsDefined() || !spec.IsPerTensor())
      return false;
   if (spec.encoding == EQuantEncoding::FP8E4M3)
      return true;
   return spec.encoding == EQuantEncoding::Integer && spec.bitWidth <= 8;
}

inline bool IsSignedInt8Grid(const QuantSpec &spec)
{
   return spec.encoding == EQuantEncoding::Integer && spec.bitWidth == 8 && spec.isSigned;
}

inline bool IsE4M3Grid(const QuantSpec &spec)
{
   return spec.encoding == EQuantEncoding::FP8E4M3 && spec.IsPerTensor();
}

// Records why a lowering cannot proceed (SOFIE_QUANT_TRACE prints it) and returns false, so
// the driver pins the operator to float.
inline bool Decline(const QuantContext &ctx, const std::string &reason)
{
   ctx.report.Record("lower", ctx.op.op.Name(), reason);
   return false;
}

// Returns the model behind a context for reading initializer data (RModel's readers are
// non-const).
inline RModel &MutableModel(const QuantContext &ctx)
{
   return const_cast<RModel &>(ctx.Graph().Model());
}

template <typename T>
std::vector<T> ReadConstant(const QuantContext &ctx, const std::string &tensor)
{
   return MutableModel(ctx).GetTensorData<T>(tensor);
}

template <typename T>
std::shared_ptr<void> ToBuffer(const std::vector<T> &values)
{
   std::shared_ptr<void> data(malloc(values.size() * sizeof(T)), free);
   std::memcpy(data.get(), values.data(), values.size() * sizeof(T));
   return data;
}

// Epilogue an arithmetic operator absorbs after its single output `start`:
//   [Encode -> Decode on one grid]  (the output's fake quant)   then
//   Relu | Clip (constant bounds), possibly both,
// each the sole reader of the previous tensor. Nothing is absorbed without a Relu or Clip:
// a bare boundary pair is a form decision, not an absorption.
struct AbsorbedEpilogue {
   QuantAbsorption absorption;
   std::optional<QuantSpec> snap;
   bool relu = false;
   std::optional<std::pair<double, double>> clamp;
};
std::optional<AbsorbedEpilogue> FindAbsorbableEpilogue(const QuantContext &ctx, const std::string &start);

// Returns the store plan for an absorbing operator's output: the absorbed steps, then the
// decided form.
QuantStorePlan StorePlanFor(const QuantBinding &output, const std::optional<AbsorbedEpilogue> &epilogue);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_HANDLERSUPPORT
