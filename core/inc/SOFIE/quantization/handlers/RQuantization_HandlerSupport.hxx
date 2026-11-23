#ifndef SOFIE_RQUANTIZATION_HANDLERSUPPORT
#define SOFIE_RQUANTIZATION_HANDLERSUPPORT

// Helpers shared by the family handlers that need a QuantContext: declining a lowering, reading constants through the context, and the epilogue absorption
// walk after an arithmetic operator. Context-free helpers live in RQuantization_Utils.hxx.

#include "SOFIE/RModel.hxx"
#include "SOFIE/quantization/handlers/RQuantization_Handler.hxx"
#include "SOFIE/quantization/lowered/ROperator_QuantLowered.hxx"
#include "SOFIE/quantization/spec/RQuantization_CodeFormat.hxx"
#include "SOFIE/quantization/utils/RQuantization_Utils.hxx"

#include <optional>
#include <string>
#include <vector>

namespace SOFIE {

// Records why a lowering cannot proceed and returns false, so
// the driver pins the operator to float.
inline bool Decline(const QuantContext &ctx, const std::string &reason)
{
   ctx.report.Record("lower", ctx.op.op.Name(), reason);
   return false;
}

// Returns the model behind a context for reading initializer data
inline RModel &MutableModel(const QuantContext &ctx)
{
   return const_cast<RModel &>(ctx.Graph().Model());
}

template <typename T>
std::vector<T> ReadConstant(const QuantContext &ctx, const std::string &tensor)
{
   return MutableModel(ctx).GetTensorData<T>(tensor);
}

// Returns a constant's codes as whole numbers or raw bytes, whatever
// their storage type.
inline std::vector<double> ReadCodes(const QuantContext &ctx, const std::string &tensor)
{
   return ReadCodes(MutableModel(ctx), tensor);
}

// Epilogue an arithmetic operator absorbs after its single output `start`:
//   [Encode -> Decode on one grid]  (the output's fake quant)   then
//   Relu | Clip (constant bounds), possibly both,
// each the sole reader of the previous tensor. Nothing is absorbed without a Relu or Clip:
// a bare boundary pair is a form decision.
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
