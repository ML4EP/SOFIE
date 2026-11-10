#include "SOFIE/quantization/handlers/RQuantization_Handlers.hxx"

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/quantization/handlers/RQuantization_HandlerSupport.hxx"

namespace SOFIE {

// Accepts codes on input 0, decoded at the load, when the operator's hook supports it. The
// hooks implement the decode for per-tensor E4M3 operands only.
QuantForms FusibleFloatHandler::AcceptedInputForms(const QuantContext &ctx, std::size_t input) const
{
   const auto &state = ctx.op.InputState(input);
   if (input != 0 || !state.HasGrid() || !IsE4M3Grid(*state.spec) || !ctx.op.op.CanFuseDequantizedInput())
      return {EQuantForm::Float};
   return {EQuantForm::Codes, EQuantForm::Float};
}

// Offers encode (Codes) or snap (Snapped) at the store, as far as the operator's hook allows.
QuantForms FusibleFloatHandler::OfferedOutputForms(const QuantContext &ctx, std::size_t output,
                                                   const QuantSpec &target) const
{
   if (output != 0 || !IsKernelGrid(target))
      return {EQuantForm::Float};
   QuantForms forms;
   if (ctx.op.op.CanFuseOutputOnGrid(EQuantizedOutputEmit::Carrier))
      forms.push_back(EQuantForm::Codes);
   if (ctx.op.op.CanFuseOutputOnGrid(EQuantizedOutputEmit::Snap))
      forms.push_back(EQuantForm::Snapped);
   forms.push_back(EQuantForm::Float);
   return forms;
}

bool FusibleFloatHandler::Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const
{
   if (decision.absorption || decision.outputs.size() != 1)
      return false;
   const std::string original = ctx.op.Input(0);
   const auto &input = decision.inputs[0];
   const bool decodeInput = input.form == EQuantForm::Codes;
   if (decodeInput && (!input.spec || !ctx.op.op.CanFuseDequantizedInput()))
      return false;
   for (std::size_t i = 1; i < decision.inputs.size(); ++i)
      if (decision.inputs[i].form == EQuantForm::Codes)
         return false;
   const auto &output = decision.outputs[0];
   const bool storeCodes = output.form == EQuantForm::Codes, storeSnapped = output.form == EQuantForm::Snapped;
   if ((storeCodes && !ctx.op.op.CanFuseOutputOnGrid(EQuantizedOutputEmit::Carrier)) ||
       (storeSnapped && !ctx.op.op.CanFuseOutputOnGrid(EQuantizedOutputEmit::Snap)))
      return false;

   const auto inputGrid = decodeInput ? ToQuantizationGrid(*input.spec) : QuantizationGrid{};
   const auto outputGrid = (storeCodes || storeSnapped) ? ToQuantizationGrid(*output.spec) : QuantizationGrid{};
   const std::string carrier = input.tensor, target = output.tensor;
   rewriter.Mutate(ctx.op.index, [=](ROperator &op) {
      if (decodeInput && !op.FuseDequantizedInput(original, carrier, inputGrid))
         throw std::runtime_error("SOFIE quantization: " + op.Name() + " refused to decode " + carrier + " at its load");
      if (storeCodes)
         op.FuseOutputOnGrid(target, outputGrid, EQuantizedOutputEmit::Carrier);
      else if (storeSnapped)
         op.FuseOutputOnGrid(target, outputGrid, EQuantizedOutputEmit::Snap);
   });
   return true;
}

} // namespace SOFIE
