#include "SOFIE/quantization/handlers/RQuantization_Handlers.hxx"

namespace SOFIE {

// ---- ValuePreservingHandler -------------------------------------------------------------

void ValuePreservingHandler::InferOutputStates(const QuantContext &ctx, std::vector<TensorQuantState> &outputs) const
{
   const auto &op = ctx.op.op;
   if (ctx.op.InputCount() == 0 || !op.PropagatesQuantizationMetadata())
      return;
   const auto &source = ctx.op.InputState(0);
   if (!source.HasGrid())
      return;
   for (std::size_t j = 0; j < outputs.size(); ++j) {
      TensorQuantState state = source;
      state.constant = false;
      // A per-channel grid follows its axis through the permutation, or is lost when the
      // axis is folded away; per-tensor grids pass unchanged.
      if (!source.spec->IsPerTensor()) {
         const auto axisMap = op.GetQuantizationMetadataAxisMap(ctx.Graph().Shape(ctx.op.Input(0)),
                                                               ctx.Graph().Shape(ctx.op.Output(j)));
         int mapped = -1;
         for (std::size_t outAxis = 0; outAxis < axisMap.size(); ++outAxis)
            if (axisMap[outAxis] == source.spec->axis)
               mapped = static_cast<int>(outAxis);
         if (mapped < 0)
            continue;
         state.spec->axis = mapped;
      }
      outputs[j] = state;
   }
}

QuantForms ValuePreservingHandler::AcceptedInputForms(const QuantContext &ctx, std::size_t input) const
{
   // Accepts codes only on the data operand, per-tensor, and only when every reader of the
   // result accepts codes too; otherwise the decode merely moves downstream.
   if (!fCarriesCodes || input != 0 || ctx.negotiation == nullptr || ctx.op.OutputCount() != 1)
      return {EQuantForm::Float};
   const auto &state = ctx.op.InputState(0);
   if (!state.HasGrid() || !state.spec->IsPerTensor())
      return {EQuantForm::Float};
   if (!ctx.negotiation->ReadersAccept(ctx.op.Output(0), EQuantForm::Codes, *state.spec))
      return {EQuantForm::Float};
   return {EQuantForm::Codes, EQuantForm::Float};
}

QuantForms ValuePreservingHandler::OfferedOutputForms(const QuantContext &ctx, std::size_t output,
                                                      const QuantSpec &target) const
{
   if (!fCarriesCodes || output != 0 || ctx.negotiation == nullptr)
      return {EQuantForm::Float};
   const auto &in = ctx.negotiation->Binding(ctx.op.Input(0));
   if (in.form == EQuantForm::Codes && in.spec && SameGrid(*in.spec, target))
      return {EQuantForm::Codes, EQuantForm::Float};
   return {EQuantForm::Float};
}

bool ValuePreservingHandler::Lower(const QuantContext &ctx, const LoweringDecision &decision,
                                   QuantRewriter &rewriter) const
{
   if (decision.inputs.empty() || decision.outputs.empty() || decision.inputs[0].form != EQuantForm::Codes ||
       decision.outputs[0].form != EQuantForm::Codes || !decision.inputs[0].spec)
      return false;
   const std::string in = decision.inputs[0].tensor;
   const std::string out = decision.outputs[0].tensor;
   const auto codeType = decision.inputs[0].spec->CodeType();
   const bool aliases = ctx.op.op.CarrierOutputAliasesInput();
   rewriter.Mutate(ctx.op.index, [in, out](ROperator &op) { op.RewireLowPrecisionCarrier(in, out); });
   rewriter.RetypeTensor(out, codeType);
   if (aliases)
      rewriter.Alias(out, in);
   return true;
}

// ---- GridPreservingHandler --------------------------------------------------------------

void GridPreservingHandler::InferOutputStates(const QuantContext &ctx, std::vector<TensorQuantState> &outputs) const
{
   if (ctx.op.InputCount() == 0 || outputs.empty())
      return;
   const auto &source = ctx.op.InputState(0);
   if (!source.HasGrid())
      return;
   // The result stays on the grid as a float, so it is Snapped regardless of the input form.
   outputs[0].spec = source.spec;
   outputs[0].form = EQuantForm::Snapped;
}

} // namespace SOFIE
