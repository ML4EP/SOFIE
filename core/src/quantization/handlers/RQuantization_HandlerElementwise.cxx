#include "SOFIE/quantization/handlers/RQuantization_Handlers.hxx"

#include "SOFIE/quantization/handlers/RQuantization_HandlerSupport.hxx"
#include "SOFIE/quantization/lowered/ROperator_QuantizedElementwise.hxx"

namespace SOFIE {

// Accepts each operand in the form it arrives in; codes on any kernel grid are read
// independently of the other operand.
QuantForms ElementwiseHandler::AcceptedInputForms(const QuantContext &ctx, std::size_t input) const
{
   const auto &state = ctx.op.InputState(input);
   if (ctx.op.InputCount() != 2 || !state.HasGrid() || !IsKernelGrid(*state.spec) ||
       ctx.Graph().Shape(ctx.op.Output(0)).size() > 8)
      return {EQuantForm::Float};
   return {EQuantForm::Codes, EQuantForm::Float};
}

// Offers any kernel grid for the output, since the result is computed in double.
QuantForms ElementwiseHandler::OfferedOutputForms(const QuantContext &ctx, std::size_t, const QuantSpec &target) const
{
   if (ctx.op.InputCount() != 2 || !IsKernelGrid(target) || ctx.Graph().Shape(ctx.op.Output(0)).size() > 8)
      return {EQuantForm::Float};
   return {EQuantForm::Codes, EQuantForm::Snapped, EQuantForm::Float};
}

bool ElementwiseHandler::Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const
{
   if (decision.inputs.size() != 2 || decision.outputs.size() != 1)
      return false;
   const auto &graph = ctx.Graph();
   auto op = std::make_unique<ROperator_QuantizedElementwise>(
      fMultiply ? ROperator_QuantizedElementwise::EOp::Mul : ROperator_QuantizedElementwise::EOp::Add,
      decision.inputs[0], decision.inputs[1], decision.outputs[0].tensor, QuantStorePlan::For(decision.outputs[0]),
      graph.Shape(ctx.op.Input(0)), graph.Shape(ctx.op.Input(1)), graph.Shape(ctx.op.Output(0)));
   op->fName = ctx.op.op.Name();
   rewriter.Replace(ctx.op.index, std::move(op));
   return true;
}

} // namespace SOFIE
