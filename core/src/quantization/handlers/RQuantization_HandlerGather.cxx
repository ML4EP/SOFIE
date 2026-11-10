#include "SOFIE/quantization/handlers/RQuantization_Handlers.hxx"

#include "SOFIE/ROperator_Gather.hxx"
#include "SOFIE/quantization/handlers/RQuantization_HandlerSupport.hxx"
#include "SOFIE/quantization/lowered/ROperator_QuantizedGather.hxx"

namespace SOFIE {

namespace {

// Returns whether `state` is a table this lowering reads: constant, integer of at most 8 bits
// or E4M3, per-tensor or symmetric per-channel (one scale per channel, decoded per element).
bool IsGatherTable(const TensorQuantState &state)
{
   if (!state.HasGrid() || !state.constant)
      return false;
   const auto &spec = *state.spec;
   if (spec.IsPerTensor())
      return IsKernelGrid(spec);
   for (std::size_t c = 0; c < spec.ChannelCount(); ++c)
      if (spec.ZeroPoint(c) != 0)
         return false;
   return spec.encoding == EQuantEncoding::Integer && spec.bitWidth <= 8;
}

std::size_t GatherAxis(const QuantContext &ctx)
{
   const auto &gather = dynamic_cast<const ROperator_Gather &>(ctx.op.op);
   const auto rank = static_cast<std::int64_t>(ctx.Graph().Shape(ctx.op.Input(0)).size());
   const auto axis = gather.GetAxis();
   return static_cast<std::size_t>(axis < 0 ? axis + rank : axis);
}

} // namespace

// Propagates a per-tensor table's grid to the output, since a gather only moves table elements.
void GatherTableHandler::InferOutputStates(const QuantContext &ctx, std::vector<TensorQuantState> &outputs) const
{
   const auto &table = ctx.op.InputState(0);
   if (outputs.empty() || !table.HasGrid() || !table.spec->IsPerTensor())
      return;
   outputs[0].spec = table.spec;
   outputs[0].form = table.form == EQuantForm::Codes ? EQuantForm::Codes : EQuantForm::Snapped;
}

QuantForms GatherTableHandler::AcceptedInputForms(const QuantContext &ctx, std::size_t input) const
{
   if (input != 0 || !IsGatherTable(ctx.op.InputState(0)))
      return {EQuantForm::Float};
   return {EQuantForm::Codes, EQuantForm::Float};
}

QuantForms GatherTableHandler::OfferedOutputForms(const QuantContext &ctx, std::size_t, const QuantSpec &target) const
{
   if (ctx.negotiation == nullptr || !IsKernelGrid(target))
      return {EQuantForm::Float};
   const auto &table = ctx.negotiation->Binding(ctx.op.Input(0));
   if (table.form != EQuantForm::Codes)
      return {EQuantForm::Float};
   return {EQuantForm::Codes, EQuantForm::Snapped, EQuantForm::Float};
}

bool GatherTableHandler::Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const
{
   if (decision.inputs.size() != 2 || decision.outputs.size() != 1 || decision.inputs[0].form != EQuantForm::Codes ||
       !decision.inputs[0].spec)
      return false;
   const auto &graph = ctx.Graph();
   const auto &table = decision.inputs[0];
   const auto tableShape = graph.Shape(table.tensor);
   const auto axis = GatherAxis(ctx);
   std::string channelScales;
   std::size_t channelStride = 1, channelCount = 1;
   if (!table.spec->IsPerTensor()) {
      channelScales = table.tensor + "_channel_scales";
      channelCount = table.spec->ChannelCount();
      for (std::size_t i = static_cast<std::size_t>(table.spec->axis) + 1; i < tableShape.size(); ++i)
         channelStride *= tableShape[i];
      std::vector<float> scales(channelCount);
      for (std::size_t c = 0; c < channelCount; ++c)
         scales[c] = static_cast<float>(table.spec->Scale(c));
      rewriter.AddConstant(channelScales, ETensorType::FLOAT, {channelCount}, ToBuffer(scales));
   }
   auto op = std::make_unique<ROperator_QuantizedGather>(
      table, ctx.op.Input(1), decision.outputs[0].tensor, QuantStorePlan::For(decision.outputs[0]), tableShape,
      axis, ConvertShapeToLength(graph.Shape(ctx.op.Input(1))), channelScales, channelStride, channelCount);
   op->fName = ctx.op.op.Name();
   rewriter.Replace(ctx.op.index, std::move(op));
   return true;
}

} // namespace SOFIE
