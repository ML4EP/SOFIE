#include "SOFIE/quantization/handlers/RQuantization_Handlers.hxx"

#include "SOFIE/ROperator_Conv.hxx"
#include "SOFIE/quantization/handlers/RQuantization_HandlerSupport.hxx"
#include "SOFIE/quantization/lowered/ROperator_QuantizedConv.hxx"

namespace SOFIE {

namespace {

// Largest im2col column scratch in bytes; a convolution needing more uses the direct path.
constexpr std::size_t kConvColumnBudgetBytes = 256ULL * 1024ULL * 1024ULL;

struct ConvPlan {
   bool fp8 = false;
   QuantizedConvConfig shape;   // geometry only
   QuantSpec x, w;
};

const ROperator_Conv<float> &ConvOf(const QuantContext &ctx)
{
   return dynamic_cast<const ROperator_Conv<float> &>(ctx.op.op);
}

std::optional<ConvPlan> PlanConv(const QuantContext &ctx)
{
   if (ctx.op.InputCount() < 2)
      return std::nullopt;
   const auto &xs = ctx.op.InputState(0), &ws = ctx.op.InputState(1);
   if (!xs.HasGrid() || !ws.HasGrid() || !ws.constant)
      return std::nullopt;
   const auto &conv = ConvOf(ctx);
   const auto &graph = ctx.Graph();
   const auto xShape = graph.Shape(ctx.op.Input(0));
   const auto wShape = graph.Shape(ctx.op.Input(1));
   const auto yShape = graph.Shape(ctx.op.Output(0));
   // Supports Conv1D and Conv2D in NCHW.
   if (xShape.size() < 3 || xShape.size() > 4 || wShape.size() != xShape.size() || yShape.size() != xShape.size())
      return std::nullopt;
   const bool twoD = xShape.size() == 4;
   ConvPlan plan;
   plan.x = *xs.spec;
   plan.w = *ws.spec;
   auto &s = plan.shape;
   s.N = xShape[0];
   s.C = xShape[1];
   s.H = twoD ? xShape[2] : 1;
   s.W = xShape.back();
   s.OC = yShape[1];
   s.OH = twoD ? yShape[2] : 1;
   s.OW = yShape.back();
   s.KH = twoD ? wShape[2] : 1;
   s.KW = wShape.back();
   s.groups = conv.GetGroup();
   const auto &strides = conv.GetStrides(), &dilations = conv.GetDilations(), &pads = conv.GetPads();
   if (s.groups == 0 || s.C % s.groups != 0 || s.OC % s.groups != 0 || wShape[0] != s.OC ||
       wShape[1] * s.groups != s.C || strides.empty() || dilations.empty() || pads.empty())
      return std::nullopt;
   if (twoD) {
      s.strideH = strides[0], s.strideW = strides[1];
      s.dilationH = dilations[0], s.dilationW = dilations[1];
      s.padH = pads[0], s.padW = pads[1];
   } else {
      s.strideW = strides[0], s.dilationW = dilations[0], s.padW = pads[0];
   }

   if (IsE4M3Grid(plan.x) && IsE4M3Grid(plan.w)) {
      plan.fp8 = true;
      return plan;
   }
   // int8: signed per-tensor input (any zero point), weight per-tensor or per output channel.
   if (!IsSignedInt8Grid(plan.x) || !plan.x.IsPerTensor() || !IsSignedInt8Grid(plan.w))
      return std::nullopt;
   if (!plan.w.IsPerTensor() && (plan.w.axis != 0 || plan.w.ChannelCount() != s.OC))
      return std::nullopt;
   return plan;
}

} // namespace

std::optional<QuantAbsorption> ConvHandler::AbsorbConsumers(const QuantContext &ctx) const
{
   if (!PlanConv(ctx))
      return std::nullopt;
   auto epilogue = FindAbsorbableEpilogue(ctx, ctx.op.Output(0));
   if (!epilogue)
      return std::nullopt;
   return epilogue->absorption;
}

QuantForms ConvHandler::AcceptedInputForms(const QuantContext &ctx, std::size_t input) const
{
   // Slots 0 and 1 are the input and the weight; the rest are the bias and the float Conv's
   // own scratch, which stay float.
   if (input > 1 || !PlanConv(ctx))
      return {EQuantForm::Float};
   return {EQuantForm::Codes, EQuantForm::Float};
}

QuantForms ConvHandler::OfferedOutputForms(const QuantContext &ctx, std::size_t, const QuantSpec &target) const
{
   if (ctx.negotiation == nullptr || !IsKernelGrid(target) || !PlanConv(ctx))
      return {EQuantForm::Float};
   if (ctx.negotiation->Binding(ctx.op.Input(0)).form != EQuantForm::Codes ||
       ctx.negotiation->Binding(ctx.op.Input(1)).form != EQuantForm::Codes)
      return {EQuantForm::Float};
   return {EQuantForm::Codes, EQuantForm::Snapped, EQuantForm::Float};
}

bool ConvHandler::Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const
{
   const auto plan = PlanConv(ctx);
   if (!plan)
      return Decline(ctx, "conv: operands no longer form a lowerable plan");
   // The float Conv lists its own im2col scratch among its operands after Initialize; only
   // input, weight, bias (found by name) and output 0 belong to the operator.
   if (decision.inputs.size() < 2 || decision.outputs.empty())
      return Decline(ctx, "conv: unexpected operand count");
   if (decision.inputs[0].form != EQuantForm::Codes || decision.inputs[1].form != EQuantForm::Codes)
      return Decline(ctx, "conv: input or weight is not carried as codes (" + decision.inputs[0].tensor + ", " +
                             decision.inputs[1].tensor + ")");
   const std::string name = ctx.op.Output(0) + "_qconv";
   QuantizedConvConfig config = plan->shape;
   config.fp8 = plan->fp8;
   config.x = decision.inputs[0].tensor;
   config.w = decision.inputs[1].tensor;
   config.y = decision.outputs[0].tensor;
   const std::size_t OC = config.OC, OCg = OC / config.groups;
   const std::size_t Kg = (config.C / config.groups) * config.KH * config.KW;

   std::optional<AbsorbedEpilogue> absorbed;
   if (decision.absorption)
      absorbed = FindAbsorbableEpilogue(ctx, ctx.op.Output(0));
   config.store = StorePlanFor(decision.outputs[0], absorbed);

   // Bias, one per output channel, from a constant.
   const auto &biasName = ConvOf(ctx).GetBiasTensorName();
   std::size_t biasSlot = 0;
   for (std::size_t i = 2; i < ctx.op.InputCount() && !biasName.empty(); ++i)
      if (ctx.op.Input(i) == biasName)
         biasSlot = i;
   if (biasSlot != 0) {
      const auto &b = decision.inputs[biasSlot];
      if (b.form == EQuantForm::Codes || !ctx.Graph().IsConstant(b.tensor))
         return Decline(ctx, "conv bias " + b.tensor + " is not a float constant");
      const auto values = ReadConstant<float>(ctx, b.tensor);
      if (values.size() != OC)
         return Decline(ctx, "conv bias " + b.tensor + " has " + std::to_string(values.size()) +
                                " values for " + std::to_string(OC) + " channels");
      config.bias = name + "_bias";
      rewriter.AddConstant(config.bias, ETensorType::FLOAT, {OC}, ToBuffer(values));
   }

   const double sx = plan->x.Scale();
   bool symmetricWeight = true;
   if (plan->fp8) {
      config.scale = sx * plan->w.Scale();
   } else {
      config.inputZeroPoint = plan->x.ZeroPoint();
      if (plan->w.IsPerTensor()) {
         config.scale = sx * plan->w.Scale();
         symmetricWeight = plan->w.ZeroPoint() == 0;
      } else {
         std::vector<float> scales(OC);
         for (std::size_t c = 0; c < OC; ++c)
            scales[c] = static_cast<float>(sx * plan->w.Scale(c));
         config.weightScales = name + "_scales";
         rewriter.AddConstant(config.weightScales, ETensorType::FLOAT, {OC}, ToBuffer(scales));
      }
      std::vector<std::int32_t> zeroPoints(OC);
      for (std::size_t c = 0; c < OC; ++c) {
         zeroPoints[c] = static_cast<std::int32_t>(plan->w.IsPerTensor() ? plan->w.ZeroPoint() : plan->w.ZeroPoint(c));
         symmetricWeight &= zeroPoints[c] == 0;
      }
      if (!symmetricWeight) {
         config.weightZeroPoints = name + "_wzp";
         rewriter.AddConstant(config.weightZeroPoints, ETensorType::INT32, {OC}, ToBuffer(zeroPoints));
      }
   }

   // Uses im2col + library GEMM for everything but depthwise, with symmetric weights, within
   // budget. The window is padded to the vendor alignment so the fast kernels stay available;
   // sofieBLAS runs an unaligned output plane (the GEMM's m) on its native fallback kernels.
   const std::size_t alignment = plan->fp8 ? 16 : 4;
   const std::size_t Kp = (Kg + alignment - 1) / alignment * alignment;
   const std::size_t columnBytes = config.N * config.groups * config.OH * config.OW * Kp;
   const bool depthwise = OCg == 1 && Kg == config.KH * config.KW;
   if (symmetricWeight && !depthwise && columnBytes <= kConvColumnBudgetBytes) {
      config.Kp = Kp;
      // Weights as [groups, OCg, Kp] codes, zero beyond Kg (contributes nothing).
      const auto weights = ReadConstant<std::uint8_t>(ctx, config.w);   // raw bytes, int8 or E4M3
      std::vector<std::uint8_t> matrix(OC * Kp, 0);
      std::vector<std::int32_t> columnSums(OC, 0);
      for (std::size_t oc = 0; oc < OC; ++oc)
         for (std::size_t kk = 0; kk < Kg; ++kk) {
            matrix[oc * Kp + kk] = weights[oc * Kg + kk];
            columnSums[oc] += static_cast<std::int8_t>(weights[oc * Kg + kk]);
         }
      config.weightMatrix = name + "_matrix";
      rewriter.AddConstant(config.weightMatrix, plan->fp8 ? ETensorType::FLOAT8E4M3FN : ETensorType::INT8,
                           {config.groups, OCg, Kp}, ToBuffer(matrix));
      if (!plan->fp8 && config.inputZeroPoint != 0) {
         config.columnSums = name + "_colsum";
         rewriter.AddConstant(config.columnSums, ETensorType::INT32, {OC}, ToBuffer(columnSums));
      }
      config.columns = name + "_col";
      config.accumulator = name + "_acc";
      rewriter.AddScratch(config.columns, plan->fp8 ? ETensorType::FLOAT8E4M3FN : ETensorType::INT8, {columnBytes});
      rewriter.AddScratch(config.accumulator, plan->fp8 ? ETensorType::FLOAT : ETensorType::INT32,
                          {config.N * OC * config.OH * config.OW});
      if (plan->fp8) {
         config.scaleX = name + "_sx";
         config.scaleW = name + "_sw";
         rewriter.AddConstant(config.scaleX, ETensorType::FLOAT, {1}, ToBuffer(std::vector<float>{static_cast<float>(sx)}));
         rewriter.AddConstant(config.scaleW, ETensorType::FLOAT, {1},
                              ToBuffer(std::vector<float>{static_cast<float>(plan->w.Scale())}));
      }
   }

   auto op = std::make_unique<ROperator_QuantizedConv>(std::move(config));
   op->fName = ctx.op.op.Name();
   rewriter.Replace(ctx.op.index, std::move(op));
   return true;
}

} // namespace SOFIE
