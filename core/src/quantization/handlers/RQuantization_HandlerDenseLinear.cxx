#include "SOFIE/quantization/handlers/RQuantization_Handlers.hxx"

#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/quantization/handlers/RQuantization_HandlerSupport.hxx"
#include "SOFIE/quantization/lowered/ROperator_QuantizedDenseLinear.hxx"

namespace SOFIE {

namespace {

// Plan derived from the graph and shared by the Decide queries and Lower, so every form
// offered to Decide is one Lower can build.
struct DensePlan {
   bool fp8 = false;
   std::size_t batch = 1, M = 0, N = 0, K = 0;
   bool weightConstant = false;
   bool weightNK = true;   // weight stored as [.., N, K]; a constant [K, N] is re-laid out
   QuantSpec x, w;
};

const ROperator_Gemm<float> &GemmOf(const QuantContext &ctx)
{
   return dynamic_cast<const ROperator_Gemm<float> &>(ctx.op.op);
}

std::optional<DensePlan> PlanDense(const QuantContext &ctx)
{
   const auto &gemm = GemmOf(ctx);
   if (ctx.op.InputCount() < 2 || gemm.GetTransA() != 0)
      return std::nullopt;
   const auto &xs = ctx.op.InputState(0), &ws = ctx.op.InputState(1);
   if (!xs.HasGrid() || !ws.HasGrid())
      return std::nullopt;
   DensePlan plan;
   plan.x = *xs.spec;
   plan.w = *ws.spec;
   const auto &graph = ctx.Graph();
   const auto xShape = graph.Shape(ctx.op.Input(0));
   const auto wShape = graph.Shape(ctx.op.Input(1));
   if (xShape.size() < 2 || wShape.size() < 2)
      return std::nullopt;
   plan.weightConstant = ws.constant;
   const bool transB = gemm.GetTransB() != 0;
   plan.K = xShape.back();
   if (plan.weightConstant) {
      // A [K, N] or [N, K] weight shared by every row of X (leading X dims folded into M).
      if (wShape.size() != 2)
         return std::nullopt;
      plan.weightNK = transB;
      plan.N = transB ? wShape[0] : wShape[1];
      if ((transB ? wShape[1] : wShape[0]) != plan.K)
         return std::nullopt;
      plan.M = ConvertShapeToLength(xShape) / plan.K;
   } else {
      // Two runtime operands: a batched product with B already [.., N, K] (the batched
      // MatMul canonicalisation re-permutes its Transpose and sets transB).
      if (!transB || wShape.size() != xShape.size())
         return std::nullopt;
      for (std::size_t i = 0; i + 2 < xShape.size(); ++i)
         if (xShape[i] != wShape[i])
            return std::nullopt;
      plan.M = xShape[xShape.size() - 2];
      plan.N = wShape[wShape.size() - 2];
      if (wShape.back() != plan.K)
         return std::nullopt;
      plan.batch = ConvertShapeToLength(xShape) / (plan.M * plan.K);
   }

   if (IsE4M3Grid(plan.x) && IsE4M3Grid(plan.w)) {
      plan.fp8 = true;
      return plan;
   }
   // int8: signed codes on both sides, the weight symmetric, per-tensor or per output channel.
   if (!IsSignedInt8Grid(plan.x) || !plan.x.IsPerTensor() || !IsSignedInt8Grid(plan.w))
      return std::nullopt;
   for (std::size_t c = 0; c < std::max<std::size_t>(plan.w.ChannelCount(), 1); ++c)
      if (plan.w.ZeroPoint(c) != 0)
         return std::nullopt;
   if (!plan.w.IsPerTensor()) {
      const int outputAxis = plan.weightNK ? 0 : 1;
      if (!plan.weightConstant || plan.w.axis != outputAxis || plan.w.ChannelCount() != plan.N)
         return std::nullopt;
   }
   // The input zero point is corrected through the weight's column sums, a constant.
   if (plan.x.ZeroPoint() != 0 && !plan.weightConstant)
      return std::nullopt;
   return plan;
}

std::string FusedEpilogueName(bool bias, bool relu)
{
   if (bias)
      return relu ? "Epilogue::ReluBias" : "Epilogue::Bias";
   return relu ? "Epilogue::Relu" : "Epilogue::Default";
}

// Returns the weight codes as [N, K], re-laid out from a [K, N] constant when needed.
template <typename T>
std::vector<T> WeightNK(const QuantContext &ctx, const std::string &tensor, const DensePlan &plan)
{
   auto codes = ReadConstant<T>(ctx, tensor);
   if (plan.weightNK)
      return codes;
   std::vector<T> nk(codes.size());
   for (std::size_t k = 0; k < plan.K; ++k)
      for (std::size_t n = 0; n < plan.N; ++n)
         nk[n * plan.K + k] = codes[k * plan.N + n];
   return nk;
}

} // namespace

std::optional<QuantAbsorption> DenseLinearHandler::AbsorbConsumers(const QuantContext &ctx) const
{
   if (!PlanDense(ctx))
      return std::nullopt;
   auto epilogue = FindAbsorbableEpilogue(ctx, ctx.op.Output(0));
   if (!epilogue)
      return std::nullopt;
   return epilogue->absorption;
}

QuantForms DenseLinearHandler::AcceptedInputForms(const QuantContext &ctx, std::size_t input) const
{
   if (input > 1 || !PlanDense(ctx))
      return {EQuantForm::Float};
   return {EQuantForm::Codes, EQuantForm::Float};
}

QuantForms DenseLinearHandler::OfferedOutputForms(const QuantContext &ctx, std::size_t, const QuantSpec &target) const
{
   if (ctx.negotiation == nullptr || !IsKernelGrid(target) || !PlanDense(ctx))
      return {EQuantForm::Float};
   if (ctx.negotiation->Binding(ctx.op.Input(0)).form != EQuantForm::Codes ||
       ctx.negotiation->Binding(ctx.op.Input(1)).form != EQuantForm::Codes)
      return {EQuantForm::Float};
   return {EQuantForm::Codes, EQuantForm::Snapped, EQuantForm::Float};
}

bool DenseLinearHandler::Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const
{
   const auto plan = PlanDense(ctx);
   if (!plan)
      return Decline(ctx, "dense-linear: operands no longer form a lowerable plan");
   if (decision.inputs.size() < 2 || decision.outputs.size() != 1)
      return Decline(ctx, "dense-linear: unexpected operand count");
   if (decision.inputs[0].form != EQuantForm::Codes || decision.inputs[1].form != EQuantForm::Codes)
      return Decline(ctx, "dense-linear: input or weight is not carried as codes (" + decision.inputs[0].tensor + ", " +
                             decision.inputs[1].tensor + ")");
   const auto &gemm = GemmOf(ctx);
   const std::string name = ctx.op.Output(0) + "_qdense";
   const double alpha = gemm.GetAlpha();
   if (plan->fp8 && alpha != 1.0)
      return Decline(ctx, "float8 Gemm with alpha != 1");

   QuantizedDenseLinearConfig config;
   config.fp8 = plan->fp8;
   config.batch = plan->batch;
   config.M = plan->M;
   config.N = plan->N;
   config.K = plan->K;
   config.strideX = static_cast<long long>(plan->M * plan->K);
   config.strideW = plan->batch > 1 ? static_cast<long long>(plan->N * plan->K) : 0;
   config.strideY = static_cast<long long>(plan->M * plan->N);
   config.x = decision.inputs[0].tensor;
   config.y = decision.outputs[0].tensor;
   // The vendor kernels require K and N aligned to 4 (int8), or K to 16 and N to 4 (float8).
   // sofieBLAS runs other shapes correctly on its native kernels, which are much slower; the
   // report records this to explain a slow layer.
   if (plan->fp8 ? (plan->K % 16 != 0 || plan->N % 4 != 0) : (plan->K % 4 != 0 || plan->N % 4 != 0))
      ctx.report.Record("lower", ctx.op.op.Name(),
                        "K=" + std::to_string(plan->K) + " N=" + std::to_string(plan->N) +
                           " are not vendor-aligned; sofieBLAS may run this layer on its native kernels");

   // The weight as [N, K] codes.
   const std::string weight = decision.inputs[1].tensor;
   config.w = weight;
   std::vector<std::int64_t> columnSums;
   if (plan->weightConstant) {
      if (plan->fp8) {
         auto nk = WeightNK<std::uint8_t>(ctx, weight, *plan);
         if (!plan->weightNK) {
            config.w = weight + "_nk";
            rewriter.AddConstant(config.w, ETensorType::FLOAT8E4M3FN, {plan->N, plan->K}, ToBuffer(nk));
         }
      } else {
         auto nk = WeightNK<std::int8_t>(ctx, weight, *plan);
         if (plan->x.ZeroPoint() != 0) {
            columnSums.assign(plan->N, 0);
            for (std::size_t n = 0; n < plan->N; ++n)
               for (std::size_t k = 0; k < plan->K; ++k)
                  columnSums[n] += nk[n * plan->K + k];
         }
         if (!plan->weightNK) {
            config.w = weight + "_nk";
            rewriter.AddConstant(config.w, ETensorType::INT8, {plan->N, plan->K}, ToBuffer(nk));
         }
      }
   }

   // Bias: beta * C, one value per output column, from a constant.
   std::vector<double> bias;
   if (decision.inputs.size() > 2 && !ctx.op.Input(2).empty()) {
      const auto &c = decision.inputs[2];
      if (c.form == EQuantForm::Codes || !ctx.Graph().IsConstant(c.tensor))
         return Decline(ctx, "bias " + c.tensor + " is not a float constant");
      const auto values = ReadConstant<float>(ctx, c.tensor);
      if (values.size() != plan->N)
         return Decline(ctx, "bias " + c.tensor + " is not one value per output column");
      bias.resize(plan->N);
      for (std::size_t n = 0; n < plan->N; ++n)
         bias[n] = static_cast<double>(gemm.GetBeta()) * values[n];
   }

   // The epilogue: what this Gemm absorbed, its own fused activation, the decided form.
   std::optional<AbsorbedEpilogue> absorbed;
   if (decision.absorption)
      absorbed = FindAbsorbableEpilogue(ctx, ctx.op.Output(0));
   config.store = StorePlanFor(decision.outputs[0], absorbed);
   if (gemm.GetActivationType() == EActivationType::RELU) {
      if (config.store.snapBefore || config.store.clamp)
         return Decline(ctx, "a parse-fused Relu before absorbed steps");
      config.store.relu = true;
   } else if (gemm.GetActivationType() != EActivationType::UNDEFINED) {
      return Decline(ctx, "fused activation other than Relu");
   }

   // Per-element scale: the library applies the float8 scales; the int8 scales are folded here.
   const double sx = plan->x.Scale();
   auto weightScale = [&](std::size_t n) { return plan->w.IsPerTensor() ? plan->w.Scale() : plan->w.Scale(n); };
   if (plan->fp8) {
      config.scale = 1.0;
      config.scaleX = name + "_sx";
      config.scaleW = name + "_sw";
      rewriter.AddConstant(config.scaleX, ETensorType::FLOAT, {1}, ToBuffer(std::vector<float>{static_cast<float>(sx)}));
      rewriter.AddConstant(config.scaleW, ETensorType::FLOAT, {1},
                           ToBuffer(std::vector<float>{static_cast<float>(plan->w.Scale())}));
   } else if (plan->w.IsPerTensor()) {
      config.scale = alpha * sx * plan->w.Scale();
   } else {
      std::vector<float> scales(plan->N);
      for (std::size_t n = 0; n < plan->N; ++n)
         scales[n] = static_cast<float>(alpha * sx * weightScale(n));
      config.weightScales = name + "_scales";
      rewriter.AddConstant(config.weightScales, ETensorType::FLOAT, {plan->N}, ToBuffer(scales));
   }
   if (!columnSums.empty()) {
      std::vector<std::int32_t> sums(columnSums.begin(), columnSums.end());
      config.inputZeroPoint = plan->x.ZeroPoint();
      config.columnSums = name + "_colsum";
      rewriter.AddConstant(config.columnSums, ETensorType::INT32, {plan->N}, ToBuffer(sums));
   }
   if (!bias.empty()) {
      std::vector<float> values(bias.begin(), bias.end());
      config.bias = name + "_bias";
      rewriter.AddConstant(config.bias, ETensorType::FLOAT, {plan->N}, ToBuffer(values));
   }

   // The int8 library epilogue, when the stored value is exactly its arithmetic.
   if (!plan->fp8 && config.store.IsPlainInt8Codes() && (plan->batch == 1 || plan->w.IsPerTensor())) {
      const double sy = config.store.spec->Scale();
      const bool hasOffset = !bias.empty() || !columnSums.empty();
      if (hasOffset) {
         std::vector<float> offset(plan->N);
         for (std::size_t n = 0; n < plan->N; ++n) {
            const double correction = columnSums.empty() ? 0.0
                                                         : alpha * sx * weightScale(n) *
                                                              static_cast<double>(plan->x.ZeroPoint()) *
                                                              static_cast<double>(columnSums[n]);
            offset[n] = static_cast<float>(((bias.empty() ? 0.0 : bias[n]) - correction) / sy);
         }
         config.fusedBias = name + "_fused_bias";
         rewriter.AddConstant(config.fusedBias, ETensorType::FLOAT, {plan->N}, ToBuffer(offset));
      }
      if (plan->w.IsPerTensor()) {
         config.fusedAlpha = alpha * sx * plan->w.Scale() / sy;
      } else {
         std::vector<float> alphas(plan->N);
         for (std::size_t n = 0; n < plan->N; ++n)
            alphas[n] = static_cast<float>(alpha * sx * weightScale(n) / sy);
         config.fusedAlphaVector = name + "_fused_alpha";
         rewriter.AddConstant(config.fusedAlphaVector, ETensorType::FLOAT, {plan->N}, ToBuffer(alphas));
      }
      config.fusedEpilogue = FusedEpilogueName(hasOffset, config.store.relu);
   }

   // The accumulator for the epilogue kernel, unless the library call writes the output itself.
   if (config.fusedEpilogue.empty()) {
      config.accumulator = name + "_acc";
      rewriter.AddScratch(config.accumulator, plan->fp8 ? ETensorType::FLOAT : ETensorType::INT32,
                          {plan->batch * plan->M * plan->N});
   }

   auto op = std::make_unique<ROperator_QuantizedDenseLinear>(std::move(config));
   op->fName = ctx.op.op.Name();
   rewriter.Replace(ctx.op.index, std::move(op));
   return true;
}

} // namespace SOFIE
