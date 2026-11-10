#ifndef SOFIE_RQUANTIZATION_HANDLERS
#define SOFIE_RQUANTIZATION_HANDLERS

// Built-in quantization handlers, grouped by role. Generic handlers serve every operator of a
// role; family handlers lower their operators to quantized operators.

#include "SOFIE/quantization/handlers/RQuantization_Handler.hxx"

namespace SOFIE {

// ---- generic --------------------------------------------------------------------------

// Fallback handler: reads and writes plain floats, exactly as code generation does without the
// workflow. Serves every operator without a dedicated handler.
class FloatOnlyHandler final : public IQuantOpHandler {
public:
   const char *Name() const override { return "float-only"; }
};

// Moves elements without reading them, so codes pass through unchanged on the same grid.
// `carriesCodes` marks operators whose implementation can be repointed at a carrier
// (ROperator::RewireLowPrecisionCarrier); the others forward only the grid annotation.
class ValuePreservingHandler final : public IQuantOpHandler {
public:
   explicit ValuePreservingHandler(bool carriesCodes) : fCarriesCodes(carriesCodes) {}
   const char *Name() const override { return fCarriesCodes ? "value-preserving (codes)" : "value-preserving"; }
   void InferOutputStates(const QuantContext &ctx, std::vector<TensorQuantState> &outputs) const override;
   QuantForms AcceptedInputForms(const QuantContext &ctx, std::size_t input) const override;
   QuantForms OfferedOutputForms(const QuantContext &ctx, std::size_t output, const QuantSpec &target) const override;
   bool Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const override;

private:
   bool fCarriesCodes;
};

// Keeps values on their grid (Relu: max(x, 0) with 0 = zero point; Clip with on-grid
// bounds). Runs in float itself; an arithmetic producer absorbs it into its epilogue.
class GridPreservingHandler final : public IQuantOpHandler {
public:
   const char *Name() const override { return "grid-preserving"; }
   void InferOutputStates(const QuantContext &ctx, std::vector<TensorQuantState> &outputs) const override;
};

// Float operators that decode an E4M3 operand at their load and encode their result onto a
// grid at their store (ROperator::FuseDequantizedInput / FuseOutputOnGrid): Softmax, LayerNorm.
class FusibleFloatHandler final : public IQuantOpHandler {
public:
   const char *Name() const override { return "fusible-float"; }
   QuantForms AcceptedInputForms(const QuantContext &ctx, std::size_t input) const override;
   QuantForms OfferedOutputForms(const QuantContext &ctx, std::size_t output, const QuantSpec &target) const override;
   bool Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const override;
};

// ---- families -------------------------------------------------------------------------

// Elementwise Add / Mul: operands in any form (codes decoded on their grid), any per-tensor
// output grid. Lowered to ROperator_QuantizedElementwise (an Alpaka kernel).
class ElementwiseHandler final : public IQuantOpHandler {
public:
   explicit ElementwiseHandler(bool multiply) : fMultiply(multiply) {}
   const char *Name() const override { return fMultiply ? "elementwise-mul" : "elementwise-add"; }
   QuantForms AcceptedInputForms(const QuantContext &ctx, std::size_t input) const override;
   QuantForms OfferedOutputForms(const QuantContext &ctx, std::size_t output, const QuantSpec &target) const override;
   bool Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const override;

private:
   bool fMultiply;
};

// Gather from a quantized constant table: the gathered codes stay on the table's grid.
// Lowered to ROperator_QuantizedGather.
class GatherTableHandler final : public IQuantOpHandler {
public:
   const char *Name() const override { return "gather-table"; }
   void InferOutputStates(const QuantContext &ctx, std::vector<TensorQuantState> &outputs) const override;
   QuantForms AcceptedInputForms(const QuantContext &ctx, std::size_t input) const override;
   QuantForms OfferedOutputForms(const QuantContext &ctx, std::size_t output, const QuantSpec &target) const override;
   bool Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const override;
};

// Gemm and MatMul (MatMul is parsed into a Gemm): int8 x int8 through sofieBLAS int8Matmul
// (int8 output with the library epilogue where it applies), E4M3 x E4M3 through
// matmulFp8Fp32, strided-batched for two runtime operands. Absorbs a following
// [encode, decode,] Relu / Clip into its epilogue. Lowered to ROperator_QuantizedDenseLinear.
class DenseLinearHandler final : public IQuantOpHandler {
public:
   const char *Name() const override { return "dense-linear"; }
   std::optional<QuantAbsorption> AbsorbConsumers(const QuantContext &ctx) const override;
   QuantForms AcceptedInputForms(const QuantContext &ctx, std::size_t input) const override;
   QuantForms OfferedOutputForms(const QuantContext &ctx, std::size_t output, const QuantSpec &target) const override;
   bool Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const override;
};

// Conv (standard, grouped, depthwise): im2col + sofieBLAS int8Matmul / matmulFp8Fp32 per
// group, or a direct kernel for depthwise and asymmetric cases. Absorbs a following
// [encode, decode,] Relu / Clip. Lowered to ROperator_QuantizedConv.
class ConvHandler final : public IQuantOpHandler {
public:
   const char *Name() const override { return "conv"; }
   std::optional<QuantAbsorption> AbsorbConsumers(const QuantContext &ctx) const override;
   QuantForms AcceptedInputForms(const QuantContext &ctx, std::size_t input) const override;
   QuantForms OfferedOutputForms(const QuantContext &ctx, std::size_t output, const QuantSpec &target) const override;
   bool Lower(const QuantContext &ctx, const LoweringDecision &decision, QuantRewriter &rewriter) const override;
};

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_HANDLERS
