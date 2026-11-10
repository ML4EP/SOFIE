#include "SOFIE/quantization/handlers/RQuantization_Handlers.hxx"

#include "SOFIE/ROperator_BasicBinary.hxx"
#include "SOFIE/ROperator_Clip.hxx"
#include "SOFIE/ROperator_Conv.hxx"
#include "SOFIE/ROperator_Gather.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_Identity.hxx"
#include "SOFIE/ROperator_LayerNormalization.hxx"
#include "SOFIE/ROperator_Relu.hxx"
#include "SOFIE/ROperator_Reshape.hxx"
#include "SOFIE/ROperator_Softmax.hxx"
#include "SOFIE/ROperator_Transpose.hxx"

namespace SOFIE {

const IQuantOpHandler &QuantOpHandlerRegistry::For(const ROperator &op) const
{
   static const FloatOnlyHandler kFloatOnly;
   auto it = fHandlers.find(std::type_index(typeid(op)));
   return it == fHandlers.end() ? kFloatOnly : *it->second;
}

// Returns the built-in registry, the single mapping from operator types to their handlers.
const QuantOpHandlerRegistry &QuantOpHandlerRegistry::Default()
{
   static const QuantOpHandlerRegistry registry = [] {
      QuantOpHandlerRegistry r;

      // Movement: codes pass through unchanged on the same grid.
      auto movesCodes = std::make_shared<ValuePreservingHandler>(true);
      r.Register<ROperator_Reshape>(movesCodes);   // Reshape, Flatten, Squeeze, Unsqueeze
      r.Register<ROperator_Transpose<float>>(movesCodes);
      r.Register<ROperator_Identity<float>>(std::make_shared<ValuePreservingHandler>(false));

      // Grid-preserving: candidates for absorption into an arithmetic producer.
      auto keepsGrid = std::make_shared<GridPreservingHandler>();
      r.Register<ROperator_Relu<float>>(keepsGrid);
      r.Register<ROperator_Clip<float>>(keepsGrid);

      // Float operators that decode an E4M3 operand at their load or encode at their store.
      auto fusible = std::make_shared<FusibleFloatHandler>();
      r.Register<ROperator_Softmax>(fusible);
      r.Register<ROperator_LayerNormalization<float>>(fusible);

      // Arithmetic families, lowered to quantized operators.
      r.Register<ROperator_Gemm<float>>(std::make_shared<DenseLinearHandler>());
      r.Register<ROperator_Conv<float>>(std::make_shared<ConvHandler>());
      r.Register<ROperator_BasicBinary<float, EBasicBinaryOperator::Add>>(std::make_shared<ElementwiseHandler>(false));
      r.Register<ROperator_BasicBinary<float, EBasicBinaryOperator::Mul>>(std::make_shared<ElementwiseHandler>(true));
      r.Register<ROperator_Gather>(std::make_shared<GatherTableHandler>());
      return r;
   }();
   return registry;
}

} // namespace SOFIE
