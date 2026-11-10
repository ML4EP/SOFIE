#ifndef SOFIE_RQUANTIZATION_HANDLER
#define SOFIE_RQUANTIZATION_HANDLER

// Per-operator quantization handlers. Each operator type maps to one handler, registered by
// C++ type in the QuantOpHandlerRegistry. The workflow passes query the handler of the
// operator in focus once per phase:
//
//   Annotate  InferOutputStates   Infers the grid and form of each output.
//   Decide    AbsorbConsumers     Selects the following operators the operator folds into itself.
//             AcceptedInputForms  Lists the forms in which the operator reads input i.
//             OfferedOutputForms  Lists the forms in which the operator writes output j onto a grid.
//   Lower     Lower               Builds or rewires the operator for the decided forms.
//
// A handler sees the operator in focus (QuantOpView), the whole graph (QuantGraphView) and a
// negotiation oracle that reports on its neighbours. It never edits the model; Lower proposes
// edits through a QuantRewriter. An operator with no registered handler gets FloatOnlyHandler.

#include "SOFIE/ROperator.hxx"
#include "SOFIE/quantization/diagnostics/RQuantization_Diagnostics.hxx"
#include "SOFIE/quantization/graph/RQuantization_Graph.hxx"
#include "SOFIE/quantization/spec/RQuantization_Spec.hxx"

#include <memory>
#include <optional>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace SOFIE {

// Physical carrier of a logical tensor: the tensor that holds it, its form and its grid.
// For an eliminated boundary, the binding of its float output points at the codes tensor
// that fed the boundary.
struct QuantBinding {
   std::string tensor;
   EQuantForm form = EQuantForm::Float;
   std::optional<QuantSpec> spec;
};

// Reports on an operator's neighbours, so a handler decides without knowing their types.
class IQuantNegotiation {
public:
   virtual ~IQuantNegotiation() = default;
   // Returns whether every emitted reader of `tensor` accepts it in `form` on `spec`.
   virtual bool ReadersAccept(const std::string &tensor, EQuantForm form, const QuantSpec &spec) const = 0;
   // Returns the binding decided for `tensor`. Producers are decided before their readers.
   virtual const QuantBinding &Binding(const std::string &tensor) const = 0;
   // Returns the operator that writes `tensor`, taking absorptions into account.
   virtual std::optional<std::size_t> Producer(const std::string &tensor) const = 0;
};

struct QuantContext {
   QuantOpView op;
   const IQuantNegotiation *negotiation;   // null during Annotate
   QuantWorkflowReport &report;

   const QuantGraphView &Graph() const { return op.graph; }
};

// Chain of following operators that an operator computes itself (e.g. the boundary pair and
// Relu after a Gemm). The chain's operators are not emitted, and the absorbing operator
// writes `output`, the chain's last tensor, in place of its own.
struct QuantAbsorption {
   std::vector<std::size_t> operators;   // in graph order
   std::string output;
};

// Decide-pass result for one operator, passed to its handler's Lower.
struct LoweringDecision {
   std::vector<QuantBinding> inputs;    // one per input slot
   std::vector<QuantBinding> outputs;   // one per output slot (the absorbed chain's output if any)
   std::optional<QuantAbsorption> absorption;

   bool IsFloatOnly() const
   {
      for (const auto &b : inputs)
         if (b.form != EQuantForm::Float && b.form != EQuantForm::Snapped)
            return false;
      for (const auto &b : outputs)
         if (b.form != EQuantForm::Float)
            return false;
      return !absorption.has_value();
   }
};

class IQuantOpHandler {
public:
   virtual ~IQuantOpHandler() = default;
   virtual const char *Name() const = 0;

   // Annotate: infers the state of each output from the inputs' states (default: plain float).
   virtual void InferOutputStates(const QuantContext &, std::vector<TensorQuantState> &) const {}

   // Decide: returns the chain of following operators this one folds into itself; the driver
   // suppresses them when Lower succeeds. Must be a pure function of the graph, since the driver
   // calls it more than once.
   virtual std::optional<QuantAbsorption> AbsorbConsumers(const QuantContext &) const { return std::nullopt; }

   // Decide: returns the forms this operator reads on `input` / writes on `output` onto
   // `target`, most preferred first. Lower must honour every form offered here. For an operator
   // with an absorption, output 0 is the absorbed chain's output.
   virtual QuantForms AcceptedInputForms(const QuantContext &, std::size_t /*input*/) const
   {
      return {EQuantForm::Float};
   }
   virtual QuantForms OfferedOutputForms(const QuantContext &, std::size_t /*output*/,
                                         const QuantSpec & /*target*/) const
   {
      return {EQuantForm::Float};
   }

   // Lower: realizes the decision. Returns false when it cannot; the driver then pins the
   // operator to float and decides again, so a failed lowering never breaks the graph.
   virtual bool Lower(const QuantContext &, const LoweringDecision &, QuantRewriter &) const { return false; }
};

class QuantOpHandlerRegistry {
public:
   template <class Op>
   void Register(std::shared_ptr<const IQuantOpHandler> handler)
   {
      fHandlers[std::type_index(typeid(Op))] = std::move(handler);
   }
   // Returns the handler for the operator's dynamic type, or the float-only fallback.
   const IQuantOpHandler &For(const ROperator &op) const;

   // Returns the registry holding every built-in registration (see RQuantization_HandlerRegistry.cxx).
   static const QuantOpHandlerRegistry &Default();

private:
   std::unordered_map<std::type_index, std::shared_ptr<const IQuantOpHandler>> fHandlers;
};

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_HANDLER
