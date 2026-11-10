#ifndef SOFIE_RQUANTIZATION_PASSES
#define SOFIE_RQUANTIZATION_PASSES

// Quantization workflow passes, the QuantWorkflow state they share, and the two pipelines that
// fix their order:
//
//   Analysis (once, after Initialize)        Lowering (per code generation)
//     Annotate      grid/form per tensor        Canonicalize  boundary clean-up (views)
//                                               Decide        absorption + form negotiation
//                                               Lower         handlers build/rewire, commit
//                                               Finalize      dead code, pruning, frontier
//
// A pass reads the model through QuantWorkflow's views and edits it only by committing a
// QuantRewriter, which keeps each pass independently testable.

#include "SOFIE/RModelCodegenPass.hxx"
#include "SOFIE/quantization/diagnostics/RQuantization_Diagnostics.hxx"
#include "SOFIE/quantization/graph/RQuantization_Graph.hxx"
#include "SOFIE/quantization/handlers/RQuantization_Handler.hxx"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace SOFIE {

// Whole-graph decisions produced by Decide.
struct QuantPlan {
   std::unordered_map<std::string, QuantBinding> bindings;   // logical tensor -> physical carrier
   std::set<std::size_t> eliminatedBoundaries;               // encode/decode boundaries not emitted
   std::map<std::size_t, LoweringDecision> decisions;        // operators with a non-float decision
   std::set<std::size_t> pinnedFloat;                        // operators whose handler declined to lower
   std::set<std::size_t> absorbed;                           // operators inside an absorption
   std::unordered_map<std::string, std::size_t> producerOverride;   // tensor -> absorbing operator

   void Clear()
   {
      bindings.clear();
      eliminatedBoundaries.clear();
      decisions.clear();
      absorbed.clear();
      producerOverride.clear();
   }
};

// State shared by one workflow run: the model, its handlers, the per-tensor states from
// Annotate, the plan from Decide, and the report.
class QuantWorkflow {
public:
   QuantWorkflow(RModel &model, const QuantOpHandlerRegistry &registry);

   QuantModelAccess &Access() { return fAccess; }
   const QuantOpHandlerRegistry &Registry() const { return fRegistry; }

   // Returns a new view over the emitted operators, carrying the annotated states.
   QuantGraphView View() { return QuantGraphView(fAccess, &fStates); }

   std::unordered_map<std::string, TensorQuantState> &States() { return fStates; }
   QuantPlan &Plan() { return fPlan; }
   QuantWorkflowReport &Report() { return fReport; }

private:
   QuantModelAccess fAccess;
   const QuantOpHandlerRegistry &fRegistry;
   std::unordered_map<std::string, TensorQuantState> fStates;
   QuantPlan fPlan;
   QuantWorkflowReport fReport;
};

class IQuantPass {
public:
   virtual ~IQuantPass() = default;
   virtual const char *Name() const = 0;
   virtual void Run(QuantWorkflow &workflow) = 0;
};

class QuantPassManager {
public:
   QuantPassManager &Add(std::unique_ptr<IQuantPass> pass)
   {
      fPasses.push_back(std::move(pass));
      return *this;
   }
   void Run(QuantWorkflow &workflow) const;

private:
   std::vector<std::unique_ptr<IQuantPass>> fPasses;
};

// ---- the passes -----------------------------------------------------------------------

// Folds redundant boundaries: duplicate decodes of one carrier on one grid become views of the
// first, and an encode of a same-grid decode becomes a view of the codes.
class CanonicalizePass final : public IQuantPass {
public:
   const char *Name() const override { return "canonicalize"; }
   void Run(QuantWorkflow &workflow) override;
};

// Assigns every tensor its TensorQuantState: boundaries define grids, handlers propagate them.
class AnnotatePass final : public IQuantPass {
public:
   const char *Name() const override { return "annotate"; }
   void Run(QuantWorkflow &workflow) override;
};

// Runs absorption, then negotiates each tensor's form between its producer and readers: a
// decode whose readers all take codes, or an encode whose producer writes codes, is not
// emitted. Edits only the plan, never the model.
class DecidePass final : public IQuantPass {
public:
   const char *Name() const override { return "decide"; }
   void Run(QuantWorkflow &workflow) override;
};

// Asks each handler with a non-float decision to lower. A declining handler's operator is
// pinned to float and Decide reruns, so every handler accepts the committed plan.
class LowerPass final : public IQuantPass {
public:
   const char *Name() const override { return "lower"; }
   void Run(QuantWorkflow &workflow) override;
};

// Removes unread operators and constants no emitted operator reads, lists the boundaries that
// remain emitted (the frontier), and closes the report.
class FinalizePass final : public IQuantPass {
public:
   const char *Name() const override { return "finalize"; }
   void Run(QuantWorkflow &workflow) override;
};

QuantPassManager MakeQuantAnalysisPipeline();
QuantPassManager MakeQuantLoweringPipeline();

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_PASSES
