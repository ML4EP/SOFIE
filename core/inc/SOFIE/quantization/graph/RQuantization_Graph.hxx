#ifndef SOFIE_RQUANTIZATION_GRAPH
#define SOFIE_RQUANTIZATION_GRAPH

// Provides the workflow's view of a model.
//   QuantModelAccess  reads and rewrites the RModel's operators and tensor types
//   QuantGraphView    indexes the emitted operators and per-tensor quant state, read-only
//   QuantOpView       presents one operator to the handlers
//   QuantRewriter     collects a pass's edits and applies them in one commit
// Handlers read through the views and propose edits through the rewriter; they never edit the model.

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/quantization/spec/RQuantization_Spec.hxx"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace SOFIE {

class QuantModelAccess {
public:
   explicit QuantModelAccess(RModel &model) : fModel(model) {}

   RModel &Model() { return fModel; }
   const RModel &Model() const { return fModel; }

   std::size_t OperatorCount() const;
   ROperator &Operator(std::size_t index);
   const ROperator &Operator(std::size_t index) const;
   const std::vector<std::string> &GraphOutputs() const;

   // Rewrite the parsed list during the Normalize phase only (before Initialize). Setting the
   // list rebuilds the model's tensor-usage table, which refers into operator storage.
   std::vector<std::unique_ptr<ROperator>> TakeOperators();
   void SetOperators(std::vector<std::unique_ptr<ROperator>> operators);

   // Manage the lowered view (after Initialize): replacements and suppressions over the parsed list.
   void ResetLoweredView();
   void InstallLowered(std::size_t anchor, std::unique_ptr<ROperator> op);
   void Suppress(std::size_t index);
   // Returns the operator emitted at `index`: its replacement, the original, or null if suppressed.
   const ROperator *Emitted(std::size_t index) const;
   ROperator *Emitted(std::size_t index);
   void EliminateDeadOperators();

   // Renames an initializer; valid in the Normalize phase only.
   void RenameInitializer(const std::string &from, const std::string &to);
   bool IsGraphInput(const std::string &tensor) const;
   // Erases an initializer with no remaining readers (e.g. a decoded copy of baked codes).
   void EraseInitializer(const std::string &tensor);

   void SetTensorType(const std::string &tensor, ETensorType type);
   void AddAlias(const std::string &name, const std::string &origin);

private:
   RModel &fModel;
};

class QuantGraphView {
public:
   explicit QuantGraphView(QuantModelAccess &access, const std::unordered_map<std::string, TensorQuantState> *states = nullptr);

   std::size_t OperatorCount() const { return fOperatorCount; }
   const ROperator *Op(std::size_t index) const { return fAccess.Emitted(index); }
   std::optional<std::size_t> Producer(const std::string &tensor) const;
   const std::vector<std::size_t> &Consumers(const std::string &tensor) const;
   bool IsGraphOutput(const std::string &tensor) const { return fGraphOutputs.count(tensor) != 0; }
   bool IsConstant(const std::string &tensor) const;
   std::vector<std::size_t> Shape(const std::string &tensor) const;
   ETensorType Type(const std::string &tensor) const;
   // Returns the tensor's quant state; Float with no grid for an unknown tensor.
   const TensorQuantState &State(const std::string &tensor) const;
   const RModel &Model() const { return fAccess.Model(); }

private:
   QuantModelAccess &fAccess;
   const std::unordered_map<std::string, TensorQuantState> *fStates;
   std::size_t fOperatorCount = 0;
   std::unordered_map<std::string, std::size_t> fProducers;
   std::unordered_map<std::string, std::vector<std::size_t>> fConsumers;
   std::set<std::string> fGraphOutputs;
};

struct QuantOpView {
   const ROperator &op;
   std::size_t index;
   const QuantGraphView &graph;

   std::size_t InputCount() const { return op.GetOpInputTensors().size(); }
   std::size_t OutputCount() const { return op.GetOpOutputTensors().size(); }
   std::string Input(std::size_t i) const { return std::string(op.GetOpInputTensors()[i]); }
   std::string Output(std::size_t i) const { return std::string(op.GetOpOutputTensors()[i]); }
   const TensorQuantState &InputState(std::size_t i) const { return graph.State(Input(i)); }
};

class QuantRewriter {
public:
   // Emits `op` in place of the operator at `anchor`.
   void Replace(std::size_t anchor, std::unique_ptr<ROperator> op);
   // Stops emitting the operator at `index` (absorbed, or an eliminated boundary).
   void Suppress(std::size_t index);
   // Edits the operator at `index` in place.
   void Mutate(std::size_t index, std::function<void(ROperator &)> edit);
   void RetypeTensor(const std::string &tensor, ETensorType type);
   void Alias(const std::string &name, const std::string &origin);
   // Adds a constant the lowered operator reads (re-laid weights, scale and bias vectors). Large
   // constants go to the weight file, small ones are embedded; all are uploaded to the device.
   void AddConstant(const std::string &name, ETensorType type, std::vector<std::size_t> shape,
                    std::shared_ptr<void> data);
   // Adds scratch the lowered operator writes and reads within one inference (int32 accumulators).
   void AddScratch(const std::string &name, ETensorType type, std::vector<std::size_t> shape);

   void Commit(QuantModelAccess &access);
   void Discard();

private:
   std::map<std::size_t, std::unique_ptr<ROperator>> fReplacements;
   std::set<std::size_t> fSuppressed;
   std::vector<std::pair<std::size_t, std::function<void(ROperator &)>>> fMutations;
   std::vector<std::pair<std::string, ETensorType>> fRetypes;
   std::vector<std::pair<std::string, std::string>> fAliases;
   struct Constant {
      std::string name;
      ETensorType type;
      std::vector<std::size_t> shape;
      std::shared_ptr<void> data;
   };
   std::vector<Constant> fConstants;
   std::vector<std::tuple<std::string, ETensorType, std::vector<std::size_t>>> fScratch;
};

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_GRAPH
