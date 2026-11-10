#include "SOFIE/quantization/passes/RQuantization_Passes.hxx"

#include "SOFIE/quantization/frontend/ROperator_QuantBoundary.hxx"

namespace SOFIE {

namespace {

// Negotiation oracle that handlers query for the forms readers accept and the bindings decided so far.
class Negotiation final : public IQuantNegotiation {
public:
   Negotiation(QuantWorkflow &workflow, const QuantGraphView &view) : fWorkflow(workflow), fView(view) {}

   QuantContext Context(std::size_t index) const
   {
      return QuantContext{QuantOpView{*fView.Op(index), index, fView}, this, fWorkflow.Report()};
   }

   const IQuantOpHandler &HandlerOf(std::size_t index) const { return fWorkflow.Registry().For(*fView.Op(index)); }

   bool ReadersAccept(const std::string &tensor, EQuantForm form, const QuantSpec &spec) const override
   {
      if (fView.IsGraphOutput(tensor))
         return false;
      const auto &readers = fView.Consumers(tensor);
      if (readers.empty())
         return false;
      const auto &plan = fWorkflow.Plan();
      for (std::size_t reader : readers) {
         const ROperator *op = fView.Op(reader);
         if (op == nullptr || plan.pinnedFloat.count(reader) != 0 || plan.absorbed.count(reader) != 0)
            return false;
         // A sole-reader encode onto the same grid accepts codes; Decide eliminates it.
         if (const auto *encode = dynamic_cast<const ROperator_QuantEncode *>(op)) {
            if (form == EQuantForm::Codes && readers.size() == 1 && !encode->IsOutputConstant() &&
                !encode->IsView() && SameGrid(encode->GetSpec(), spec))
               continue;
            return false;
         }
         if (dynamic_cast<const ROperator_QuantBoundary *>(op) != nullptr)
            return false;
         const auto &handler = HandlerOf(reader);
         const auto ctx = Context(reader);
         for (std::size_t slot = 0; slot < ctx.op.InputCount(); ++slot)
            if (ctx.op.Input(slot) == tensor && !Contains(handler.AcceptedInputForms(ctx, slot), form))
               return false;
      }
      return true;
   }

   const QuantBinding &Binding(const std::string &tensor) const override
   {
      auto &bindings = fWorkflow.Plan().bindings;
      auto it = bindings.find(tensor);
      if (it != bindings.end())
         return it->second;
      // An undecided tensor binds to itself in its parsed form.
      const auto &state = fView.State(tensor);
      QuantBinding binding{tensor, state.form == EQuantForm::Codes ? EQuantForm::Codes : EQuantForm::Float,
                           state.spec};
      return bindings.emplace(tensor, std::move(binding)).first->second;
   }

   std::optional<std::size_t> Producer(const std::string &tensor) const override
   {
      const auto &overrides = fWorkflow.Plan().producerOverride;
      if (auto it = overrides.find(tensor); it != overrides.end())
         return it->second;
      return fView.Producer(tensor);
   }

private:
   QuantWorkflow &fWorkflow;
   const QuantGraphView &fView;
};

// Returns the decode that is the sole reader of `codes`, or nullptr.
const ROperator_QuantDecode *SoleDecodeReader(const QuantGraphView &view, const std::string &codes)
{
   const auto &readers = view.Consumers(codes);
   if (readers.size() != 1 || view.IsGraphOutput(codes))
      return nullptr;
   const auto *decode = dynamic_cast<const ROperator_QuantDecode *>(view.Op(readers[0]));
   return decode != nullptr && !decode->IsView() ? decode : nullptr;
}

} // namespace

void DecidePass::Run(QuantWorkflow &workflow)
{
   auto &plan = workflow.Plan();
   plan.Clear();
   auto view = workflow.View();
   Negotiation negotiation(workflow, view);
   auto &report = workflow.Report();
   const auto isBoundary = [&view](std::size_t index) {
      return dynamic_cast<const ROperator_QuantBoundary *>(view.Op(index)) != nullptr;
   };

   // 1. Absorptions: an operator that computes a following chain itself becomes the producer
   //    of the chain's output; the steps below skip the chain.
   std::map<std::size_t, QuantAbsorption> absorptions;
   for (std::size_t index = 0; index < view.OperatorCount(); ++index) {
      if (view.Op(index) == nullptr || isBoundary(index) || plan.pinnedFloat.count(index) != 0 ||
          plan.absorbed.count(index) != 0)
         continue;
      auto absorption = negotiation.HandlerOf(index).AbsorbConsumers(negotiation.Context(index));
      if (!absorption)
         continue;
      bool free = true;
      for (auto member : absorption->operators)
         free &= plan.absorbed.count(member) == 0 && plan.pinnedFloat.count(member) == 0;
      if (!free)
         continue;
      plan.absorbed.insert(absorption->operators.begin(), absorption->operators.end());
      plan.producerOverride[absorption->output] = index;
      report.Record(Name(), view.Op(index)->Name(),
                    "absorbs " + std::to_string(absorption->operators.size()) + " operators up to " + absorption->output);
      absorptions.emplace(index, std::move(*absorption));
   }

   // Returns an operator's logical outputs: its absorbed chain's output, or its own outputs.
   const auto logicalOutputs = [&](std::size_t index) {
      if (auto it = absorptions.find(index); it != absorptions.end())
         return std::vector<std::string>{it->second.output};
      std::vector<std::string> outputs;
      for (const auto &out : view.Op(index)->GetOpOutputTensors())
         outputs.emplace_back(out);
      return outputs;
   };

   // 2. Forms, in execution order, so every input's binding is final before its reader is visited.
   for (std::size_t index = 0; index < view.OperatorCount(); ++index) {
      const ROperator *op = view.Op(index);
      if (op == nullptr || plan.absorbed.count(index) != 0 || plan.eliminatedBoundaries.count(index) != 0)
         continue;

      if (const auto *decode = dynamic_cast<const ROperator_QuantDecode *>(op)) {
         if (decode->IsView())
            continue;
         const auto &out = decode->GetOutputTensor();
         if (negotiation.ReadersAccept(out, EQuantForm::Codes, decode->GetSpec())) {
            plan.bindings[out] = QuantBinding{decode->GetInputTensor(), EQuantForm::Codes, decode->GetSpec()};
            plan.eliminatedBoundaries.insert(index);
            report.Record(Name(), out, "decode eliminated: every reader takes the codes");
         }
         continue;
      }

      if (const auto *encode = dynamic_cast<const ROperator_QuantEncode *>(op)) {
         if (encode->IsView() || encode->IsOutputConstant())
            continue;
         const auto &in = encode->GetInputTensor();
         const auto producer = negotiation.Producer(in);
         if (!producer || view.Consumers(in).size() != 1 || view.IsGraphOutput(in) ||
             plan.pinnedFloat.count(*producer) != 0 || isBoundary(*producer) || view.Op(*producer) == nullptr)
            continue;
         const auto ctx = negotiation.Context(*producer);
         const auto offered = negotiation.HandlerOf(*producer).OfferedOutputForms(ctx, 0, encode->GetSpec());
         // A fake-quant pair whose readers take the float: the producer writes the snapped float
         // straight into the decode's output, and both boundaries are eliminated.
         const auto *decode = SoleDecodeReader(view, encode->GetOutputTensor());
         if (decode != nullptr && SameGrid(decode->GetSpec(), encode->GetSpec()) &&
             !negotiation.ReadersAccept(decode->GetOutputTensor(), EQuantForm::Codes, decode->GetSpec()) &&
             Contains(offered, EQuantForm::Snapped)) {
            plan.bindings[in] = QuantBinding{decode->GetOutputTensor(), EQuantForm::Snapped, encode->GetSpec()};
            plan.eliminatedBoundaries.insert(index);
            const auto decodeIndex = view.Consumers(encode->GetOutputTensor())[0];
            plan.eliminatedBoundaries.insert(decodeIndex);
            report.Record(Name(), decode->GetOutputTensor(),
                          "fake-quant pair eliminated: " + view.Op(*producer)->Name() + " writes the snapped float");
            continue;
         }
         if (Contains(offered, EQuantForm::Codes)) {
            // The producer writes the encode's codes tensor itself.
            plan.bindings[in] = QuantBinding{encode->GetOutputTensor(), EQuantForm::Codes, encode->GetSpec()};
            plan.eliminatedBoundaries.insert(index);
            report.Record(Name(), encode->GetOutputTensor(),
                          "encode eliminated: " + view.Op(*producer)->Name() + " writes the codes");
         }
         continue;
      }

      if (plan.pinnedFloat.count(index) != 0)
         continue;
      // An operator reading codes keeps them on its outputs when it can write them and its
      // readers take them (for example a movement op between a decode and an encode).
      const auto ctx = negotiation.Context(index);
      bool readsCodes = false;
      for (std::size_t i = 0; i < ctx.op.InputCount(); ++i)
         readsCodes |= negotiation.Binding(ctx.op.Input(i)).form == EQuantForm::Codes;
      if (!readsCodes || absorptions.count(index) != 0)
         continue;
      const auto &handler = negotiation.HandlerOf(index);
      for (std::size_t j = 0; j < ctx.op.OutputCount(); ++j) {
         const auto out = ctx.op.Output(j);
         const auto &state = view.State(out);
         if (!state.HasGrid() || plan.bindings.count(out) != 0)
            continue;
         if (Contains(handler.OfferedOutputForms(ctx, j, *state.spec), EQuantForm::Codes) &&
             negotiation.ReadersAccept(out, EQuantForm::Codes, *state.spec))
            plan.bindings[out] = QuantBinding{out, EQuantForm::Codes, state.spec};
      }
   }

   // 3. One decision per operator whose bindings are not plain float.
   for (std::size_t index = 0; index < view.OperatorCount(); ++index) {
      const ROperator *op = view.Op(index);
      if (op == nullptr || isBoundary(index) || plan.pinnedFloat.count(index) != 0 ||
          plan.absorbed.count(index) != 0)
         continue;
      const auto ctx = negotiation.Context(index);
      LoweringDecision decision;
      if (auto it = absorptions.find(index); it != absorptions.end())
         decision.absorption = it->second;
      for (std::size_t i = 0; i < ctx.op.InputCount(); ++i)
         decision.inputs.push_back(negotiation.Binding(ctx.op.Input(i)));
      for (const auto &out : logicalOutputs(index))
         decision.outputs.push_back(negotiation.Binding(out));
      if (!decision.IsFloatOnly())
         plan.decisions.emplace(index, std::move(decision));
   }
}

} // namespace SOFIE
