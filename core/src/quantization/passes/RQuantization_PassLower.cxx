#include "SOFIE/quantization/passes/RQuantization_Passes.hxx"

#include "SOFIE/quantization/frontend/ROperator_QuantBoundary.hxx"

#include <set>
#include <utility>

namespace SOFIE {

void LowerPass::Run(QuantWorkflowState &workflow)
{
   auto &plan = workflow.Plan();
   auto &report = workflow.Report();
   // Every declined lowering pins one more operator to float, so the loop converges within the
   // number of decided operators; the bound guards against a misbehaving handler.
   const std::size_t maxRounds = plan.decisions.size() + 2;
   for (std::size_t round = 0; round < maxRounds; ++round) {
      auto view = workflow.View();
      std::vector<QuantRewriter> edits(plan.decisions.size());
      std::size_t slot = 0;
      bool declined = false;
      for (const auto &[index, decision] : plan.decisions) {
         const ROperator *op = view.Op(index);
         const auto &handler = workflow.Registry().For(*op);
         QuantContext ctx{QuantOpView{*op, index, view}, nullptr, report};
         auto &rewriter = edits[slot++];
         if (handler.Lower(ctx, decision, rewriter)) {
            if (decision.absorption)
               for (auto absorbed : decision.absorption->operators)
                  rewriter.Suppress(absorbed);
            continue;
         }
         report.Record(Name(), op->Name(),
                       std::string(handler.Name()) + " declined the decided forms; pinned to float");
         plan.pinnedFloat.insert(index);
         declined = true;
      }
      if (declined) {
         // No edit is committed; Decide reruns without the declined operators.
         DecidePass().Run(workflow);
         continue;
      }
      for (auto &rewriter : edits) {
         for (auto &name : rewriter.ConstantNames())
            plan.loweredConstants.insert(std::move(name));
         rewriter.Commit(workflow.Access());
      }
      for (auto index : plan.eliminatedBoundaries)
         workflow.Access().SuppressOperator(index);
      report.operatorsLowered += plan.decisions.size();
      report.boundariesEliminated += plan.eliminatedBoundaries.size();
      return;
   }
   throw std::runtime_error("SOFIE quantization lowering did not converge");
}

void FinalizePass::Run(QuantWorkflowState &workflow)
{
   auto &access = workflow.Access();
   access.EliminateDeadOperators();

   // Erases the constants the workflow made obsolete, when no emitted operator reads them, so they
   // are neither embedded, uploaded, nor written to the weight file:
   //  - those a boundary folded or read (baked codes, their decoded floats, and the scale /
   //    zero-point / bit-width initializers its spec was resolved from);
   //  - those a lowered or suppressed operator read in the parsed graph (e.g. a float bias the
   //    lowered GEMM folded, the bounds of an absorbed Clip);
   //  - those the lowering added and a later choice left unread.
   // A constant no parsed operator read is left alone: the workflow did not make it obsolete.
   {
      auto view = workflow.View();
      const auto &model = std::as_const(access.Model());
      std::set<std::string> candidates(workflow.Plan().loweredConstants.begin(),
                                       workflow.Plan().loweredConstants.end());
      for (std::size_t index = 0; index < access.OperatorCount(); ++index) {
         if (model.IsOperatorLowered(index) || model.GetEmittedOperatorAt(index) == nullptr)
            for (const auto &input : access.Operator(index).GetOpInputTensors())
               candidates.insert(std::string(input));
         const auto *boundary = dynamic_cast<const ROperator_QuantBoundary *>(&access.Operator(index));
         if (boundary == nullptr)
            continue;
         candidates.insert(boundary->GetInputTensor());
         if (boundary->IsOutputConstant())
            candidates.insert(boundary->GetOutputTensor());
         // The spec carries these values; boundary kernels embed them as literals or tables.
         const auto &params = boundary->GetParameters();
         for (const auto *parameter : {&params.scaleTensor, &params.zeroPointTensor, &params.bitWidthTensor})
            if (!parameter->empty())
               candidates.insert(*parameter);
      }
      std::set<std::string> read;
      for (std::size_t index = 0; index < view.OperatorCount(); ++index) {
         const ROperator *op = view.Op(index);
         // A folded boundary generates no code, so its inputs are not read at run time.
         if (op == nullptr || op->IsOutputConstant())
            continue;
         for (const auto &input : op->GetOpInputTensors())
            read.insert(std::string(input));
      }
      for (const auto &tensor : candidates)
         if (view.IsConstant(tensor) && read.count(tensor) == 0 && !view.IsGraphOutput(tensor))
            access.EraseInitializer(tensor);
   }

   // Records each boundary emitted as a kernel, with the reader or producer that keeps it.
   auto view = workflow.View();
   auto &report = workflow.Report();
   for (std::size_t index = 0; index < view.OperatorCount(); ++index) {
      const auto *boundary = dynamic_cast<const ROperator_QuantBoundary *>(view.Op(index));
      if (boundary == nullptr || boundary->IsOutputConstant() || boundary->IsView())
         continue;
      ++report.boundariesKept;
      const bool isEncode = dynamic_cast<const ROperator_QuantEncode *>(boundary) != nullptr;
      // An encode is kept by its producer, a decode by its readers.
      std::string why = "graph boundary";
      const auto &tensor = isEncode ? boundary->GetInputTensor() : boundary->GetOutputTensor();
      if (isEncode) {
         if (auto producer = view.Producer(tensor); producer && view.Op(*producer))
            why = "producer " + view.Op(*producer)->Name() + " (" +
                  workflow.Registry().For(*view.Op(*producer)).Name() + ")";
      } else {
         for (auto reader : view.Consumers(tensor)) {
            // A view of this boundary does not keep it.
            const auto *readerBoundary = dynamic_cast<const ROperator_QuantBoundary *>(view.Op(reader));
            if (readerBoundary != nullptr && readerBoundary->IsView())
               continue;
            if (view.Op(reader)) {
               why = "reader " + view.Op(reader)->Name() + " (" + workflow.Registry().For(*view.Op(reader)).Name() + ")";
               break;
            }
         }
      }
      report.Record(Name(), tensor, std::string(isEncode ? "encode" : "decode") + " kept by " + why);
   }
}

} // namespace SOFIE
