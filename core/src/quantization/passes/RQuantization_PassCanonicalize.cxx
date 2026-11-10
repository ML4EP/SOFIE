#include "SOFIE/quantization/passes/RQuantization_Passes.hxx"

#include "SOFIE/quantization/frontend/ROperator_QuantBoundary.hxx"

#include <map>

namespace SOFIE {

void CanonicalizePass::Run(QuantWorkflow &workflow)
{
   auto &model = workflow.Access().Model();
   auto view = workflow.View();
   QuantRewriter rewriter;

   // Duplicate decodes: exporters emit one decode per reader of the same carrier.
   std::map<std::string, std::size_t> firstDecodeOf;   // carrier -> first decode's index
   for (std::size_t index = 0; index < view.OperatorCount(); ++index) {
      const ROperator *op = view.Op(index);
      const auto *boundary = dynamic_cast<const ROperator_QuantBoundary *>(op);
      if (boundary == nullptr || boundary->IsOutputConstant() || boundary->IsView())
         continue;
      const auto &out = boundary->GetOutputTensor();
      if (view.IsGraphOutput(out) || model.IsAliasTensor(out))
         continue;

      if (dynamic_cast<const ROperator_QuantDecode *>(op) != nullptr) {
         auto [it, first] = firstDecodeOf.emplace(boundary->GetInputTensor(), index);
         if (first)
            continue;
         const auto *survivor = dynamic_cast<const ROperator_QuantBoundary *>(view.Op(it->second));
         if (!SameGrid(survivor->GetSpec(), boundary->GetSpec()))
            continue;
         const std::string origin = survivor->GetOutputTensor();
         rewriter.Mutate(index, [origin](ROperator &target) {
            static_cast<ROperator_QuantBoundary &>(target).MarkViewOf(origin);
         });
         rewriter.Alias(out, origin);
         workflow.Report().Record(Name(), out, "duplicate decode becomes a view of " + origin);
         continue;
      }

      // An encode of a same-grid decode's output becomes a view of the decode's input codes.
      const auto producer = view.Producer(boundary->GetInputTensor());
      if (!producer)
         continue;
      const auto *decode = dynamic_cast<const ROperator_QuantDecode *>(view.Op(*producer));
      if (decode == nullptr || decode->IsOutputConstant() || !SameGrid(decode->GetSpec(), boundary->GetSpec()))
         continue;
      const std::string origin = decode->GetInputTensor();
      rewriter.Mutate(index, [origin](ROperator &target) {
         static_cast<ROperator_QuantBoundary &>(target).MarkViewOf(origin);
      });
      rewriter.Alias(out, origin);
      workflow.Report().Record(Name(), out, "encode of a same-grid decode becomes a view of " + origin);
   }
   rewriter.Commit(workflow.Access());
}

} // namespace SOFIE
