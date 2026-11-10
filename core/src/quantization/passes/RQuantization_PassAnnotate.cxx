#include "SOFIE/quantization/passes/RQuantization_Passes.hxx"

#include "SOFIE/quantization/frontend/ROperator_QuantBoundary.hxx"

namespace SOFIE {

void AnnotatePass::Run(QuantWorkflow &workflow)
{
   auto &states = workflow.States();
   states.clear();
   auto view = workflow.View();

   for (std::size_t index = 0; index < view.OperatorCount(); ++index) {
      const ROperator *op = view.Op(index);
      if (op == nullptr)
         continue;

      // Boundaries define grids: an encode writes codes; a decode reads codes and writes a
      // float on the grid.
      if (const auto *encode = dynamic_cast<const ROperator_QuantEncode *>(op)) {
         states[encode->GetOutputTensor()] = {encode->GetSpec(), EQuantForm::Codes, encode->IsOutputConstant()};
         continue;
      }
      if (const auto *decode = dynamic_cast<const ROperator_QuantDecode *>(op)) {
         const auto &codes = decode->GetInputTensor();
         states[codes] = {decode->GetSpec(), EQuantForm::Codes, view.IsConstant(codes)};
         states[decode->GetOutputTensor()] = {decode->GetSpec(), EQuantForm::Snapped, decode->IsOutputConstant()};
         continue;
      }

      // Any other operator: its handler propagates or drops the inputs' grids.
      const auto &handler = workflow.Registry().For(*op);
      QuantContext ctx{QuantOpView{*op, index, view}, nullptr, workflow.Report()};
      std::vector<TensorQuantState> outputs(ctx.op.OutputCount());
      handler.InferOutputStates(ctx, outputs);
      for (std::size_t j = 0; j < outputs.size(); ++j)
         if (outputs[j].HasGrid() || outputs[j].form != EQuantForm::Float)
            states[ctx.op.Output(j)] = outputs[j];
   }
}

} // namespace SOFIE
