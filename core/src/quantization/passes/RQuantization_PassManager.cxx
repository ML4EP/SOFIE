#include "SOFIE/quantization/passes/RQuantization_Passes.hxx"

namespace SOFIE {

QuantWorkflow::QuantWorkflow(RModel &model, const QuantOpHandlerRegistry &registry)
   : fAccess(model), fRegistry(registry)
{
}

void QuantPassManager::Run(QuantWorkflow &workflow) const
{
   for (const auto &pass : fPasses)
      pass->Run(workflow);
}

QuantPassManager MakeQuantAnalysisPipeline()
{
   QuantPassManager manager;
   manager.Add(std::make_unique<AnnotatePass>());
   return manager;
}

QuantPassManager MakeQuantLoweringPipeline()
{
   QuantPassManager manager;
   manager.Add(std::make_unique<CanonicalizePass>())
      .Add(std::make_unique<DecidePass>())
      .Add(std::make_unique<LowerPass>())
      .Add(std::make_unique<FinalizePass>());
   return manager;
}

} // namespace SOFIE
