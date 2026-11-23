#ifndef SOFIE_RQUANTIZATION_WORKFLOW
#define SOFIE_RQUANTIZATION_WORKFLOW

// Provides the quantization workflow as the RModelPlugin whose hooks run the frontends (OnParsed),
// the analysis pipeline (OnInitialized) and the lowering pipeline (PrepareGeneration).
// core/src/quantization/README.md describes the full workflow.

#include "SOFIE/RModelExtension.hxx"
#include "SOFIE/RModelPlugin.hxx"
#include "SOFIE/quantization/passes/RQuantization_Passes.hxx"

#include <memory>

namespace SOFIE {

// Per-model workflow state, held in the model's extension slot across the hooks.
class QuantWorkflowExtension final : public RModelExtension {
public:
   std::size_t frontendNodesTranslated = 0;
   std::unique_ptr<QuantWorkflowState> workflow;

   static QuantWorkflowExtension &Of(RModel &model);
   static const QuantWorkflowExtension *Find(const RModel &model);
};

class QuantizationWorkflow final : public RModelPlugin {
public:
   void OnParsed(RModel &model) override;
   void OnInitialized(RModel &model) override;
   void PrepareGeneration(RModel &model, EGenerationTarget target) override;
   bool OwnsStorage(const RModel &model, const std::string &tensorName) const override;
   std::string SupportCode(const RModel &model, std::string_view generatedCode) const override;
};

// Installs the quantization workflow into code generation unless it is already installed. A
// static registrar calls it when the library loads; the ONNX parser calls it too, which keeps
// the linker from dropping the library.
void InstallQuantizationWorkflow();

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_WORKFLOW
