#ifndef SOFIE_RQUANTIZATION_WORKFLOW
#define SOFIE_RQUANTIZATION_WORKFLOW

// Provides the quantization workflow as one RModelCodegenPass whose hooks run the frontends
// (Normalize), the analysis pipeline (Analyze) and the lowering pipeline (BuildLoweredView).
// core/src/quantization/README.md describes the full workflow.

#include "SOFIE/RModelCodegenPass.hxx"
#include "SOFIE/RModelExtension.hxx"
#include "SOFIE/quantization/passes/RQuantization_Passes.hxx"

#include <memory>

namespace SOFIE {

// Per-model workflow state, held in the model's extension slot across the hooks.
class QuantWorkflowExtension final : public RModelExtension {
public:
   std::size_t frontendNodesTranslated = 0;
   std::unique_ptr<QuantWorkflow> workflow;

   static QuantWorkflowExtension &Of(RModel &model);
   static const QuantWorkflowExtension *Find(const RModel &model);
};

class QuantizationWorkflowPass final : public RModelCodegenPass {
public:
   void Normalize(RModel &model) override;
   void Analyze(RModel &model) override;
   bool RequiresWeightFile(const RModel &model) const override;
   void BuildLoweredView(RModel &model, ECodegenTarget target) override;
   void ContributeSupportHeaders(RModel &model, ECodegenTarget target) override;
   bool HasExternalStorage(const RModel &model, const std::string &tensorName) const override;
   std::vector<std::string> DiagnosticComments(const RModel &model) const override;
};

// Installs the quantization workflow into code generation. The ONNX parser calls it, and a
// static registrar calls it so hand-built models run the workflow too.
void InstallQuantizationCodegenPass();

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_WORKFLOW
