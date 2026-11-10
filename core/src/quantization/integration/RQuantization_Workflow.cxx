#include "SOFIE/quantization/integration/RQuantization_Workflow.hxx"

#include "SOFIE/RModel.hxx"
#include "SOFIE/quantization/frontend/RQuantization_Frontend.hxx"

#include <stdexcept>

namespace SOFIE {

QuantWorkflowExtension &QuantWorkflowExtension::Of(RModel &model)
{
   if (auto *attached = dynamic_cast<QuantWorkflowExtension *>(model.GetExtension()))
      return *attached;
   auto owned = std::make_unique<QuantWorkflowExtension>();
   auto &extension = *owned;
   model.SetExtension(std::move(owned));
   return extension;
}

const QuantWorkflowExtension *QuantWorkflowExtension::Find(const RModel &model)
{
   return dynamic_cast<const QuantWorkflowExtension *>(model.GetExtension());
}

void QuantizationWorkflowPass::Normalize(RModel &model)
{
   QuantWorkflowExtension::Of(model).frontendNodesTranslated = NormalizeQuantizationFrontends(model);
}

void QuantizationWorkflowPass::Analyze(RModel &model)
{
   auto &extension = QuantWorkflowExtension::Of(model);
   if (extension.frontendNodesTranslated == 0)
      return;   // Unquantized model: code generation runs as without the workflow.
   extension.workflow =
      std::make_unique<QuantWorkflow>(model, QuantOpHandlerRegistry::Default());
   MakeQuantAnalysisPipeline().Run(*extension.workflow);
}

bool QuantizationWorkflowPass::RequiresWeightFile(const RModel &model) const
{
   const auto *extension = QuantWorkflowExtension::Find(model);
   return extension != nullptr && extension->frontendNodesTranslated > 0;
}

void QuantizationWorkflowPass::BuildLoweredView(RModel &model, ECodegenTarget)
{
   auto &extension = QuantWorkflowExtension::Of(model);
   if (!extension.workflow)
      return;
   auto &workflow = *extension.workflow;
   workflow.Access().ResetLoweredView();
   workflow.Plan().pinnedFloat.clear();
   MakeQuantLoweringPipeline().Run(workflow);
}

void QuantizationWorkflowPass::ContributeSupportHeaders(RModel &model, ECodegenTarget target)
{
   // Quantization is Alpaka-only: lowered operators, boundary views and BLAS calls all generate
   // Alpaka code. The plain C++ generator throws instead of emitting float code that silently
   // ignores the quantization.
   if (target != ECodegenTarget::ALPAKA && RequiresWeightFile(model))   // True for any quantized model.
      throw std::runtime_error("SOFIE: quantized models are supported by the Alpaka generator only "
                               "(RModel::GenerateGPU_ALPAKA); plain C++ generation does not support quantization");
   if (RequiresWeightFile(model))
      model.AddNeededCustomHeader("SOFIE/quantization/spec/RQuantization_Codec.hxx");
}

bool QuantizationWorkflowPass::HasExternalStorage(const RModel &model, const std::string &tensorName) const
{
   // Baked codes (int8 / uint8 / float8 weights) above the embedding threshold go to the weight
   // file like float weights instead of being embedded as constants.
   if (!RequiresWeightFile(model) || !model.IsInitializedTensor(tensorName))
      return false;
   const auto type = model.GetTensorType(tensorName);
   const bool codes = type == ETensorType::INT8 || type == ETensorType::UINT8 || type == ETensorType::FLOAT8E4M3FN;
   return codes && ConvertShapeToLength(model.GetTensorShape(tensorName)) > kQuantizedFoldedConstantEmbedMaxLength;
}

std::vector<std::string> QuantizationWorkflowPass::DiagnosticComments(const RModel &model) const
{
   const auto *extension = QuantWorkflowExtension::Find(model);
   if (extension == nullptr || !extension->workflow)
      return {};
   return extension->workflow->Report().Summary();
}

void InstallQuantizationCodegenPass()
{
   InstallCodegenPass(std::make_unique<QuantizationWorkflowPass>());
}

namespace {
// Installs the pass from a static registrar so generation behaves the same for every caller,
// including models built without the parser.
const bool kQuantizationCodegenPassInstalled = [] {
   InstallQuantizationCodegenPass();
   return true;
}();
} // namespace

} // namespace SOFIE
