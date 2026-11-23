#include "SOFIE/quantization/integration/RQuantization_Workflow.hxx"

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Cast.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_Transpose.hxx"
#include "SOFIE/quantization/frontend/RQuantization_Frontend.hxx"
#include "SOFIE/quantization/graph/RQuantization_Graph.hxx"
#include "SOFIE/quantization/spec/RQuantization_CodeFormat.hxx"

#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

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

namespace {
// True once the frontends translated a node, i.e. the model is quantized.
bool IsQuantized(const RModel &model)
{
   const auto *extension = QuantWorkflowExtension::Find(model);
   return extension != nullptr && extension->frontendNodesTranslated > 0;
}

// Puts the B operand of each batched (rank >= 3) MatMul in [.., N, K], the layout the quantized
// batched GEMMs require: re-permutes the Transpose that produces it and sets transB. Runs in
// OnParsed, after the frontends and before RModel::Initialize, because it changes a Transpose's
// perm and shape inference must see the new layout.
void CanonicaliseBatchedMatMulOperands(RModel &model)
{
   QuantModelAccess access(model);
   // A subgraph body can reference an outer tensor invisibly, so the single-reader guards below
   // would not be sound.
   if (access.HasSubGraphs())
      return;

   // Connectivity exists here but shapes do not, so shape inference derives every downstream
   // shape from the new perm.
   std::unordered_map<std::string, std::size_t> producer;
   std::unordered_map<std::string, std::size_t> readers;
   for (std::size_t index = 0; index < access.OperatorCount(); ++index) {
      const auto &op = access.Operator(index);
      for (const auto &output : op.GetOpOutputTensors())
         producer[std::string(output)] = index;
      for (const auto &input : op.GetOpInputTensors())
         ++readers[std::string(input)];
   }
   const std::set<std::string> graphOutputs(access.GraphOutputs().begin(), access.GraphOutputs().end());

   for (std::size_t index = 0; index < access.OperatorCount(); ++index) {
      auto *gemm = dynamic_cast<ROperator_Gemm<float> *>(&access.Operator(index));
      // A bias would take this off the strided-batched branch, and a transpose already set means
      // somebody else owns the layout.
      if (gemm == nullptr || gemm->GetTransA() != 0 || gemm->GetTransB() != 0 || gemm->HasBias())
         continue;

      // Walks back from the B operand, through the boundaries and Casts that stage it in low
      // precision, to the Transpose that lays it out. Every tensor on the way must be read only here.
      std::string cursor = gemm->GetWeightTensorName();
      ROperator_Transpose<float> *transpose = nullptr;
      // Staging places at most two operators between the operand and its Transpose.
      constexpr int kStagingWalkMaxHops = 3;
      for (int hop = 0; hop < kStagingWalkMaxHops && !cursor.empty(); ++hop) {
         if (readers[cursor] != 1 || graphOutputs.count(cursor) != 0)
            break;
         auto found = producer.find(cursor);
         if (found == producer.end())
            break;
         auto &op = access.Operator(found->second);
         if (auto *candidate = dynamic_cast<ROperator_Transpose<float> *>(&op)) {
            transpose = candidate;
            break;
         }
         // A boundary or a Cast only retypes the operand; the walk continues behind it.
         if (const auto *boundary = dynamic_cast<const ROperator_QuantBoundary *>(&op)) {
            cursor = boundary->GetInputTensor();
            continue;
         }
         if (dynamic_cast<ROperator_Cast *>(&op) != nullptr && !op.GetOpInputTensors().empty()) {
            cursor = std::string(op.GetOpInputTensors()[0]);
            continue;
         }
         break;
      }
      if (transpose == nullptr)
         continue;

      auto perm = transpose->GetPerm();
      // Rank 2 is the dense path, whose lowering repacks the weight to [N, K] itself.
      if (perm.size() < 3)
         continue;

      // B arrives as [.., K, N]; swapping the last two perm entries yields [.., N, K] at identical
      // cost, with transB restoring the semantics.
      std::swap(perm[perm.size() - 2], perm[perm.size() - 1]);
      transpose->SetPerm(std::move(perm));
      gemm->SetTransB(1);
      gemm->MarkBatchedOperandBCanonicalised();
   }
}
} // namespace

void QuantizationWorkflow::OnParsed(RModel &model)
{
   const auto translated = NormalizeQuantizationFrontends(model);
   QuantWorkflowExtension::Of(model).frontendNodesTranslated = translated;
   if (translated > 0)
      CanonicaliseBatchedMatMulOperands(model);
}

void QuantizationWorkflow::OnInitialized(RModel &model)
{
   if (!IsQuantized(model))
      return;   // Unquantized model: code generation runs as without the workflow.
   auto &extension = QuantWorkflowExtension::Of(model);
   extension.workflow = std::make_unique<QuantWorkflowState>(model, QuantOpHandlerRegistry::Default());
   MakeQuantAnalysisPipeline().Run(*extension.workflow);
}

void QuantizationWorkflow::PrepareGeneration(RModel &model, EGenerationTarget target)
{
   if (!IsQuantized(model))
      return;
   // Quantization is Alpaka-only: lowered operators, boundary views and BLAS calls all generate
   // Alpaka code. The plain C++ generator throws instead of emitting float code that silently
   // ignores the quantization.
   if (target != EGenerationTarget::ALPAKA)
      throw std::runtime_error("SOFIE: quantized models are supported by the Alpaka generator only "
                               "(RModel::GenerateGPU_ALPAKA); plain C++ generation does not support quantization");
   // The codec functions copied into the generated header use these.
   model.AddNeededStdLib("cmath");
   model.AddNeededStdLib("cstdint");
   auto &workflow = *QuantWorkflowExtension::Of(model).workflow;
   workflow.Access().ResetLoweredOperators();
   workflow.Plan().pinnedFloat.clear();
   MakeQuantLoweringPipeline().Run(workflow);
}

bool QuantizationWorkflow::OwnsStorage(const RModel &model, const std::string &tensorName) const
{
   // Baked codes above the embedding threshold go to the weight file like float weights instead
   // of being embedded as constants.
   if (!IsQuantized(model) || !model.IsInitializedTensor(tensorName))
      return false;
   return IsBakeableCodeType(model.GetTensorType(tensorName)) &&
          ConvertShapeToLength(model.GetTensorShape(tensorName)) > kQuantizedFoldedConstantEmbedMaxLength;
}

std::string QuantizationWorkflow::SupportCode(const RModel &model, std::string_view generatedCode) const
{
   const auto *extension = QuantWorkflowExtension::Find(model);
   if (extension == nullptr || !extension->workflow)
      return {};
   std::string out;
   for (const auto &line : extension->workflow->Report().Summary())
      out += line + "\n";
   return out + QuantCodecSupportCode(generatedCode);
}

void InstallQuantizationWorkflow()
{
   // Idempotent: the static registrar and the parser both call this, and only the first installs.
   if (dynamic_cast<QuantizationWorkflow *>(ActiveModelPlugin()) == nullptr)
      InstallModelPlugin(std::make_unique<QuantizationWorkflow>());
}

namespace {
// Installs the workflow at library load so generation behaves the same for every caller, including
// models built without the parser.
const bool kQuantizationWorkflowInstalled = [] {
   InstallQuantizationWorkflow();
   return true;
}();
} // namespace

} // namespace SOFIE
