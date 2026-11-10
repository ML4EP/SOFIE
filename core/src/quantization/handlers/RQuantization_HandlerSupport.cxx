#include "SOFIE/quantization/handlers/RQuantization_HandlerSupport.hxx"

#include "SOFIE/ROperator_Clip.hxx"
#include "SOFIE/ROperator_Relu.hxx"
#include "SOFIE/quantization/frontend/ROperator_QuantBoundary.hxx"

namespace SOFIE {

namespace {
// Returns the sole emitted reader of `tensor`, or null when it has several readers or is a
// graph output.
const ROperator *SoleReader(const QuantContext &ctx, const std::string &tensor, std::size_t &index)
{
   const auto &graph = ctx.Graph();
   const auto &readers = graph.Consumers(tensor);
   if (readers.size() != 1 || graph.IsGraphOutput(tensor))
      return nullptr;
   index = readers[0];
   return graph.Op(index);
}
} // namespace

std::optional<AbsorbedEpilogue> FindAbsorbableEpilogue(const QuantContext &ctx, const std::string &start)
{
   AbsorbedEpilogue epilogue;
   std::string tensor = start;
   std::size_t index = 0;

   // The output's own fake quant: an encode read only by a decode on the same grid.
   if (const auto *encode = dynamic_cast<const ROperator_QuantEncode *>(SoleReader(ctx, tensor, index))) {
      const std::size_t encodeIndex = index;
      const auto *decode = dynamic_cast<const ROperator_QuantDecode *>(SoleReader(ctx, encode->GetOutputTensor(), index));
      if (encode->IsView() || decode == nullptr || decode->IsView() || !SameGrid(encode->GetSpec(), decode->GetSpec()))
         return std::nullopt;
      epilogue.snap = encode->GetSpec();
      epilogue.absorption.operators = {encodeIndex, index};
      tensor = decode->GetOutputTensor();
   }

   for (int step = 0; step < 2; ++step) {
      const ROperator *reader = SoleReader(ctx, tensor, index);
      if (const auto *relu = dynamic_cast<const ROperator_Relu<float> *>(reader); relu != nullptr && !epilogue.relu) {
         epilogue.relu = true;
      } else if (const auto *clip = dynamic_cast<const ROperator_Clip<float> *>(reader);
                 clip != nullptr && !epilogue.clamp && clip->ClipBoundsAreConstant()) {
         epilogue.clamp = std::make_pair(static_cast<double>(clip->ClipMin()), static_cast<double>(clip->ClipMax()));
      } else {
         break;
      }
      epilogue.absorption.operators.push_back(index);
      tensor = std::string(reader->GetOpOutputTensors()[0]);
   }
   if (!epilogue.relu && !epilogue.clamp)
      return std::nullopt;
   epilogue.absorption.output = tensor;
   return epilogue;
}

QuantStorePlan StorePlanFor(const QuantBinding &output, const std::optional<AbsorbedEpilogue> &epilogue)
{
   auto plan = QuantStorePlan::For(output);
   if (epilogue) {
      plan.snapBefore = epilogue->snap;
      plan.relu = epilogue->relu;
      plan.clamp = epilogue->clamp;
   }
   return plan;
}

} // namespace SOFIE
