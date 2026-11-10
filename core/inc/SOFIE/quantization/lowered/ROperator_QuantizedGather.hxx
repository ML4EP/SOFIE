#ifndef SOFIE_ROPERATOR_QUANTIZEDGATHER
#define SOFIE_ROPERATOR_QUANTIZEDGATHER

// Quantized Gather from a table held as codes (an embedding table of int8 / float8 weights).
// The output copies the gathered codes when it stays on the table's grid; otherwise the decoded
// values are stored per a QuantStorePlan. A per-channel table is decoded with the scale of each
// element's channel. Negative indices count from the end, as in ONNX.

#include "SOFIE/quantization/lowered/ROperator_QuantLowered.hxx"

namespace SOFIE {

class ROperator_QuantizedGather final : public ROperator_QuantLowered {
   QuantBinding fTable;
   std::string fIndices;
   std::string fY;
   QuantStorePlan fStore;
   std::vector<std::size_t> fTableShape;
   std::size_t fAxis = 0;
   std::size_t fIndexCount = 0;
   std::string fChannelScales;   // Per-channel table scales, float [channels]; empty for per-tensor
   std::size_t fChannelStride = 1, fChannelCount = 1;

   bool CopiesCodes() const
   {
      return fTable.form == EQuantForm::Codes && fStore.form == EQuantForm::Codes && fTable.spec && fStore.spec &&
             !fStore.snapBefore && !fStore.relu && !fStore.clamp && SameGrid(*fTable.spec, *fStore.spec);
   }

public:
   ROperator_QuantizedGather(QuantBinding table, std::string indices, std::string y, QuantStorePlan store,
                             std::vector<std::size_t> tableShape, std::size_t axis, std::size_t indexCount,
                             std::string channelScales = {}, std::size_t channelStride = 1, std::size_t channelCount = 1)
      : fTable(std::move(table)), fIndices(std::move(indices)), fY(std::move(y)), fStore(std::move(store)),
        fTableShape(std::move(tableShape)), fAxis(axis), fIndexCount(indexCount),
        fChannelScales(std::move(channelScales)), fChannelStride(channelStride), fChannelCount(channelCount)
   {
      fInputTensorNames = {fTable.tensor, fIndices};
      if (!fChannelScales.empty())
         fInputTensorNames.push_back(fChannelScales);
      fOutputTensorNames = {fY};
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      std::size_t inner = 1;
      for (std::size_t i = fAxis + 1; i < fTableShape.size(); ++i)
         inner *= fTableShape[i];
      const std::size_t axisLength = fTableShape[fAxis];
      std::string op = "\n//------ QUANTIZED_GATHER_KERNEL_ALPAKA " + opName + "\n";
      op += SP + "struct QuantizedGatherKernel_" + opName + " {\n";
      op += SP + SP + "template<typename TAcc, typename TT, typename TI, typename TS, typename TOut>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TT const * table, TI const * indices, "
                      "TS const * channelScales, TOut * y, std::size_t const n) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= n) return;\n";
      // Decomposes idx into (outer, position, inner) over [outer, indexCount, inner].
      op += SP + SP + SP + "std::size_t const innerIdx = idx % " + std::to_string(inner) + ";\n";
      op += SP + SP + SP + "std::size_t const position = (idx / " + std::to_string(inner) + ") % " +
            std::to_string(fIndexCount) + ";\n";
      op += SP + SP + SP + "std::size_t const outer = idx / " + std::to_string(inner * fIndexCount) + ";\n";
      op += SP + SP + SP + "long long index = static_cast<long long>(indices[position]);\n";
      op += SP + SP + SP + "if (index < 0) index += " + std::to_string(axisLength) + ";\n";
      op += SP + SP + SP + "std::size_t const source = (outer * " + std::to_string(axisLength) +
            " + static_cast<std::size_t>(index)) * " + std::to_string(inner) + " + innerIdx;\n";
      if (CopiesCodes()) {
         op += SP + SP + SP + "y[idx] = static_cast<TOut>(table[source]);\n";
      } else {
         std::string load = QuantLoadExpression(fTable, "table[source]");
         if (!fChannelScales.empty()) {
            // Per-channel table: decodes symmetric codes with their channel's scale.
            op += SP + SP + SP + "std::size_t const channel = (source / " + std::to_string(fChannelStride) + ") % " +
                  std::to_string(fChannelCount) + ";\n";
            load = fTable.spec && fTable.spec->IsFloatingPoint()
                      ? "(SOFIE::QuantCodec::DecodeE4M3(static_cast<std::uint8_t>(table[source])) * "
                        "static_cast<double>(channelScales[channel]))"
                      : "(static_cast<double>(table[source]) * static_cast<double>(channelScales[channel]))";
         }
         op += SP + SP + SP + "double v = " + load + ";\n";
         op += fStore.Statements("v", "y[idx]", SP + SP + SP);
      }
      op += SP + SP + "}\n" + SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      return SP + "QuantizedGatherKernel_" + opName + " quantizedGatherKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      std::size_t outer = 1, inner = 1;
      for (std::size_t i = 0; i < fAxis; ++i)
         outer *= fTableShape[i];
      for (std::size_t i = fAxis + 1; i < fTableShape.size(); ++i)
         inner *= fTableShape[i];
      const std::string scales =
         fChannelScales.empty() ? "static_cast<float const *>(nullptr)" : Ptr(fChannelScales);
      return "\n//------ QUANTIZED_GATHER " + opName + "\n" +
             Launch1D(opName, "quantizedGatherKernel_" + opName, outer * fIndexCount * inner,
                      Ptr(fTable.tensor) + ", " + Ptr(fIndices) + ", " + scales + ", " + Ptr(fY));
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZEDGATHER
