#ifndef SOFIE_ROPERATOR_QUANTIZEDDENSELINEAR
#define SOFIE_ROPERATOR_QUANTIZEDDENSELINEAR

// Quantized dense linear operator: Y = op(X . W^T) on codes, for Gemm and MatMul. Row-major
// Y[M,N] = X[M,K] . W[N,K]^T is computed as the column-major D[N x M] = op(W)[N x K] . op(X)[K x M]
// with ('t','n'), the 'TN' layout the quantized BLAS kernels require; W is stored [N,K]. The GEMM
// family (RQuantization_GemmFamily.hxx) fixes the operand codes and the library call:
//
//   fused:    families with fusedCodesOutput (int8): one blas.int8Matmul / int8MatmulVectorAlpha /
//             ...StridedBatched call writes the codes with a float scale, an output-unit bias and
//             the library's Relu epilogue; requires a plain full-range store (per-channel weights
//             use int8MatmulVectorAlpha)
//   unfused:  the family's accumulate call (int8 -> int32, e4m3 -> float with the A/B scales as
//             device scalars), then the epilogue kernel
// The epilogue kernel computes v = scale * (acc - zpX * colsum[n]) + bias[n] in double and
// stores it through the QuantStorePlan.

#include "SOFIE/quantization/lowered/ROperator_QuantLowered.hxx"
#include "SOFIE/quantization/lowered/RQuantization_GemmFamily.hxx"

namespace SOFIE {

struct QuantizedDenseLinearConfig {
   EQuantGemmFamily family = EQuantGemmFamily::Int8;
   std::size_t batch = 1, M = 0, N = 0, K = 0;
   long long strideX = 0, strideW = 0, strideY = 0;   // per batch entry, in elements
   std::string x, w, y;                               // X codes, W codes [.., N, K], output
   std::string accumulator;                           // Scratch of the family's accumulator type;
                                                      // empty when the fused call writes the output
   // Epilogue kernel operands; an empty name marks an absent operand.
   double scale = 1.0;            // Per-tensor alpha*sX*sW; 1 when the library applies the scales
   std::string weightScales;      // Per-channel alpha*sX*sW[n], float [N]
   std::int64_t inputZeroPoint = 0;
   std::string columnSums;        // Weight column sums, int32 [N]; set when inputZeroPoint != 0
   std::string bias;              // beta * bias, float [N]
   QuantStorePlan store;
   // Fused codes-output path; an empty fusedEpilogue disables it.
   std::string fusedEpilogue;     // "Epilogue::Default" / "Bias" / "Relu" / "ReluBias"
   double fusedAlpha = 1.0;       // Per-tensor alpha*sX*sW/sY
   std::string fusedAlphaVector;  // Per-channel alpha*sX*sW[n]/sY, float [N]
   std::string fusedBias;         // Output-unit bias with zero-point correction, float [N]
   // Operand scales as float [1] device scalars, for families with scalesInLibrary.
   std::string scaleX, scaleW;
};

class ROperator_QuantizedDenseLinear final : public ROperator_QuantLowered {
   QuantizedDenseLinearConfig fC;

   const QuantGemmTraits &Traits() const { return GemmTraits(fC.family); }


   // Returns the call's operands with D = `d`: W as A and X as B, batched over the leading dims.
   QuantGemmCall Call(const std::string &d) const
   {
      QuantGemmCall call;
      call.m = std::to_string(fC.N);
      call.n = std::to_string(fC.M);
      call.k = std::to_string(fC.K);
      call.a = Ptr(fC.w);
      call.b = Ptr(fC.x);
      call.d = d;
      call.batched = fC.batch > 1;
      call.strideA = std::to_string(fC.strideW);
      call.strideB = std::to_string(fC.strideX);
      call.strideD = std::to_string(fC.strideY);
      call.batch = fC.batch;
      if (Traits().scalesInLibrary) {
         call.scaleA = Ptr(fC.scaleW);
         call.scaleB = Ptr(fC.scaleX);
      }
      return call;
   }

   std::string FusedCall() const
   {
      QuantGemmFusedOutput fused;
      fused.alpha = ExactDoubleLiteral(fC.fusedAlpha) + "f";
      if (!fC.fusedAlphaVector.empty())
         fused.alphaVector = Ptr(fC.fusedAlphaVector);
      fused.bias = OrNull(fC.fusedBias, "float");
      fused.epilogue = fC.fusedEpilogue;
      return QuantGemmFusedCodesCall(Traits(), Call(Ptr(fC.y)), fused);
   }

   std::string AccumulateCall() const { return QuantGemmAccumulateCall(Traits(), Call(Ptr(fC.accumulator))); }

public:
   explicit ROperator_QuantizedDenseLinear(QuantizedDenseLinearConfig config) : fC(std::move(config))
   {
      fKind = OperatorKind::QUANTIZED_GEMM;
      fInputTensorNames = {fC.x, fC.w};
      for (const auto *t : {&fC.weightScales, &fC.columnSums, &fC.bias, &fC.fusedAlphaVector, &fC.fusedBias,
                            &fC.scaleX, &fC.scaleW})
         if (!t->empty())
            fInputTensorNames.push_back(*t);
      fOutputTensorNames = {fC.y};
      if (!fC.accumulator.empty())
         fOutputTensorNames.push_back(fC.accumulator);
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override
   {
      if (fC.accumulator.empty())
         return "";
      opName = "op_" + opName;
      const std::string n = std::to_string(fC.N);
      std::string op = "\n//------ QUANTIZED_DENSE_EPILOGUE_KERNEL_ALPAKA " + opName + "\n";
      op += SP + "struct QuantizedDenseEpilogueKernel_" + opName + " {\n";
      op += SP + SP + "template<typename TAcc, typename TA, typename TS, typename TC, typename TB, typename TOut>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TA const * accumulator, TS const * scales, "
                      "TC const * columnSums, TB const * bias, TOut * y, std::size_t const n) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= n) return;\n";
      op += SP + SP + SP + "std::size_t const col = idx % " + n + ";\n";
      std::string sum = "static_cast<double>(accumulator[idx])";
      if (!fC.columnSums.empty())
         sum = "(" + sum + " - " + std::to_string(fC.inputZeroPoint) + ".0 * static_cast<double>(columnSums[col]))";
      const std::string scale = fC.weightScales.empty() ? ExactDoubleLiteral(fC.scale)
                                                        : "static_cast<double>(scales[col])";
      op += SP + SP + SP + "double v = " + scale + " * " + sum +
            (fC.bias.empty() ? "" : " + static_cast<double>(bias[col])") + ";\n";
      op += fC.store.Statements("v", "y[idx]", SP + SP + SP);
      op += SP + SP + "}\n" + SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      if (fC.accumulator.empty())
         return "";
      opName = "op_" + opName;
      return SP + "QuantizedDenseEpilogueKernel_" + opName + " quantizedDenseEpilogueKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      std::string out = "\n//------ QUANTIZED_DENSE_LINEAR " + opName + " (" + Traits().label +
                        ", batch " + std::to_string(fC.batch) + ", M=" + std::to_string(fC.M) +
                        " N=" + std::to_string(fC.N) + " K=" + std::to_string(fC.K) + ")\n";
      if (Traits().requiresOcpFloat8)
         out += RequireOcpFloat8();
      if (fC.accumulator.empty())
         return out + SP + FusedCall() + ";\n";
      const std::size_t elements = fC.batch * fC.M * fC.N;
      const std::string unfused =
         SP + AccumulateCall() + ";\n" +
         Launch1D(opName, "quantizedDenseEpilogueKernel_" + opName, elements,
                  Ptr(fC.accumulator) + ", " + OrNull(fC.weightScales, "float") + ", " +
                     OrNull(fC.columnSums, "std::int32_t") + ", " + OrNull(fC.bias, "float") + ", " + Ptr(fC.y));
      return out + unfused;
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZEDDENSELINEAR
