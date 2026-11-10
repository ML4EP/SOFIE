#ifndef SOFIE_ROPERATOR_QUANTIZEDDENSELINEAR
#define SOFIE_ROPERATOR_QUANTIZEDDENSELINEAR

// Quantized dense linear operator: Y = op(X . W^T) on codes, for Gemm and MatMul. Row-major
// Y[M,N] = X[M,K] . W[N,K]^T is computed as the column-major D[N x M] = op(W)[N x K] . op(X)[K x M]
// with ('t','n'), the 'TN' layout the int8 and float8 BLAS kernels require; W is stored [N,K].
//
//   int8   fused:   one blas.int8Matmul / int8MatmulVectorAlpha / ...StridedBatched call writes
//                   the int8 codes with a float scale, an output-unit bias and the library's
//                   Relu epilogue; requires a plain int8 store (per-channel weights use
//                   int8MatmulVectorAlpha)
//   int8   unfused: int8 x int8 -> int32 accumulator, then the epilogue kernel
//   float8:         blas.matmulFp8Fp32 (A/B scales as device scalars) -> float, then the
//                   epilogue kernel
// The epilogue kernel computes v = scale * (acc - zpX * colsum[n]) + bias[n] in double and
// stores it through the QuantStorePlan.

#include "SOFIE/quantization/lowered/ROperator_QuantLowered.hxx"

namespace SOFIE {

struct QuantizedDenseLinearConfig {
   bool fp8 = false;
   std::size_t batch = 1, M = 0, N = 0, K = 0;
   long long strideX = 0, strideW = 0, strideY = 0;   // per batch entry, in elements
   std::string x, w, y;                               // X codes, W codes [.., N, K], output
   std::string accumulator;                           // Scratch: int32 (int8) or float (float8);
                                                      // empty when the fused call writes the output
   // Epilogue kernel operands; an empty name marks an absent operand.
   double scale = 1.0;            // Per-tensor alpha*sX*sW (int8); 1 for float8, which the library scales
   std::string weightScales;      // Per-channel alpha*sX*sW[n], float [N]
   std::int64_t inputZeroPoint = 0;
   std::string columnSums;        // Weight column sums, int32 [N]; set when inputZeroPoint != 0
   std::string bias;              // beta * bias, float [N]
   QuantStorePlan store;
   // int8 fused path; an empty fusedEpilogue disables it.
   std::string fusedEpilogue;     // "Epilogue::Default" / "Bias" / "Relu" / "ReluBias"
   double fusedAlpha = 1.0;       // Per-tensor alpha*sX*sW/sY
   std::string fusedAlphaVector;  // Per-channel alpha*sX*sW[n]/sY, float [N]
   std::string fusedBias;         // Output-unit bias with zero-point correction, float [N]
   // float8 operand scales, as float [1] device scalars.
   std::string scaleX, scaleW;
};

class ROperator_QuantizedDenseLinear final : public ROperator_QuantLowered {
   QuantizedDenseLinearConfig fC;

   static std::string Null(const char *type) { return "static_cast<" + std::string(type) + " const *>(nullptr)"; }
   std::string OrNull(const std::string &tensor, const char *type) const { return tensor.empty() ? Null(type) : Ptr(tensor); }
   std::string Dims() const
   {
      return "'t', 'n', " + std::to_string(fC.N) + ", " + std::to_string(fC.M) + ", " + std::to_string(fC.K);
   }

   std::string Fp8Ptr(const std::string &tensor) const
   {
      return "reinterpret_cast<typename std::remove_reference_t<decltype(blas)>::Fp8E4M3Type const *>(" +
             Ptr(tensor) + ")";
   }

   std::string FusedCall() const
   {
      const std::string bias = OrNull(fC.fusedBias, "float");
      if (fC.batch > 1)
         return "blas.int8MatmulStridedBatched(" + Dims() + ", " + ExactDoubleLiteral(fC.fusedAlpha) + "f, " +
                Ptr(fC.w) + ", " + std::to_string(fC.strideW) + "LL, " + Ptr(fC.x) + ", " + std::to_string(fC.strideX) +
                "LL, " + bias + ", " + Ptr(fC.y) + ", " + std::to_string(fC.strideY) + "LL, " +
                std::to_string(fC.batch) + ", " + fC.fusedEpilogue + ")";
      if (!fC.fusedAlphaVector.empty())
         return "blas.int8MatmulVectorAlpha(" + Dims() + ", " + Ptr(fC.fusedAlphaVector) + ", " + Ptr(fC.w) + ", " +
                Ptr(fC.x) + ", " + bias + ", " + Ptr(fC.y) + ", " + fC.fusedEpilogue + ")";
      return "blas.int8Matmul(" + Dims() + ", " + ExactDoubleLiteral(fC.fusedAlpha) + "f, " + Ptr(fC.w) + ", " +
             Ptr(fC.x) + ", " + bias + ", " + Ptr(fC.y) + ", " + fC.fusedEpilogue + ")";
   }

   std::string AccumulateCall() const
   {
      const std::string acc = Ptr(fC.accumulator);
      if (fC.fp8) {
         const std::string bf16Null =
            "static_cast<typename std::remove_reference_t<decltype(blas)>::Bf16Type const *>(nullptr)";
         if (fC.batch > 1)
            return "blas.matmulFp8Fp32StridedBatched(" + Dims() + ", 1.0f, " + Fp8Ptr(fC.w) + ", " + Ptr(fC.scaleW) +
                   ", " + std::to_string(fC.strideW) + "LL, " + Fp8Ptr(fC.x) + ", " + Ptr(fC.scaleX) + ", " +
                   std::to_string(fC.strideX) + "LL, " + bf16Null + ", " + acc + ", " + std::to_string(fC.strideY) +
                   "LL, " + std::to_string(fC.batch) + ", Epilogue::Default)";
         return "blas.matmulFp8Fp32(" + Dims() + ", 1.0f, " + Fp8Ptr(fC.w) + ", " + Ptr(fC.scaleW) + ", " +
                Fp8Ptr(fC.x) + ", " + Ptr(fC.scaleX) + ", " + bf16Null + ", " + acc + ", Epilogue::Default)";
      }
      if (fC.batch > 1)
         return "blas.int8MatmulStridedBatched(" + Dims() + ", " + Ptr(fC.w) + ", " + std::to_string(fC.strideW) +
                "LL, " + Ptr(fC.x) + ", " + std::to_string(fC.strideX) + "LL, " + acc + ", " +
                std::to_string(fC.strideY) + "LL, " + std::to_string(fC.batch) + ")";
      return "blas.int8Matmul(" + Dims() + ", " + Ptr(fC.w) + ", " + Ptr(fC.x) + ", " + acc + ")";
   }

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
      std::string out = "\n//------ QUANTIZED_DENSE_LINEAR " + opName + " (" + (fC.fp8 ? "e4m3" : "int8") +
                        ", batch " + std::to_string(fC.batch) + ", M=" + std::to_string(fC.M) +
                        " N=" + std::to_string(fC.N) + " K=" + std::to_string(fC.K) + ")\n";
      if (fC.fp8)
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
