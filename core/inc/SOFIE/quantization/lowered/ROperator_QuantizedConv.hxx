#ifndef SOFIE_ROPERATOR_QUANTIZEDCONV
#define SOFIE_ROPERATOR_QUANTIZEDCONV

// Quantized Conv1D / Conv2D (NCHW, groups, padding, stride, dilation) on codes.
//
//   direct:  one thread per output element accumulates (x - zpX)(w - zpW) over its window
//            (int8, exact in 64-bit integers) or decoded E4M3 products (double), then stores.
//            Handles every shape; the only path for depthwise and asymmetric weights.
//   im2col:  per batch entry, the input windows are laid out as col[group][P][Kp] (Kp = the
//            window length padded to the library's alignment), then one strided-batched
//            sofieBLAS call over the groups computes acc[group][OCg][P] = Wg . col^T
//            ('t','n', m = P, n = OCg, k = Kp), which is NCHW; the epilogue kernel scales,
//            corrects the input zero point through the weight's column sums, adds the bias
//            and stores. The handler plans this path only for shapes the library has kernels for.

#include "SOFIE/quantization/lowered/ROperator_QuantLowered.hxx"

namespace SOFIE {

struct QuantizedConvConfig {
   bool fp8 = false;
   std::size_t N = 1, C = 0, H = 1, W = 1, OC = 0, OH = 1, OW = 1, KH = 1, KW = 1, groups = 1;
   std::size_t strideH = 1, strideW = 1, padH = 0, padW = 0, dilationH = 1, dilationW = 1;
   std::string x, w, y;           // X codes, W codes [OC, Cg, KH, KW], output
   QuantStorePlan store;
   // Scales: int8 per-tensor alpha*sX*sW literal or per-channel float [OC]; float8 sX*sW.
   double scale = 1.0;
   std::string weightScales;
   std::int64_t inputZeroPoint = 0;
   std::string weightZeroPoints;  // int32 [OC] for asymmetric weights; direct path only
   std::string bias;              // float [OC]
   // im2col path operands; the path is disabled when weightMatrix is empty.
   std::string weightMatrix;      // W codes as [groups, OCg, Kp], zero padded
   std::string columnSums;        // Weight column sums, int32 [OC]; set when inputZeroPoint != 0
   std::string columns, accumulator;
   std::size_t Kp = 0;
   std::string scaleX, scaleW;    // float8 operand scales, as device scalars for the library
};

class ROperator_QuantizedConv final : public ROperator_QuantLowered {
   QuantizedConvConfig fC;

   std::size_t Cg() const { return fC.C / fC.groups; }
   std::size_t OCg() const { return fC.OC / fC.groups; }
   std::size_t P() const { return fC.OH * fC.OW; }
   std::size_t Kg() const { return Cg() * fC.KH * fC.KW; }
   static std::string Null(const char *type) { return "static_cast<" + std::string(type) + " const *>(nullptr)"; }
   std::string OrNull(const std::string &t, const char *type) const { return t.empty() ? Null(type) : Ptr(t); }
   static std::string L(std::size_t v) { return std::to_string(v); }

   // Emits the loop that accumulates one output element's window products into `sum`;
   // positions in the padding contribute nothing.
   std::string WindowSum() const
   {
      std::string s;
      const std::string I = SP + SP + SP;
      s += I + "for (std::size_t c = 0; c < " + L(Cg()) + "; ++c) {\n";
      s += I + "   std::size_t const ic = g * " + L(Cg()) + " + c;\n";
      s += I + "   for (std::size_t kh = 0; kh < " + L(fC.KH) + "; ++kh) {\n";
      s += I + "      long long const ih = static_cast<long long>(oh * " + L(fC.strideH) + " + kh * " +
           L(fC.dilationH) + ") - " + L(fC.padH) + ";\n";
      s += I + "      if (ih < 0 || ih >= " + L(fC.H) + ") continue;\n";
      s += I + "      for (std::size_t kw = 0; kw < " + L(fC.KW) + "; ++kw) {\n";
      s += I + "         long long const iw = static_cast<long long>(ow * " + L(fC.strideW) + " + kw * " +
           L(fC.dilationW) + ") - " + L(fC.padW) + ";\n";
      s += I + "         if (iw < 0 || iw >= " + L(fC.W) + ") continue;\n";
      s += I + "         std::size_t const xi = ((n * " + L(fC.C) + " + ic) * " + L(fC.H) +
           " + static_cast<std::size_t>(ih)) * " + L(fC.W) + " + static_cast<std::size_t>(iw);\n";
      s += I + "         std::size_t const wi = ((oc * " + L(Cg()) + " + c) * " + L(fC.KH) + " + kh) * " + L(fC.KW) +
           " + kw;\n";
      if (fC.fp8) {
         s += I + "         sum += SOFIE::QuantCodec::DecodeE4M3(static_cast<std::uint8_t>(x[xi])) * "
                  "SOFIE::QuantCodec::DecodeE4M3(static_cast<std::uint8_t>(w[wi]));\n";
      } else {
         const std::string wz = fC.weightZeroPoints.empty() ? "" : " - static_cast<long long>(weightZeroPoints[oc])";
         s += I + "         sum += (static_cast<long long>(x[xi]) - " + std::to_string(fC.inputZeroPoint) +
              "LL) * (static_cast<long long>(w[wi])" + wz + ");\n";
      }
      s += I + "      }\n" + I + "   }\n" + I + "}\n";
      return s;
   }

   std::string ScaleExpr() const
   {
      return fC.weightScales.empty() ? ExactDoubleLiteral(fC.scale) : "static_cast<double>(scales[oc])";
   }

   bool HasMatrixPath() const { return !fC.weightMatrix.empty(); }

   std::string MatrixCall() const
   {
      const std::string m = L(P()), n = L(OCg()), k = L(fC.Kp);
      const std::string strideA = L(P() * fC.Kp), strideB = L(OCg() * fC.Kp), strideC = L(OCg() * P());
      const std::string col = Ptr(fC.columns) + " + b * " + L(fC.groups * P() * fC.Kp);
      const std::string acc = Ptr(fC.accumulator) + " + b * " + L(fC.OC * P());
      if (fC.fp8) {
         const std::string t = "typename std::remove_reference_t<decltype(blas)>::";
         return "blas.matmulFp8Fp32StridedBatched('t', 'n', " + m + ", " + n + ", " + k + ", 1.0f, reinterpret_cast<" +
                t + "Fp8E4M3Type const *>(" + col + "), " + Ptr(fC.scaleX) + ", " + strideA + "LL, reinterpret_cast<" +
                t + "Fp8E4M3Type const *>(" + Ptr(fC.weightMatrix) + "), " + Ptr(fC.scaleW) + ", " + strideB +
                "LL, static_cast<" + t + "Bf16Type const *>(nullptr), " + acc + ", " + strideC + "LL, " +
                L(fC.groups) + ", Epilogue::Default)";
      }
      return "blas.int8MatmulStridedBatched('t', 'n', " + m + ", " + n + ", " + k + ", " + col + ", " + strideA +
             "LL, " + Ptr(fC.weightMatrix) + ", " + strideB + "LL, " + acc + ", " + strideC + "LL, " + L(fC.groups) + ")";
   }

public:
   explicit ROperator_QuantizedConv(QuantizedConvConfig config) : fC(std::move(config))
   {
      fKind = OperatorKind::QUANTIZED_CONV;
      fInputTensorNames = {fC.x, fC.w};
      for (const auto *t : {&fC.weightScales, &fC.weightZeroPoints, &fC.bias, &fC.weightMatrix, &fC.columnSums,
                            &fC.scaleX, &fC.scaleW})
         if (!t->empty())
            fInputTensorNames.push_back(*t);
      fOutputTensorNames = {fC.y};
      if (HasMatrixPath()) {
         fOutputTensorNames.push_back(fC.columns);
         fOutputTensorNames.push_back(fC.accumulator);
      }
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      const std::string I = SP + SP + SP;
      const std::string accType = fC.fp8 ? "double" : "long long";
      // Decomposes `idx` into (n, oc, oh, ow) of the NCHW output and the group g.
      const std::string coords = I + "std::size_t const ow = idx % " + L(fC.OW) + ";\n" + I + "std::size_t const oh = (idx / " +
                                 L(fC.OW) + ") % " + L(fC.OH) + ";\n" + I + "std::size_t const oc = (idx / " + L(P()) +
                                 ") % " + L(fC.OC) + ";\n" + I + "std::size_t const n = idx / " + L(fC.OC * P()) + ";\n" +
                                 I + "std::size_t const g = oc / " + L(OCg()) + ";\n";
      const std::string bias = fC.bias.empty() ? "" : " + static_cast<double>(bias[oc])";
      const std::string sumScale = fC.fp8 ? ExactDoubleLiteral(fC.scale) : ScaleExpr();

      std::string op;
      if (!HasMatrixPath()) {
         op += "\n//------ QUANTIZED_CONV_DIRECT_KERNEL_ALPAKA " + opName + "\n";
         op += SP + "struct QuantizedConvDirectKernel_" + opName + " {\n";
         op += SP + SP + "template<typename TAcc, typename TX, typename TW, typename TS, typename TZ, typename TB, "
                         "typename TOut>\n";
         op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TX const * x, TW const * w, TS const * scales, "
                         "TZ const * weightZeroPoints, TB const * bias, TOut * y, std::size_t const total) const {\n";
         op += I + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += I + "if (idx >= total) return;\n" + coords;
         op += I + accType + " sum = 0;\n" + WindowSum();
         op += I + "double v = " + sumScale + " * static_cast<double>(sum)" + bias + ";\n";
         op += fC.store.Statements("v", "y[idx]", I);
         op += SP + SP + "}\n" + SP + "};\n";
      } else {
         // col[(n * groups + g) * P + p][kk], kk = (c * KH + kh) * KW + kw; padding holds the zero point.
         op += "\n//------ QUANTIZED_CONV_IM2COL_KERNEL_ALPAKA " + opName + "\n";
         op += SP + "struct QuantizedConvIm2ColKernel_" + opName + " {\n";
         op += SP + SP + "template<typename TAcc, typename TX, typename TCol>\n";
         op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TX const * x, TCol * col, "
                         "std::size_t const total) const {\n";
         op += I + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += I + "if (idx >= total) return;\n";
         op += I + "std::size_t const kk = idx % " + L(fC.Kp) + ";\n";
         op += I + "std::size_t const p = (idx / " + L(fC.Kp) + ") % " + L(P()) + ";\n";
         op += I + "std::size_t const g = (idx / " + L(fC.Kp * P()) + ") % " + L(fC.groups) + ";\n";
         op += I + "std::size_t const n = idx / " + L(fC.Kp * P() * fC.groups) + ";\n";
         const std::string outside = fC.fp8 ? "0" : std::to_string(fC.inputZeroPoint);
         op += I + "if (kk >= " + L(Kg()) + ") { col[idx] = static_cast<TCol>(" + outside + "); return; }\n";
         op += I + "std::size_t const kw = kk % " + L(fC.KW) + ", kh = (kk / " + L(fC.KW) + ") % " + L(fC.KH) +
               ", c = kk / " + L(fC.KW * fC.KH) + ";\n";
         op += I + "std::size_t const ow = p % " + L(fC.OW) + ", oh = p / " + L(fC.OW) + ";\n";
         op += I + "long long const ih = static_cast<long long>(oh * " + L(fC.strideH) + " + kh * " + L(fC.dilationH) +
               ") - " + L(fC.padH) + ";\n";
         op += I + "long long const iw = static_cast<long long>(ow * " + L(fC.strideW) + " + kw * " + L(fC.dilationW) +
               ") - " + L(fC.padW) + ";\n";
         op += I + "if (ih < 0 || ih >= " + L(fC.H) + " || iw < 0 || iw >= " + L(fC.W) + ") { col[idx] = static_cast<TCol>(" +
               outside + "); return; }\n";
         op += I + "col[idx] = static_cast<TCol>(x[((n * " + L(fC.C) + " + g * " + L(Cg()) + " + c) * " + L(fC.H) +
               " + static_cast<std::size_t>(ih)) * " + L(fC.W) + " + static_cast<std::size_t>(iw)]);\n";
         op += SP + SP + "}\n" + SP + "};\n";

         op += "\n//------ QUANTIZED_CONV_EPILOGUE_KERNEL_ALPAKA " + opName + "\n";
         op += SP + "struct QuantizedConvEpilogueKernel_" + opName + " {\n";
         op += SP + SP + "template<typename TAcc, typename TA, typename TS, typename TC, typename TB, typename TOut>\n";
         op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TA const * accumulator, TS const * scales, "
                         "TC const * columnSums, TB const * bias, TOut * y, std::size_t const total) const {\n";
         op += I + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += I + "if (idx >= total) return;\n";
         op += I + "std::size_t const oc = (idx / " + L(P()) + ") % " + L(fC.OC) + ";\n";
         std::string sum = "static_cast<double>(accumulator[idx])";
         if (!fC.columnSums.empty())
            sum = "(" + sum + " - " + std::to_string(fC.inputZeroPoint) + ".0 * static_cast<double>(columnSums[oc]))";
         op += I + "double v = " + (fC.fp8 ? std::string("1.0") : ScaleExpr()) + " * " + sum + bias + ";\n";
         op += fC.store.Statements("v", "y[idx]", I);
         op += SP + SP + "}\n" + SP + "};\n";
      }
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      if (!HasMatrixPath())
         return SP + "QuantizedConvDirectKernel_" + opName + " quantizedConvDirectKernel_" + opName + ";\n";
      return SP + "QuantizedConvIm2ColKernel_" + opName + " quantizedConvIm2ColKernel_" + opName + ";\n" + SP +
             "QuantizedConvEpilogueKernel_" + opName + " quantizedConvEpilogueKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      const std::size_t outputs = fC.N * fC.OC * P();
      std::string out = "\n//------ QUANTIZED_CONV " + opName + " (" + (fC.fp8 ? "e4m3" : "int8") + ", groups " +
                        L(fC.groups) + ", " + L(fC.OC) + "x" + L(Cg()) + "x" + L(fC.KH) + "x" + L(fC.KW) + ")\n";
      if (HasMatrixPath() && fC.fp8)
         out += RequireOcpFloat8();
      if (!HasMatrixPath())
         return out + Launch1D(opName, "quantizedConvDirectKernel_" + opName, outputs,
                               Ptr(fC.x) + ", " + Ptr(fC.w) + ", " + OrNull(fC.weightScales, "float") + ", " +
                                  OrNull(fC.weightZeroPoints, "std::int32_t") + ", " + OrNull(fC.bias, "float") +
                                  ", " + Ptr(fC.y));
      out += Launch1D(opName, "quantizedConvIm2ColKernel_" + opName, fC.N * fC.groups * P() * fC.Kp,
                      Ptr(fC.x) + ", " + Ptr(fC.columns));
      out += SP + "for (std::size_t b = 0; b < " + L(fC.N) + "; ++b)\n";
      out += SP + SP + MatrixCall() + ";\n";
      out += Launch1D(opName, "quantizedConvEpilogueKernel_" + opName, outputs,
                      Ptr(fC.accumulator) + ", " + OrNull(fC.weightScales, "float") + ", " +
                         OrNull(fC.columnSums, "std::int32_t") + ", " + OrNull(fC.bias, "float") + ", " + Ptr(fC.y));
      return out;
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZEDCONV
