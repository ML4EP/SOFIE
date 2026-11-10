#ifndef SOFIE_RQUANTIZATION
#define SOFIE_RQUANTIZATION

// Provides the quantization types base SOFIE uses: the per-tensor grid an operator decodes at its
// load or encodes at its store (ROperator::FuseDequantizedInput / FuseOutputOnGrid), and the
// generated code for those conversions. Frontends, specs, handlers and lowering live in the
// quantization library, core/inc/SOFIE/quantization/.
//
// The emitted code calls SOFIE::QuantCodec (quantization/spec/RQuantization_Codec.hxx), a portable
// header the quantization workflow adds to every quantized model's includes.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SOFIE {

enum class EQuantizationRoundingMode { UNDEFINED = 0, ROUND = 1, FLOOR = 2, TRUNCATE = 3 };

enum class EQuantizationGranularity { UNDEFINED = 0, PerTensor = 1, PerChannel = 2, PerWeight = 3 };

// Returns a round-trippable literal. The default ostream precision truncates a scale to six
// significant digits, which shifts every quantized value.
inline std::string ExactDoubleLiteral(double value)
{
   std::ostringstream out;
   out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
   return out.str();
}

// Kind of grid a fused boundary encodes onto. Integer and float8 grids share
// value = (code - zeroPoint) * scale and differ only in the set of codes.
enum class EQuantizationGridKind { Undefined = 0, Integer = 1, Float8E4M3 = 2, Float8E5M2 = 3 };

struct QuantizationGrid {
   EQuantizationGridKind kind = EQuantizationGridKind::Undefined;
   double scale = 1.0;
   std::int64_t zeroPoint = 0;   // Always 0 for float8, which has no zero point.
   EQuantizationGranularity granularity = EQuantizationGranularity::PerTensor;
   EQuantizationRoundingMode rounding = EQuantizationRoundingMode::UNDEFINED;
   // Extremes of the representable code set: -128/127 for int8, -448/448 for E4M3.
   double codeMin = 0.0;
   double codeMax = 0.0;

   bool IsDefined() const { return kind != EQuantizationGridKind::Undefined; }
   bool IsFloatingPoint() const
   {
      return kind == EQuantizationGridKind::Float8E4M3 || kind == EQuantizationGridKind::Float8E5M2;
   }
};

// Returns true for a per-tensor E4M3 grid, the only float8 grid operators decode at their load.
inline bool IsPerTensorE4M3(const QuantizationGrid &grid)
{
   return grid.kind == EQuantizationGridKind::Float8E4M3 &&
          grid.granularity == EQuantizationGranularity::PerTensor && grid.zeroPoint == 0 && grid.scale > 0.0;
}

// Element count above which a folded quantization constant goes to the weight file; scale
// scalars, zero points and per-channel parameters fall below it and are embedded.
inline constexpr std::size_t kQuantizedFoldedConstantEmbedMaxLength = 4096;

// Implements the output-adopting hooks: the operator takes the boundary's output tensor in place
// of its own, which dead-code elimination removes.
inline void AdoptFusedOutputName(std::string &outputName, std::vector<std::string> &outputList,
                                 const std::string &adopted, const char *who)
{
   if (outputList.empty() || outputList[0] != outputName) {
      throw std::runtime_error(std::string("TMVA::SOFIE ") + who +
                               " cannot adopt a fused output: its output tensor list does not begin with " +
                               outputName);
   }
   outputName = adopted;
   outputList[0] = adopted;
}

// Returns the integer encode a QuantEncode boundary computes: round half to even, add the zero
// point, saturate to the code range.
inline std::string IntegerEncodeExpression(const std::string &valueExpr, const QuantizationGrid &grid)
{
   return "SOFIE::QuantCodec::EncodeInteger(static_cast<double>(" + valueExpr + "), " +
          ExactDoubleLiteral(grid.scale) + ", " + std::to_string(grid.zeroPoint) + ".0, " +
          ExactDoubleLiteral(grid.codeMin) + ", " + ExactDoubleLiteral(grid.codeMax) + ")";
}

// Returns the float8 encode: the scale division in double, rounded to float, then the saturating
// E4M3 encode.
inline std::string FP8EncodeExpression(const std::string &valueExpr, const QuantizationGrid &grid)
{
   return "SOFIE::QuantCodec::EncodeE4M3Scaled(static_cast<double>(" + valueExpr + "), " +
          ExactDoubleLiteral(grid.scale) + ")";
}

// Emits a fused fake-quant boundary: an encode onto the grid and a decode straight back,
// declaring `resultVar` (a snapped float).
inline std::string FakeQuantRoundTripStatements(const std::string &resultVar, const std::string &valueExpr,
                                                const QuantizationGrid &grid, const std::string &indent)
{
   const std::string scale = ExactDoubleLiteral(grid.scale);
   if (grid.IsFloatingPoint())
      return indent + "float const " + resultVar + " = static_cast<float>(SOFIE::QuantCodec::DecodeE4M3(" +
             FP8EncodeExpression(valueExpr, grid) + ") * " + scale + ");\n";
   return indent + "double const " + resultVar + " = SOFIE::QuantCodec::DecodeInteger(" +
          IntegerEncodeExpression(valueExpr, grid) + ", " + scale + ", " + std::to_string(grid.zeroPoint) +
          ".0);\n";
}

// Emits the encode-only counterpart, which writes the code instead of the snapped float.
inline std::string EncodeToGridStatements(const std::string &destExpr, const std::string &valueExpr,
                                          const QuantizationGrid &grid, const std::string &indent)
{
   const std::string code =
      grid.IsFloatingPoint() ? FP8EncodeExpression(valueExpr, grid) : IntegerEncodeExpression(valueExpr, grid);
   return indent + destExpr + " = static_cast<std::remove_reference_t<decltype(" + destExpr + ")>>(" + code +
          ");\n";
}

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION
