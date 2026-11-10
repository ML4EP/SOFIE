#include "SOFIE/quantization/spec/RQuantization_Spec.hxx"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace SOFIE {

ETensorType QuantSpec::CodeType() const
{
   switch (encoding) {
   case EQuantEncoding::FP8E4M3: return ETensorType::FLOAT8E4M3FN;
   case EQuantEncoding::FP8E5M2: return ETensorType::FLOAT8E5M2;
   case EQuantEncoding::Integer:
      if (bitWidth <= 8)
         return isSigned ? ETensorType::INT8 : ETensorType::UINT8;
      if (bitWidth <= 16)
         return isSigned ? ETensorType::INT16 : ETensorType::UINT16;
      if (bitWidth <= 32)
         return isSigned ? ETensorType::INT32 : ETensorType::UINT32;
      break;
   case EQuantEncoding::Undefined: break;
   }
   throw std::runtime_error("SOFIE quantization spec has no code type: " + ToString(*this));
}

QuantSpec MakeIntegerQuantSpec(unsigned bitWidth, bool isSigned, bool narrow, std::vector<double> scales,
                               std::vector<std::int64_t> zeroPoints, int axis, EQuantRounding rounding)
{
   if (bitWidth == 0 || bitWidth > 32)
      throw std::runtime_error("SOFIE quantization spec bit width must be in [1, 32]");
   if (scales.empty())
      throw std::runtime_error("SOFIE quantization spec needs at least one scale");
   QuantSpec spec;
   spec.encoding = EQuantEncoding::Integer;
   spec.bitWidth = bitWidth;
   spec.isSigned = isSigned;
   spec.scales = std::move(scales);
   spec.zeroPoints = zeroPoints.empty() ? std::vector<std::int64_t>{0} : std::move(zeroPoints);
   spec.axis = spec.IsPerTensor() ? -1 : axis;
   spec.rounding = rounding;
   if (isSigned) {
      const double qmax = static_cast<double>((std::int64_t{1} << (bitWidth - 1)) - 1);
      spec.codeMax = qmax;
      spec.codeMin = narrow ? -qmax : -qmax - 1.0;
   } else {
      spec.codeMax = static_cast<double>((std::int64_t{1} << bitWidth) - 1);
      spec.codeMin = narrow ? 1.0 : 0.0;
   }
   return spec;
}

QuantSpec MakeFloat8QuantSpec(EQuantEncoding encoding, double scale)
{
   if (encoding != EQuantEncoding::FP8E4M3 && encoding != EQuantEncoding::FP8E5M2)
      throw std::runtime_error("SOFIE float8 quantization spec needs a float8 encoding");
   QuantSpec spec;
   spec.encoding = encoding;
   spec.bitWidth = 8;
   spec.isSigned = true;
   spec.scales = {scale};
   spec.zeroPoints = {0};
   const double limit = encoding == EQuantEncoding::FP8E5M2 ? 57344.0 : 448.0;
   spec.codeMin = -limit;
   spec.codeMax = limit;
   return spec;
}

bool SameGrid(const QuantSpec &a, const QuantSpec &b)
{
   if (!a.IsDefined() || !b.IsDefined())
      return false;
   return a.encoding == b.encoding && a.bitWidth == b.bitWidth && a.isSigned == b.isSigned &&
          a.scales == b.scales && a.zeroPoints == b.zeroPoints && a.axis == b.axis &&
          a.codeMin == b.codeMin && a.codeMax == b.codeMax && a.rounding == b.rounding;
}

std::string ToString(const QuantSpec &spec)
{
   std::ostringstream out;
   switch (spec.encoding) {
   case EQuantEncoding::Integer: out << (spec.isSigned ? "int" : "uint") << spec.bitWidth; break;
   case EQuantEncoding::FP8E4M3: out << "fp8e4m3"; break;
   case EQuantEncoding::FP8E5M2: out << "fp8e5m2"; break;
   case EQuantEncoding::Undefined: return "undefined";
   }
   if (spec.IsPerTensor())
      out << " scale=" << ExactDoubleLiteral(spec.Scale()) << " zp=" << spec.ZeroPoint();
   else
      out << " per-channel(axis=" << spec.axis << ", n=" << spec.ChannelCount() << ")";
   out << " codes=[" << spec.codeMin << "," << spec.codeMax << "]";
   return out.str();
}

namespace {
EQuantizationRoundingMode RoundingTo(EQuantRounding rounding)
{
   switch (rounding) {
   case EQuantRounding::Floor: return EQuantizationRoundingMode::FLOOR;
   case EQuantRounding::Truncate: return EQuantizationRoundingMode::TRUNCATE;
   case EQuantRounding::NearestEven: break;
   }
   return EQuantizationRoundingMode::ROUND;
}
} // namespace

QuantizationGrid ToQuantizationGrid(const QuantSpec &spec)
{
   if (!spec.IsPerTensor())
      throw std::runtime_error("SOFIE QuantizationGrid describes per-tensor grids only");
   QuantizationGrid grid;
   switch (spec.encoding) {
   case EQuantEncoding::Integer: grid.kind = EQuantizationGridKind::Integer; break;
   case EQuantEncoding::FP8E4M3: grid.kind = EQuantizationGridKind::Float8E4M3; break;
   case EQuantEncoding::FP8E5M2: grid.kind = EQuantizationGridKind::Float8E5M2; break;
   case EQuantEncoding::Undefined: return grid;
   }
   grid.scale = spec.Scale();
   grid.zeroPoint = spec.ZeroPoint();
   grid.granularity = EQuantizationGranularity::PerTensor;
   grid.rounding = RoundingTo(spec.rounding);
   grid.codeMin = spec.codeMin;
   grid.codeMax = spec.codeMax;
   return grid;
}

const char *ToString(EQuantForm form)
{
   switch (form) {
   case EQuantForm::Float: return "float";
   case EQuantForm::Snapped: return "snapped";
   case EQuantForm::Codes: return "codes";
   }
   return "unknown";
}

} // namespace SOFIE
