#include "SOFIE/quantization/lowered/ROperator_QuantLowered.hxx"

#include "SOFIE/RQuantization.hxx"

namespace SOFIE {

namespace {

std::string EncodeExpression(const QuantSpec &spec, const std::string &value)
{
   if (spec.IsFloatingPoint())
      return "SOFIE::QuantCodec::EncodeE4M3Scaled(" + value + ", " + ExactDoubleLiteral(spec.Scale()) + ")";
   return "SOFIE::QuantCodec::EncodeInteger(" + value + ", " + ExactDoubleLiteral(spec.Scale()) + ", " +
          std::to_string(spec.ZeroPoint()) + ".0, " + ExactDoubleLiteral(spec.codeMin) + ", " +
          ExactDoubleLiteral(spec.codeMax) + ", " + std::to_string(static_cast<int>(spec.rounding)) + ")";
}

std::string DecodeExpression(const QuantSpec &spec, const std::string &code)
{
   if (spec.IsFloatingPoint())
      return "(SOFIE::QuantCodec::DecodeE4M3(static_cast<std::uint8_t>(" + code + ")) * " +
             ExactDoubleLiteral(spec.Scale()) + ")";
   return "SOFIE::QuantCodec::DecodeInteger(static_cast<double>(" + code + "), " + ExactDoubleLiteral(spec.Scale()) +
          ", " + std::to_string(spec.ZeroPoint()) + ".0)";
}

} // namespace

std::string QuantLoadExpression(const QuantBinding &binding, const std::string &element)
{
   if (binding.form == EQuantForm::Codes && binding.spec)
      return DecodeExpression(*binding.spec, element);
   return "static_cast<double>(" + element + ")";
}

QuantStorePlan QuantStorePlan::For(const QuantBinding &binding)
{
   QuantStorePlan plan;
   plan.form = binding.form == EQuantForm::Codes || binding.form == EQuantForm::Snapped ? binding.form
                                                                                         : EQuantForm::Float;
   if (plan.form != EQuantForm::Float)
      plan.spec = binding.spec;
   return plan;
}

std::string QuantStorePlan::Statements(const std::string &valueVar, const std::string &dest,
                                       const std::string &indent) const
{
   std::string out;
   if (snapBefore)
      out += indent + valueVar + " = " + DecodeExpression(*snapBefore, EncodeExpression(*snapBefore, valueVar)) + ";\n";
   if (relu)
      out += indent + valueVar + " = " + valueVar + " < 0.0 ? 0.0 : " + valueVar + ";\n";
   if (clamp)
      out += indent + valueVar + " = " + valueVar + " < " + ExactDoubleLiteral(clamp->first) + " ? " +
             ExactDoubleLiteral(clamp->first) + " : (" + valueVar + " > " + ExactDoubleLiteral(clamp->second) + " ? " +
             ExactDoubleLiteral(clamp->second) + " : " + valueVar + ");\n";
   switch (form) {
   case EQuantForm::Codes:
      out += indent + dest + " = static_cast<std::remove_reference_t<decltype(" + dest + ")>>(" +
             EncodeExpression(*spec, valueVar) + ");\n";
      break;
   case EQuantForm::Snapped:
      out += indent + dest + " = static_cast<float>(" + DecodeExpression(*spec, EncodeExpression(*spec, valueVar)) +
             ");\n";
      break;
   default:
      out += indent + dest + " = static_cast<float>(" + valueVar + ");\n";
      break;
   }
   return out;
}

bool QuantStorePlan::IsPlainInt8Codes() const
{
   if (form != EQuantForm::Codes || !spec || clamp)
      return false;
   const auto &s = *spec;
   const bool int8 = s.encoding == EQuantEncoding::Integer && s.bitWidth == 8 && s.isSigned && s.IsPerTensor() &&
                     s.ZeroPoint() == 0 && s.codeMin == -128.0 && s.codeMax == 127.0 &&
                     s.rounding == EQuantRounding::NearestEven;
   // The encode re-snaps an absorbed pair on the same grid; a pair on another grid needs a separate snap.
   return int8 && (!snapBefore || SameGrid(*snapBefore, s));
}

} // namespace SOFIE
