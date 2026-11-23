#include "SOFIE/quantization/lowered/ROperator_QuantLowered.hxx"

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/quantization/spec/RQuantization_CodeFormat.hxx"

namespace SOFIE {

std::string QuantLoadExpression(const QuantBinding &binding, const std::string &element)
{
   if (binding.form == EQuantForm::Codes && binding.spec)
      return QuantDecodeExpression(*binding.spec, element);
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
      out += indent + valueVar + " = " + QuantDecodeExpression(*snapBefore, QuantEncodeExpression(*snapBefore, valueVar)) + ";\n";
   if (relu)
      out += indent + valueVar + " = " + valueVar + " < 0.0 ? 0.0 : " + valueVar + ";\n";
   if (clamp)
      out += indent + valueVar + " = " + valueVar + " < " + ExactDoubleLiteral(clamp->first) + " ? " +
             ExactDoubleLiteral(clamp->first) + " : (" + valueVar + " > " + ExactDoubleLiteral(clamp->second) + " ? " +
             ExactDoubleLiteral(clamp->second) + " : " + valueVar + ");\n";
   switch (form) {
   case EQuantForm::Codes:
      out += indent + dest + " = static_cast<std::remove_reference_t<decltype(" + dest + ")>>(" +
             QuantEncodeExpression(*spec, valueVar) + ");\n";
      break;
   case EQuantForm::Snapped:
      out += indent + dest + " = static_cast<float>(" + QuantDecodeExpression(*spec, QuantEncodeExpression(*spec, valueVar)) +
             ");\n";
      break;
   default:
      out += indent + dest + " = static_cast<float>(" + valueVar + ");\n";
      break;
   }
   return out;
}

bool QuantStorePlan::IsPlainCodes(ETensorType type) const
{
   if (form != EQuantForm::Codes || !spec || clamp || !IsFullRangeIntegerGrid(*spec, type))
      return false;
   // The encode re-snaps an absorbed pair on the same grid; a pair on another grid needs a separate snap.
   return !snapBefore || SameGrid(*snapBefore, *spec);
}

} // namespace SOFIE
