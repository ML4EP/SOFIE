#include "SOFIE/quantization/diagnostics/RQuantization_Diagnostics.hxx"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace SOFIE {

const QuantizationOptions &QuantizationOptions::FromEnvironment()
{
   static const QuantizationOptions options = [] {
      QuantizationOptions o;
      o.trace = std::getenv("SOFIE_QUANT_TRACE") != nullptr;
      return o;
   }();
   return options;
}

void QuantWorkflowReport::Record(const std::string &pass, const std::string &subject,
                                 const std::string &message) const
{
   if (QuantizationOptions::FromEnvironment().trace)
      std::cerr << "[SOFIE quant:" << pass << "] " << subject << ": " << message << "\n";
}

std::vector<std::string> QuantWorkflowReport::Summary() const
{
   return {"// SOFIE quantization workflow: " + std::to_string(operatorsLowered) + " operators lowered, " +
           std::to_string(boundariesEliminated) + " boundaries eliminated, " + std::to_string(boundariesKept) +
           " kept (SOFIE_QUANT_TRACE=1 explains each)"};
}

} // namespace SOFIE
