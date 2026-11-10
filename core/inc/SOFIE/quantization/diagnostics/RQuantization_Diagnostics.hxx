#ifndef SOFIE_RQUANTIZATION_DIAGNOSTICS
#define SOFIE_RQUANTIZATION_DIAGNOSTICS

// Provides options and reporting for the quantization workflow. Options are read from the
// environment once, in one place, for all passes.
//   SOFIE_QUANT_TRACE   prints every workflow event to stderr when set

#include <cstddef>
#include <string>
#include <vector>

namespace SOFIE {

struct QuantizationOptions {
   bool trace = false;

   static const QuantizationOptions &FromEnvironment();
};

// Outcome of a workflow run: three counts for the generated header, plus the event trace.
struct QuantWorkflowReport {
   std::size_t boundariesEliminated = 0;   // Boundaries absorbed by a negotiated form.
   std::size_t boundariesKept = 0;         // Boundaries emitted as encode/decode kernels.
   std::size_t operatorsLowered = 0;       // Operators replaced or rewired by a handler.

   // Records one workflow decision (pass, the operator or tensor it concerns, and the reason).
   // Prints it to stderr when SOFIE_QUANT_TRACE is set and stores nothing.
   void Record(const std::string &pass, const std::string &subject, const std::string &message) const;
   // Returns the summary line for the generated header, starting with "//".
   std::vector<std::string> Summary() const;
};

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_DIAGNOSTICS
