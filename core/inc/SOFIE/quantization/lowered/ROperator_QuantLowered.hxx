#ifndef SOFIE_ROPERATOR_QUANTLOWERED
#define SOFIE_ROPERATOR_QUANTLOWERED

// Base class and load/store helpers for lowered quantized operators. Handlers build these
// operators after Initialize; they exist only in the lowered view and generate Alpaka code,
// with every matrix product emitted as a call on the session's sofieBLAS object (`blas`).

#include "SOFIE/ROperator.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/quantization/handlers/RQuantization_Handler.hxx"
#include "SOFIE/quantization/spec/RQuantization_Spec.hxx"

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SOFIE {

// Returns the device expression that reads `element` of an operand binding as a double.
// Codes are decoded on their grid; plain and snapped floats are read unchanged.
std::string QuantLoadExpression(const QuantBinding &binding, const std::string &element);

// Store recipe for a lowered operator's result. The arithmetic yields the real value `double v`;
// the plan applies the absorbed operators, then writes the output form:
//   snap onto an absorbed boundary pair's grid -> Relu -> Clip -> float / snapped / codes.
struct QuantStorePlan {
   std::optional<QuantSpec> snapBefore;
   bool relu = false;
   std::optional<std::pair<double, double>> clamp;
   EQuantForm form = EQuantForm::Float;   // Float, Snapped or Codes
   std::optional<QuantSpec> spec;         // Output grid; set for Snapped and Codes

   // Returns the plan that writes `binding` with nothing absorbed.
   static QuantStorePlan For(const QuantBinding &binding);
   // Returns the statements that convert `valueVar` (a double) into the value stored in `dest`.
   std::string Statements(const std::string &valueVar, const std::string &dest, const std::string &indent) const;
   // Returns true when the store is exactly "round(v / scale) saturated to int8" (zero point 0,
   // full range, nearest-even, no clamp), which a BLAS int8-output epilogue computes.
   bool IsPlainInt8Codes() const;
};

class ROperator_QuantLowered : public ROperator {
protected:
   static std::string Ptr(const std::string &tensor) { return "alpaka::getPtrNative(deviceBuf_" + tensor + ")"; }

   // Emits a compile-time check placed before a float8 BLAS call. SOFIE's codec writes OCP E4M3FN
   // bytes, which a FNUZ-only HIP target (sofieBLAS reports Fp8IsFnuz) decodes differently; the
   // check rejects such builds.
   std::string RequireOcpFloat8() const
   {
      return SP + "if constexpr (requires { std::remove_reference_t<decltype(blas)>::Fp8IsFnuz; })\n" + SP + SP +
             "static_assert(!std::remove_reference_t<decltype(blas)>::Fp8IsFnuz, \"SOFIE quantizes to OCP "
             "E4M3FN, but this HIP target uses the FNUZ float8 formats\");\n";
   }

   // Emits a 1D launch of one thread per element; the kernel takes (acc, args..., n).
   std::string Launch1D(const std::string &opName, const std::string &kernelInstance, std::size_t elements,
                        const std::string &args) const
   {
      std::stringstream out;
      out << SP << "{\n";
      out << SP << SP << "auto const workDiv = sofie_workdiv(Vec::all(Idx{" << elements << "}));\n";
      out << SP << SP << "alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, " << kernelInstance << ", "
          << args << ", static_cast<std::size_t>(" << elements << ")));\n";
      out << SP << "}\n";
      (void)opName;
      return out.str();
   }

public:
   // No-op: handlers build lowered operators from an already initialized model.
   void Initialize(RModel &) override {}
   std::string Generate(std::string) override
   {
      throw std::runtime_error("SOFIE quantized operator " + fName +
                               " generates Alpaka code only; plain C++ generation keeps the float path");
   }
   std::vector<ETensorType> TypeInference(std::vector<ETensorType>) override { return {}; }
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>>) override { return {}; }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTLOWERED
