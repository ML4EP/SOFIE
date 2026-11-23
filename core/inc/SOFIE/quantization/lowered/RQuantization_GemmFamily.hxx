#ifndef SOFIE_RQUANTIZATION_GEMMFAMILY
#define SOFIE_RQUANTIZATION_GEMMFAMILY

// Matrix-product families: the code formats the lowered Dense and Conv operators multiply with
// sofieBLAS. A family fixes the operand storage the library reads, the accumulator it writes,
// the vendor alignment and the call that computes acc = op(A) . op(B). Handlers pick a family
// with SelectGemmFamily and ask its traits for everything that depends on the format. They never
// test a bit width or an encoding themselves.
//
// To support another integer space:
//   1. add an EQuantGemmFamily value and its row in GemmTraits (RQuantization_GemmFamily.cxx);
//   2. return it from SelectGemmFamily for the operand grids it covers;
//   3. emit its sofieBLAS call in QuantGemmAccumulateCall (and add the entry point to sofieBLAS);
//   4. if its operand storage is new, make it bakeable (RQuantization_CodeFormat.hxx).
// Dense and Conv then lower the new family with no further changes.

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/quantization/spec/RQuantization_Spec.hxx"

#include <cstddef>
#include <optional>
#include <string>

namespace SOFIE {

enum class EQuantGemmFamily : std::uint8_t {
   Int8,   // int8 x int8 -> int32 (blas.int8Matmul*)
   E4M3,   // e4m3 x e4m3 -> float (blas.matmulFp8Fp32*)
};

struct QuantGemmTraits {
   EQuantGemmFamily family;
   const char *label;               // Name used in generated comments.
   ETensorType operandType;         // Storage of both operands' codes.
   ETensorType accumulatorType;     // INT32 or FLOAT.
   std::size_t kAlignment;          // Alignment the vendor kernels need for K
   std::size_t nAlignment;          // and N; other shapes run on sofieBLAS's native kernels.
   // Integer products accumulate exactly. Zero points are corrected through the weight's column
   // sums or subtracted in the direct kernel, and padding holds the input zero point.
   bool integerAccumulate;
   // The call takes the operand scales as float device scalars and returns real values, so the
   // epilogue adds no sX*sW factor and Gemm alpha must be 1.
   bool scalesInLibrary;
   // The library has an epilogue that writes full-range operandType codes directly (scale, bias,
   // Relu, saturate); see ROperator_QuantizedDenseLinear's fused path.
   bool fusedCodesOutput;
   // The call reads OCP float8 bytes; FNUZ-only targets are rejected at compile time.
   bool requiresOcpFloat8;
};

const QuantGemmTraits &GemmTraits(EQuantGemmFamily family);

// Returns the family that multiplies codes on `x` (per-tensor) by codes on `w`, or none. Zero
// points and per-channel weights are checked by the operator, which knows its correction paths.
std::optional<EQuantGemmFamily> SelectGemmFamily(const QuantSpec &x, const QuantSpec &w);

// Operands of one accumulate call, computing the column-major D[m x n] = op(A)[m x k] . op(B)[k x n]
// with ('t', 'n'). Expressions are C++ code: pointers to the operands' storage, sizes as literals.
struct QuantGemmCall {
   std::string m, n, k;
   std::string a, b, d;                  // A and B codes, D accumulator
   bool batched = false;                 // Strided-batched call (the strides and batch apply)
   std::string strideA, strideB, strideD;
   std::size_t batch = 1;
   std::string scaleA, scaleB;           // Device scalars when the family has scalesInLibrary
};

// Returns the sofieBLAS call (without a trailing ';') that writes the family's accumulator.
std::string QuantGemmAccumulateCall(const QuantGemmTraits &traits, const QuantGemmCall &call);

// Library epilogue of a codes-output call: D = saturate(alpha * acc + bias), with Relu when the
// epilogue names it. Expressions are C++ code.
struct QuantGemmFusedOutput {
   std::string alpha;          // Per-tensor float scale
   std::string alphaVector;    // Per-row float scales [m] (unbatched only); replaces alpha when set
   std::string bias;           // float [m] pointer, or a typed nullptr
   std::string epilogue;       // "Epilogue::Default" / "Bias" / "Relu" / "ReluBias"
};

// Returns the call that writes operandType codes to call.d directly. Requires fusedCodesOutput.
std::string QuantGemmFusedCodesCall(const QuantGemmTraits &traits, const QuantGemmCall &call,
                                    const QuantGemmFusedOutput &fused);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_GEMMFAMILY
