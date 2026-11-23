#ifndef SOFIE_RQUANTIZATION_CODEFORMAT
#define SOFIE_RQUANTIZATION_CODEFORMAT

// Code formats: how each encoding's codes are stored, converted on the host and converted in
// generated code. This is the only module that switches on EQuantEncoding to produce code or
// pack codes. Other modules read QuantSpec fields and call these functions.
//
// Integer grids of every bit width share one codec (affine, saturating to [codeMin, codeMax]).
// To support a new integer space:
//   - for wider storage (int16 / int32), add its tensor type to IsBakeableCodeType and PackCodes.
//     Every kernel-grid predicate then accepts it.
//   - for a new float encoding, add its case to each switch in RQuantization_CodeFormat.cxx and
//     its codec to RQuantization_Codec.hxx.

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/quantization/spec/RQuantization_Spec.hxx"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace SOFIE {

// Returns true when SOFIE can bake codes of `type` into an initializer or the weight file and
// read them in generated kernels: int8 / uint8 codes and E4M3 bytes.
bool IsBakeableCodeType(ETensorType type);

// Returns true when the encoding has a codec and its storage type is bakeable, so generated
// kernels can read and write the spec's codes at any granularity.
bool HasKernelCodes(const QuantSpec &spec);

// Returns true when a generated kernel can decode or encode codes on `spec` per tensor:
// HasKernelCodes for a per-tensor spec.
inline bool IsKernelGrid(const QuantSpec &spec)
{
   return spec.IsPerTensor() && HasKernelCodes(spec);
}

// Generated-code expressions. `value` must be a double expression. `scale` and `zeroPoint` are
// C++ expressions, either literals or per-channel table reads.
std::string QuantEncodeExpression(const QuantSpec &spec, const std::string &value, const std::string &scale,
                                  const std::string &zeroPoint);
std::string QuantDecodeExpression(const QuantSpec &spec, const std::string &code, const std::string &scale,
                                  const std::string &zeroPoint);
// Per-tensor overloads that take the spec's parameters as literals.
std::string QuantEncodeExpression(const QuantSpec &spec, const std::string &value);
std::string QuantDecodeExpression(const QuantSpec &spec, const std::string &code);
// Returns the code read on a unit-scale grid with no zero point, as a double: the integer itself or
// the decoded float8 value. Products of these values times sX*sW give the real product.
std::string QuantUnscaledCodeExpression(const QuantSpec &spec, const std::string &code);

// Returns the codec support code for generated code: the SOFIE::QuantCodec functions that
// `generatedCode` calls, with the functions they call, copied from RQuantization_Codec.hxx in
// header order. Empty when the code calls none.
std::string QuantCodecSupportCode(std::string_view generatedCode);
// Returns the text of RQuantization_Codec.hxx, embedded at build time.
std::string_view QuantCodecHeaderText();

// Host conversions on the parameters of `channel`, used for constant folding.
double QuantEncodeValue(const QuantSpec &spec, double value, std::size_t channel);
double QuantDecodeCode(const QuantSpec &spec, double code, std::size_t channel);

// Packs whole-number codes (or E4M3 bytes) into a host buffer of `type`. Throws when `type` is
// not bakeable.
std::shared_ptr<void> PackCodes(ETensorType type, const std::vector<double> &codes);
// Reads `count` codes of `type` from `data` as whole numbers (or E4M3 bytes). Inverse of PackCodes.
std::vector<double> UnpackCodes(ETensorType type, const void *data, std::size_t count);

// Returns true when `spec` is the full-range, zero-point-free, nearest-even per-tensor integer
// grid of `type` (for INT8: codes [-128, 127]). Library epilogues that saturate to `type`
// compute exactly this grid.
bool IsFullRangeIntegerGrid(const QuantSpec &spec, ETensorType type);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_CODEFORMAT
