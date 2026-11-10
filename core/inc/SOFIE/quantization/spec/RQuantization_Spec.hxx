#ifndef SOFIE_RQUANTIZATION_SPEC
#define SOFIE_RQUANTIZATION_SPEC

// Defines the quantization vocabulary of the workflow: the grid a tensor's values lie on
// (QuantSpec) and the physical form a tensor takes (EQuantForm). Frontends translate into it and
// handlers read it, so no code past the frontends depends on whether a model is ONNX Q/DQ or QONNX.

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/SOFIE_common.hxx"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace SOFIE {

// Mapping from a code to a value. Integer grids are affine; float8 grids carry only a scale,
// since the format fixes their code set.
enum class EQuantEncoding { Undefined = 0, Integer = 1, FP8E4M3 = 2, FP8E5M2 = 3 };

enum class EQuantRounding { NearestEven = 0, Floor = 1, Truncate = 2 };

// Quantization grid value = (code - zeroPoint[c]) * scale[c] with codes in [codeMin, codeMax]. One
// entry in scales/zeroPoints is per-tensor; more entries are per-channel along `axis`.
struct QuantSpec {
   EQuantEncoding encoding = EQuantEncoding::Undefined;
   unsigned bitWidth = 0;
   bool isSigned = false;
   std::vector<double> scales;
   std::vector<std::int64_t> zeroPoints;
   int axis = -1;
   // Extremes of the code set; narrow range and absorbed clamps tighten them.
   double codeMin = 0.0;
   double codeMax = 0.0;
   EQuantRounding rounding = EQuantRounding::NearestEven;
   // Initializers that hold the parameters, so generated code reads per-channel parameters from
   // device buffers instead of embedding them.
   std::string scaleTensor;
   std::string zeroPointTensor;

   bool IsDefined() const { return encoding != EQuantEncoding::Undefined && !scales.empty(); }
   bool IsFloatingPoint() const
   {
      return encoding == EQuantEncoding::FP8E4M3 || encoding == EQuantEncoding::FP8E5M2;
   }
   bool IsPerTensor() const { return scales.size() <= 1 && zeroPoints.size() <= 1; }
   std::size_t ChannelCount() const { return std::max(scales.size(), zeroPoints.size()); }
   double Scale(std::size_t channel = 0) const { return scales.size() == 1 ? scales[0] : scales.at(channel); }
   std::int64_t ZeroPoint(std::size_t channel = 0) const
   {
      if (zeroPoints.empty())
         return 0;
      return zeroPoints.size() == 1 ? zeroPoints[0] : zeroPoints.at(channel);
   }
   // Returns the tensor type that stores this grid's codes.
   ETensorType CodeType() const;
};

// Builds a spec for each encoding. Integer ranges follow bit width, signedness and narrow range;
// float8 ranges span the format's largest finite values.
QuantSpec MakeIntegerQuantSpec(unsigned bitWidth, bool isSigned, bool narrow, std::vector<double> scales,
                               std::vector<std::int64_t> zeroPoints, int axis, EQuantRounding rounding);
QuantSpec MakeFloat8QuantSpec(EQuantEncoding encoding, double scale);

// Returns true when a value encoded on `a` decodes unchanged on `b`: same encoding, parameters,
// code range and rounding. Source tensor names are ignored, so equal parameters from different
// initializers agree.
bool SameGrid(const QuantSpec &a, const QuantSpec &b);

std::string ToString(const QuantSpec &spec);

// Returns the grid core operators fuse onto (ROperator::FuseDequantizedInput / FuseOutputOnGrid).
// Accepts per-tensor specs only.
QuantizationGrid ToQuantizationGrid(const QuantSpec &spec);

// Physical form of a tensor. The model fixes the grid; the workflow chooses the form, and every
// optimization is expressed as a choice of form.
enum class EQuantForm : std::uint8_t {
   Float = 1,         // Plain float with no grid guarantee.
   Snapped = 2,       // Float that lies on the spec's grid (a fake quant).
   Codes = 4,         // The spec's codes (int8 / uint8 / float8 bytes).
};

const char *ToString(EQuantForm form);

// Forms in order of preference; negotiation picks the first form both sides share.
using QuantForms = std::vector<EQuantForm>;

inline bool Contains(const QuantForms &forms, EQuantForm form)
{
   for (auto f : forms)
      if (f == form)
         return true;
   return false;
}

// Quantization state of one tensor.
struct TensorQuantState {
   std::optional<QuantSpec> spec;            // Grid the model fixes; empty when there is none.
   EQuantForm form = EQuantForm::Float;      // Form of the tensor in the parsed graph.
   bool constant = false;                    // True for an initializer, whose codes are bakeable.

   bool HasGrid() const { return spec.has_value() && spec->IsDefined(); }
};

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_SPEC
