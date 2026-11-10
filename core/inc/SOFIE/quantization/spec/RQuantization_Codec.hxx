#ifndef SOFIE_RQUANTIZATION_CODEC
#define SOFIE_RQUANTIZATION_CODEC

// Provides portable encode/decode of quantization codes as constexpr arithmetic with no vendor
// intrinsics. The same functions run on the host (constant folding at code-generation time) and in
// Alpaka kernels on every backend, through --expt-relaxed-constexpr on CUDA and natively on HIP and CPU.

#include <cmath>
#include <cstdint>

// Marks a codec function callable from host code and from GPU device code.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SOFIE_QUANT_CODEC_FN __host__ __device__ constexpr
#else
#define SOFIE_QUANT_CODEC_FN constexpr
#endif

namespace SOFIE {
namespace QuantCodec {

// Rounds half to even (the ONNX QuantizeLinear rounding) without <cmath>, which is not constexpr.
SOFIE_QUANT_CODEC_FN double RoundHalfEven(double value)
{
   const bool negative = value < 0.0;
   const double magnitude = negative ? -value : value;
   // Every double at or beyond 2^52 is an integer.
   if (!(magnitude < 4503599627370496.0))
      return value;
   const double floorMagnitude = static_cast<double>(static_cast<std::int64_t>(magnitude));
   const double fraction = magnitude - floorMagnitude;
   double rounded = floorMagnitude;
   if (fraction > 0.5 || (fraction == 0.5 && static_cast<std::int64_t>(floorMagnitude) % 2 != 0))
      rounded += 1.0;
   return negative ? -rounded : rounded;
}

SOFIE_QUANT_CODEC_FN double Floor(double value)
{
   const double truncated = static_cast<double>(static_cast<std::int64_t>(value));
   return truncated > value ? truncated - 1.0 : truncated;
}

SOFIE_QUANT_CODEC_FN double Truncate(double value)
{
   return static_cast<double>(static_cast<std::int64_t>(value));
}

// Rounds with mode 0 = nearest-even, 1 = floor, 2 = truncate (the order of EQuantRounding).
SOFIE_QUANT_CODEC_FN double Round(double value, int rounding)
{
   return rounding == 1 ? Floor(value) : (rounding == 2 ? Truncate(value) : RoundHalfEven(value));
}

// Encodes onto the integer grid value = (code - zeroPoint) * scale, saturating to [codeMin, codeMax].
SOFIE_QUANT_CODEC_FN double EncodeInteger(double value, double scale, double zeroPoint, double codeMin, double codeMax,
                               int rounding = 0)
{
   double code = Round(value / scale, rounding) + zeroPoint;
   code = code < codeMin ? codeMin : code;
   return code > codeMax ? codeMax : code;
}

SOFIE_QUANT_CODEC_FN double DecodeInteger(double code, double scale, double zeroPoint)
{
   return (code - zeroPoint) * scale;
}

// Decodes an OCP E4M3 (e4m3fn) byte: 1 sign, 4 exponent (bias 7) and 3 mantissa bits, no
// infinities, S.1111.111 is NaN. The largest finite magnitude is 448.
SOFIE_QUANT_CODEC_FN double DecodeE4M3(std::uint8_t bits)
{
   const double sign = (bits & 0x80u) ? -1.0 : 1.0;
   const unsigned exponent = (bits >> 3) & 0x0Fu;
   const unsigned mantissa = bits & 0x07u;
   if (exponent == 0x0Fu && mantissa == 0x07u)
      return static_cast<double>(NAN);
   // Subnormal: mantissa/8 * 2^-6.
   if (exponent == 0)
      return sign * static_cast<double>(mantissa) * (1.0 / 512.0);
   double power = 1.0;
   for (int e = 7; e < static_cast<int>(exponent); ++e)
      power *= 2.0;
   for (int e = static_cast<int>(exponent); e < 7; ++e)
      power *= 0.5;
   return sign * (1.0 + static_cast<double>(mantissa) / 8.0) * power;
}

// Encodes to E4M3 with round-to-nearest-even, saturating to the largest finite value (the ONNX
// saturate default and the __nv_fp8_e4m3 conversion). NaN encodes to NaN.
SOFIE_QUANT_CODEC_FN std::uint8_t EncodeE4M3(double value)
{
   if (value != value)
      return 0x7Fu;
   const std::uint8_t sign = value < 0.0 ? 0x80u : 0x00u;
   const double magnitude = value < 0.0 ? -value : value;
   if (magnitude >= 448.0)
      return static_cast<std::uint8_t>(sign | 0x7Eu);
   if (magnitude < 1.0 / 64.0) {
      // Subnormal range, step 2^-9. A rounded mantissa of 8 is exactly the smallest normal.
      const auto mantissa = static_cast<unsigned>(RoundHalfEven(magnitude * 512.0));
      return static_cast<std::uint8_t>(sign | mantissa);
   }
   // Normal range: finds e with 2^e <= magnitude < 2^(e+1), e in [-6, 8].
   int exponent = 0;
   double power = 1.0;
   while (magnitude >= 2.0 * power) {
      power *= 2.0;
      ++exponent;
   }
   while (magnitude < power) {
      power *= 0.5;
      --exponent;
   }
   auto mantissa = static_cast<unsigned>(RoundHalfEven(magnitude / power * 8.0));   // in [8, 16]
   if (mantissa == 16u) {
      mantissa = 8u;
      ++exponent;
   }
   // 1.875 * 2^8 = 480 is the NaN pattern, so values that round there saturate.
   if (exponent > 8 || (exponent == 8 && mantissa > 14u))
      return static_cast<std::uint8_t>(sign | 0x7Eu);
   return static_cast<std::uint8_t>(sign | (static_cast<unsigned>(exponent + 7) << 3) | (mantissa - 8u));
}

// Encodes onto a scaled E4M3 grid: divides in double, rounds to float (the value a float tensor
// holds), then applies the saturating encode. Every float8 encode in generated code, boundary or
// fused, uses this function.
SOFIE_QUANT_CODEC_FN std::uint8_t EncodeE4M3Scaled(double value, double scale)
{
   return EncodeE4M3(static_cast<double>(static_cast<float>(value / scale)));
}

} // namespace QuantCodec
} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_CODEC
