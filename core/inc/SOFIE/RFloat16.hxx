#ifndef SOFIE_RFLOAT16_HXX
#define SOFIE_RFLOAT16_HXX

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#define SOFIE_HALF_HD __host__ __device__
#elif defined(__CUDACC__)
#include <cuda_fp16.h>
#define SOFIE_HALF_HD __host__ __device__
#else
#define SOFIE_HALF_HD
#endif

namespace SOFIE {
struct alignas(2) Float16 {
   std::uint16_t bits;
   SOFIE_HALF_HD constexpr Float16() = default;
   SOFIE_HALF_HD static constexpr Float16 FromBits(std::uint16_t value)
   {
      Float16 result;
      result.bits = value;
      return result;
   }
   SOFIE_HALF_HD Float16(float value)
   {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
      bits = __half_as_ushort(__float2half_rn(value));
#else
      std::uint32_t word;
      std::memcpy(&word, &value, sizeof(word));
      const auto sign = static_cast<std::uint16_t>((word >> 16) & 0x8000);
      const auto exponent = static_cast<int>((word >> 23) & 255);
      const auto fraction = word & 0x7fffff;
      if (exponent == 255) {
         bits = sign | (fraction ? 0x7e00 : 0x7c00);
      } else if (exponent > 142) {
         bits = sign | 0x7c00;
      } else if (exponent < 102) {
         bits = sign;
      } else {
         const unsigned shift = exponent < 113 ? 126 - exponent : 13;
         const std::uint32_t mantissa = exponent < 113 ? fraction | 0x800000 : fraction;
         const std::uint32_t rounded = (mantissa + ((std::uint32_t{1} << (shift - 1)) - 1) +
                                       ((mantissa >> shift) & 1)) >> shift;
         bits = sign | static_cast<std::uint16_t>((exponent < 113 ? 0 : (exponent - 112) << 10) + rounded);
      }
#endif
   }
   SOFIE_HALF_HD operator float() const
   {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
      return __half2float(__ushort_as_half(bits));
#else
      std::uint32_t exponent = (bits >> 10) & 31;
      std::uint32_t fraction = bits & 1023;
      std::uint32_t word = static_cast<std::uint32_t>(bits & 0x8000) << 16;
      if (exponent == 31) {
         word |= 0x7f800000 | (fraction << 13);
      } else if (exponent != 0) {
         word |= ((exponent + 112) << 23) | (fraction << 13);
      } else if (fraction != 0) {
         exponent = 113;
         while ((fraction & 1024) == 0) { fraction <<= 1; --exponent; }
         word |= (exponent << 23) | ((fraction & 1023) << 13);
      }
      float result;
      std::memcpy(&result, &word, sizeof(result));
      return result;
#endif
   }
   SOFIE_HALF_HD operator float() const volatile { return static_cast<float>(FromBits(bits)); }
   SOFIE_HALF_HD Float16 &operator+=(float value) { return *this = Float16(float(*this) + value); }
   SOFIE_HALF_HD Float16 &operator-=(float value) { return *this = Float16(float(*this) - value); }
   SOFIE_HALF_HD Float16 &operator*=(float value) { return *this = Float16(float(*this) * value); }
   SOFIE_HALF_HD Float16 &operator/=(float value) { return *this = Float16(float(*this) / value); }
};
static_assert(sizeof(Float16) == 2 && alignof(Float16) == 2 && std::is_trivially_copyable_v<Float16>);
template<class Value> using AccumulationType = std::conditional_t<std::is_same_v<Value, Float16>, float, Value>;
}

namespace std {
template<> class numeric_limits<SOFIE::Float16> {
public:
   static constexpr bool is_specialized = true;
   static constexpr bool is_signed = true;
   static constexpr bool is_integer = false;
   static constexpr bool is_exact = false;
   static constexpr bool has_infinity = true;
   static constexpr bool has_quiet_NaN = true;
   static constexpr int digits = 11;
   static constexpr int digits10 = 3;
   static constexpr int max_digits10 = 5;
   static constexpr int radix = 2;
   SOFIE_HALF_HD static constexpr SOFIE::Float16 min() noexcept { return SOFIE::Float16::FromBits(0x0400); }
   SOFIE_HALF_HD static constexpr SOFIE::Float16 max() noexcept { return SOFIE::Float16::FromBits(0x7bff); }
   SOFIE_HALF_HD static constexpr SOFIE::Float16 lowest() noexcept { return SOFIE::Float16::FromBits(0xfbff); }
   SOFIE_HALF_HD static constexpr SOFIE::Float16 epsilon() noexcept { return SOFIE::Float16::FromBits(0x1400); }
   SOFIE_HALF_HD static constexpr SOFIE::Float16 infinity() noexcept { return SOFIE::Float16::FromBits(0x7c00); }
   SOFIE_HALF_HD static constexpr SOFIE::Float16 quiet_NaN() noexcept { return SOFIE::Float16::FromBits(0x7e00); }
   SOFIE_HALF_HD static constexpr SOFIE::Float16 denorm_min() noexcept { return SOFIE::Float16::FromBits(1); }
};
}
#undef SOFIE_HALF_HD
#endif
