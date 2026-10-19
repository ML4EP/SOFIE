#pragma once

// Minimal protobuf wire-format writers, shared by the tests building ONNX models by hand
// (cpu/TestSofieParser.cxx and cpu/TestSafetensorsWeights.cxx).

#include <cstdint>
#include <string>

inline void AppendVarint(std::string &out, std::uint64_t v)
{
   while (v >= 0x80) {
      out.push_back(char((v & 0x7f) | 0x80));
      v >>= 7;
   }
   out.push_back(char(v));
}

inline void AppendVarintField(std::string &out, int field, std::uint64_t v)
{
   AppendVarint(out, std::uint64_t(field) << 3 | 0);
   AppendVarint(out, v);
}

inline void AppendBytesField(std::string &out, int field, const std::string &payload)
{
   AppendVarint(out, std::uint64_t(field) << 3 | 2);
   AppendVarint(out, payload.size());
   out += payload;
}
