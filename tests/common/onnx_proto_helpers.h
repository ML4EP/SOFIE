#pragma once

// Minimal protobuf wire-format writers, shared by the tests building ONNX models by hand
// (cpu/TestSofieParser.cxx, cpu/TestSafetensorsWeights.cxx and models/parser/).

#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

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

// ONNX ValueInfoProto of a float tensor; a dim that doesn't start with a digit is a dim_param
struct TestTensor {
   std::string name;
   std::vector<std::string> dims;
};

inline std::string FloatValueInfo(const TestTensor &tensor)
{
   std::string shape;
   for (const std::string &dim : tensor.dims) {
      std::string entry;
      if (std::isdigit(static_cast<unsigned char>(dim[0])))
         AppendVarintField(entry, 1, std::stoull(dim));
      else
         AppendBytesField(entry, 2, dim);
      AppendBytesField(shape, 1, entry);
   }

   std::string tensorType;
   AppendVarintField(tensorType, 1, 1);
   AppendBytesField(tensorType, 2, shape);
   std::string type;
   AppendBytesField(type, 1, tensorType);
   std::string out;
   AppendBytesField(out, 1, tensor.name);
   AppendBytesField(out, 2, type);
   return out;
}

// ONNX TensorProto of a 1D float tensor, with the values in raw_data (little endian)
inline std::string FloatTensor(const std::string &name, const std::vector<float> &values)
{
   std::string raw;
   for (float value : values) {
      std::uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      for (int i = 0; i < 4; ++i)
         raw.push_back(char((bits >> (8 * i)) & 0xff));
   }
   std::string out;
   AppendVarintField(out, 1, values.size());
   AppendVarintField(out, 2, 1);
   AppendBytesField(out, 8, name);
   AppendBytesField(out, 9, raw);
   return out;
}

// ONNX AttributeProto of type INT
inline std::string IntAttribute(const std::string &name, std::int64_t value)
{
   std::string out;
   AppendBytesField(out, 1, name);
   AppendVarintField(out, 3, static_cast<std::uint64_t>(value));
   AppendVarintField(out, 20, 2);
   return out;
}
