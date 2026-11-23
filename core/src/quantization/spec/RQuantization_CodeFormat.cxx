#include "SOFIE/quantization/spec/RQuantization_CodeFormat.hxx"

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/quantization/spec/RQuantization_Codec.hxx"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <set>
#include <stdexcept>

namespace SOFIE {

namespace {

[[noreturn]] void ThrowNoCodec(const QuantSpec &spec)
{
   throw std::runtime_error("SOFIE quantization has no codec for " + ToString(spec));
}

template <typename T>
std::shared_ptr<void> Pack(const std::vector<double> &codes)
{
   std::shared_ptr<void> data(std::malloc(codes.size() * sizeof(T)), std::free);
   auto *out = static_cast<T *>(data.get());
   for (std::size_t i = 0; i < codes.size(); ++i)
      out[i] = static_cast<T>(codes[i]);
   return data;
}

template <typename T>
std::vector<double> Unpack(const void *data, std::size_t count)
{
   const auto *in = static_cast<const T *>(data);
   return std::vector<double>(in, in + count);
}

} // namespace

bool IsBakeableCodeType(ETensorType type)
{
   switch (type) {
   case ETensorType::INT8:
   case ETensorType::UINT8:
   case ETensorType::FLOAT8E4M3FN: return true;
   default: return false;
   }
}

bool HasKernelCodes(const QuantSpec &spec)
{
   if (!spec.IsDefined())
      return false;
   switch (spec.encoding) {
   case EQuantEncoding::Integer:
   case EQuantEncoding::FP8E4M3: return IsBakeableCodeType(spec.CodeType());
   case EQuantEncoding::FP8E5M2:
   case EQuantEncoding::Undefined: break;
   }
   return false;
}

std::string QuantEncodeExpression(const QuantSpec &spec, const std::string &value, const std::string &scale,
                                  const std::string &zeroPoint)
{
   switch (spec.encoding) {
   case EQuantEncoding::Integer:
      return "SOFIE::QuantCodec::EncodeInteger(" + value + ", " + scale + ", " + zeroPoint + ", " +
             ExactDoubleLiteral(spec.codeMin) + ", " + ExactDoubleLiteral(spec.codeMax) + ", " +
             std::to_string(static_cast<int>(spec.rounding)) + ")";
   case EQuantEncoding::FP8E4M3: return "SOFIE::QuantCodec::EncodeE4M3Scaled(" + value + ", " + scale + ")";
   case EQuantEncoding::FP8E5M2:
   case EQuantEncoding::Undefined: break;
   }
   ThrowNoCodec(spec);
}

std::string QuantUnscaledCodeExpression(const QuantSpec &spec, const std::string &code)
{
   switch (spec.encoding) {
   case EQuantEncoding::Integer: return "static_cast<double>(" + code + ")";
   case EQuantEncoding::FP8E4M3: return "SOFIE::QuantCodec::DecodeE4M3(static_cast<std::uint8_t>(" + code + "))";
   case EQuantEncoding::FP8E5M2:
   case EQuantEncoding::Undefined: break;
   }
   ThrowNoCodec(spec);
}

std::string QuantDecodeExpression(const QuantSpec &spec, const std::string &code, const std::string &scale,
                                  const std::string &zeroPoint)
{
   switch (spec.encoding) {
   case EQuantEncoding::Integer:
      return "SOFIE::QuantCodec::DecodeInteger(" + QuantUnscaledCodeExpression(spec, code) + ", " + scale + ", " +
             zeroPoint + ")";
   case EQuantEncoding::FP8E4M3: return "(" + QuantUnscaledCodeExpression(spec, code) + " * " + scale + ")";
   case EQuantEncoding::FP8E5M2:
   case EQuantEncoding::Undefined: break;
   }
   ThrowNoCodec(spec);
}

std::string QuantEncodeExpression(const QuantSpec &spec, const std::string &value)
{
   return QuantEncodeExpression(spec, value, ExactDoubleLiteral(spec.Scale()),
                                std::to_string(spec.ZeroPoint()) + ".0");
}

std::string QuantDecodeExpression(const QuantSpec &spec, const std::string &code)
{
   return QuantDecodeExpression(spec, code, ExactDoubleLiteral(spec.Scale()), std::to_string(spec.ZeroPoint()) + ".0");
}

double QuantEncodeValue(const QuantSpec &spec, double value, std::size_t channel)
{
   switch (spec.encoding) {
   case EQuantEncoding::Integer:
      return QuantCodec::EncodeInteger(value, spec.Scale(channel), static_cast<double>(spec.ZeroPoint(channel)),
                                       spec.codeMin, spec.codeMax, static_cast<int>(spec.rounding));
   case EQuantEncoding::FP8E4M3: return static_cast<double>(QuantCodec::EncodeE4M3Scaled(value, spec.Scale(channel)));
   case EQuantEncoding::FP8E5M2:
   case EQuantEncoding::Undefined: break;
   }
   ThrowNoCodec(spec);
}

double QuantDecodeCode(const QuantSpec &spec, double code, std::size_t channel)
{
   switch (spec.encoding) {
   case EQuantEncoding::Integer:
      return QuantCodec::DecodeInteger(code, spec.Scale(channel), static_cast<double>(spec.ZeroPoint(channel)));
   case EQuantEncoding::FP8E4M3:
      return QuantCodec::DecodeE4M3(static_cast<std::uint8_t>(code)) * spec.Scale(channel);
   case EQuantEncoding::FP8E5M2:
   case EQuantEncoding::Undefined: break;
   }
   ThrowNoCodec(spec);
}

std::shared_ptr<void> PackCodes(ETensorType type, const std::vector<double> &codes)
{
   switch (type) {
   case ETensorType::INT8: return Pack<std::int8_t>(codes);
   case ETensorType::UINT8:
   case ETensorType::FLOAT8E4M3FN: return Pack<std::uint8_t>(codes);
   default: break;
   }
   throw std::runtime_error("SOFIE quantization cannot bake codes of type " + ConvertTypeToString(type));
}

std::vector<double> UnpackCodes(ETensorType type, const void *data, std::size_t count)
{
   switch (type) {
   case ETensorType::INT8: return Unpack<std::int8_t>(data, count);
   case ETensorType::UINT8:
   case ETensorType::FLOAT8E4M3FN: return Unpack<std::uint8_t>(data, count);
   default: break;
   }
   throw std::runtime_error("SOFIE quantization cannot read codes of type " + ConvertTypeToString(type));
}

bool IsFullRangeIntegerGrid(const QuantSpec &spec, ETensorType type)
{
   if (spec.encoding != EQuantEncoding::Integer || !spec.IsPerTensor() || spec.ZeroPoint() != 0 ||
       spec.rounding != EQuantRounding::NearestEven || spec.CodeType() != type ||
       spec.bitWidth != 8 * GetTypeSize(type))
      return false;
   const auto full = MakeIntegerQuantSpec(spec.bitWidth, spec.isSigned, false, {1.0}, {0}, -1, spec.rounding);
   return spec.codeMin == full.codeMin && spec.codeMax == full.codeMax;
}

std::string QuantCodecSupportCode(std::string_view generatedCode)
{
   struct Block {
      std::string name;
      std::string text;
   };
   // The macro block and the function blocks of the header, in header order; parsed once.
   static const auto blocks = [] {
      std::vector<Block> parsed;
      const std::string header(QuantCodecHeaderText());
      const std::regex guarded(R"(#ifndef SOFIE_QUANT_CODEC_(\w+)\n[\s\S]*?#endif // SOFIE_QUANT_CODEC_\1\n)");
      for (auto it = std::sregex_iterator(header.begin(), header.end(), guarded); it != std::sregex_iterator(); ++it)
         parsed.push_back({(*it)[1].str(), it->str()});
      if (parsed.empty() || parsed.front().name != "FN")
         throw std::runtime_error("SOFIE quantization codec header has no SOFIE_QUANT_CODEC_FN block");
      return parsed;
   }();

   const auto callsOf = [](const std::string &text, const std::string &self) {
      std::set<std::string> calls;
      for (const auto &block : blocks)
         if (block.name != "FN" && block.name != self &&
             std::regex_search(text, std::regex("\\b" + block.name + "\\(")))
            calls.insert(block.name);
      return calls;
   };

   // Functions the generated code calls, closed under the calls inside the codec itself.
   std::set<std::string> needed;
   const std::string code(generatedCode);
   for (const auto &block : blocks)
      if (block.name != "FN" && code.find("QuantCodec::" + block.name + "(") != std::string::npos)
         needed.insert(block.name);
   if (needed.empty())
      return {};
   for (bool grew = true; grew;) {
      grew = false;
      for (const auto &block : blocks)
         if (needed.count(block.name) != 0)
            for (const auto &callee : callsOf(block.text, block.name))
               grew |= needed.insert(callee).second;
   }

   std::string out = "\n// SOFIE quantization codec (from SOFIE/quantization/spec/RQuantization_Codec.hxx)\n" +
                     blocks.front().text + "\nnamespace SOFIE {\nnamespace QuantCodec {\n";
   for (const auto &block : blocks)
      if (needed.count(block.name) != 0)
         out += "\n" + block.text;
   return out + "\n} // namespace QuantCodec\n} // namespace SOFIE\n";
}

} // namespace SOFIE
