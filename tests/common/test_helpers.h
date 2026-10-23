#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <type_traits>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-3f;

class SofieReference {
public:
   const std::vector<float> &f32(std::string const &key) const { return at(fF32, key); }
   const std::vector<double> &f64(std::string const &key) const { return at(fF64, key); }
   const std::vector<int64_t> &i64(std::string const &key) const { return at(fI64, key); }
   const std::vector<uint8_t> &u8(std::string const &key) const { return at(fU8, key); }

   std::map<std::string, std::vector<float>> fF32;
   std::map<std::string, std::vector<double>> fF64;
   std::map<std::string, std::vector<int64_t>> fI64;
   std::map<std::string, std::vector<uint8_t>> fU8;

private:
   template <class Map>
   static const typename Map::mapped_type &at(Map const &m, std::string const &key)
   {
      auto it = m.find(key);
      if (it == m.end())
         throw std::runtime_error("no reference data entry \"" + key + "\" of this type");
      return it->second;
   }
};

inline SofieReference readReference(std::string const &modelName)
{
   const std::string path = "models/onnx/references/" + modelName + ".ref";
   std::ifstream in(path);
   if (!in)
      throw std::runtime_error("cannot open reference data file " + path +
                               " (it is written by the SofieGenerateModels_ONNX test)");
   SofieReference ref;
   std::string key, type;
   std::size_t count;
   while (in >> key >> type >> count) {
      bool ok = true;
      if (type == "f32") {
         auto &v = ref.fF32[key];
         v.resize(count);
         for (auto &x : v)
            ok &= bool(in >> x);
      } else if (type == "f64") {
         auto &v = ref.fF64[key];
         v.resize(count);
         for (auto &x : v)
            ok &= bool(in >> x);
      } else if (type == "i64") {
         auto &v = ref.fI64[key];
         v.resize(count);
         for (auto &x : v)
            ok &= bool(in >> x);
      } else if (type == "u8") {
         auto &v = ref.fU8[key];
         v.resize(count);
         for (auto &x : v) {
            int tmp;
            ok &= bool(in >> tmp);
            x = static_cast<uint8_t>(tmp);
         }
      } else {
         ok = false;
      }
      if (!ok)
         throw std::runtime_error("malformed reference data file " + path + " at entry \"" + key + "\"");
   }
   return ref;
}

template <typename T, typename U>
void expectNear(std::vector<T> const &output, std::vector<U> const &expected, float tolerance)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i)
      EXPECT_LE(std::abs(static_cast<double>(output[i]) - static_cast<double>(expected[i])), tolerance)
         << "at output index " << i;
}

template <typename T, typename U>
void expectNear(std::vector<std::vector<T>> const &output, std::vector<std::vector<U>> const &expected, float tolerance)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i) {
      SCOPED_TRACE("output tensor " + std::to_string(i));
      expectNear(output[i], expected[i], tolerance);
   }
}

inline void
expectNearCapped(const float *actual, const float *expected, std::size_t n, double tolerance, std::size_t maxPrint = 10)
{
   std::size_t mismatchCount = 0;
   for (std::size_t i = 0; i < n; ++i) {
      const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
      if (diff > tolerance) {
         if (mismatchCount < maxPrint) {
            ADD_FAILURE() << "Mismatch at index " << i << " actual=" << actual[i] << " expected=" << expected[i]
                          << " diff=" << diff;
         }
         ++mismatchCount;
      }
   }
   if (mismatchCount > maxPrint) {
      ADD_FAILURE() << "Further mismatches suppressed (total mismatches: " << mismatchCount << ")";
   }
}

template <typename T, typename U>
void expectEqual(std::vector<T> const &output, std::vector<U> const &expected)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i)
      EXPECT_EQ(static_cast<int64_t>(output[i]), static_cast<int64_t>(expected[i])) << "at output index " << i;
}

template <typename T, typename U>
void expectEqual(std::vector<std::vector<T>> const &output, std::vector<std::vector<U>> const &expected)
{
   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < output.size(); ++i) {
      SCOPED_TRACE("output tensor " + std::to_string(i));
      expectEqual(output[i], expected[i]);
   }
}

using TupleFloatInt64_t = std::tuple<std::vector<float>, std::vector<int64_t>>;

template <class T>
auto sofieArg(T const &value)
{
   if constexpr (std::is_arithmetic_v<T>)
      return value;
   else
      return value.data();
}

template <class Session, class... Ts>
auto sofieInfer(Session &session, Ts const &...inputs)
{
   return session.infer(sofieArg(inputs)...);
}

#define ASSERT_RUN_0(OutputType, Model)                                                \
   OutputType output = [&]() -> OutputType {                                           \
      SOFIE_##Model::Session sofieSession(std::string(#Model) + modelDataSuffix);      \
      return sofieSession.infer();                                                     \
   }()

#define ASSERT_RUN(OutputType, Model, ...)                                             \
   OutputType output = [&]() -> OutputType {                                           \
      SOFIE_##Model::Session sofieSession(std::string(#Model) + modelDataSuffix);      \
      return sofieInfer(sofieSession, __VA_ARGS__);                                    \
   }()

#define ASSERT_RUN_NO_SESSION(OutputType, Model, ...)                                  \
   OutputType output = [&](auto const &...sofieInputs) -> OutputType {                 \
      return SOFIE_##Model::infer(sofieArg(sofieInputs)...);                           \
   }(__VA_ARGS__)

#define ASSERT_RUN_SESSION_ARGS(OutputType, Model, sessionArgs, ...)                   \
   OutputType output = [&]() -> OutputType {                                           \
      SOFIE_##Model::Session sofieSession sessionArgs;                                 \
      return sofieInfer(sofieSession, __VA_ARGS__);                                    \
   }()

inline std::string hexFloat(double value)
{
   char buf[64];
   std::snprintf(buf, sizeof(buf), "%a", value);
   return buf;
}

inline std::string floatArrayLiteral(std::vector<float> const &values)
{
   std::string out = "{";
   for (std::size_t i = 0; i < values.size(); ++i) {
      if (i)
         out += ",";
      float v = values[i];
      if (std::isinf(v))
         out += v > 0 ? "std::numeric_limits<float>::infinity()" : "-std::numeric_limits<float>::infinity()";
      else if (std::isnan(v))
         out += "std::numeric_limits<float>::quiet_NaN()";
      else
         out += hexFloat(v) + "f";
   }
   return out + "}";
}

#ifdef SOFIE_TEST_CXX
inline int sofieExec(std::string const &command)
{
   return std::system(command.c_str());
}

inline bool compileGeneratedProgram(std::string const &name, std::string const &source)
{
   {
      std::ofstream out(name + ".cxx");
      out << source;
   }
   const std::string command = std::string(SOFIE_TEST_CXX) + " " + SOFIE_TEST_CXX_FLAGS + " -I. " + name + ".cxx -o " +
                               name + ".exe " + SOFIE_TEST_LINK_FLAGS + " > " + name + ".log 2>&1";
   if (sofieExec(command) != 0) {
      std::ifstream log(name + ".log");
      std::cerr << "[compileGeneratedProgram] compilation of " << name << ".cxx failed:\n"
                << log.rdbuf() << std::endl;
      return false;
   }
   return true;
}

inline std::string runGeneratedProgram(std::string const &name)
{
   const std::string outFile = name + ".out";
   if (sofieExec("./" + name + ".exe > " + outFile) != 0)
      throw std::runtime_error("running " + name + ".exe failed");
   std::ifstream in(outFile);
   return std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

inline std::vector<float> parseFloatList(std::string const &text)
{
   std::vector<float> values;
   std::istringstream in(text);
   std::string token;
   while (in >> token)
      values.push_back(std::strtof(token.c_str(), nullptr));
   return values;
}
#endif
