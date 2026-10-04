
#include <SOFIE/RModel.hxx>
#include <SOFIE/RModelParser_ONNX.hxx>

#include <filesystem>

#include <gtest/gtest.h>

#include <cstdint>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SOFIE;

namespace {

void AppendVarint(std::string &out, std::uint64_t v)
{
   while (v >= 0x80) {
      out.push_back(char((v & 0x7f) | 0x80));
      v >>= 7;
   }
   out.push_back(char(v));
}

void AppendVarintField(std::string &out, int field, std::uint64_t v)
{
   AppendVarint(out, std::uint64_t(field) << 3 | 0);
   AppendVarint(out, v);
}

void AppendBytesField(std::string &out, int field, const std::string &payload)
{
   AppendVarint(out, std::uint64_t(field) << 3 | 2);
   AppendVarint(out, payload.size());
   out += payload;
}

std::string StringEntry(const std::string &key, const std::string &value)
{
   std::string out;
   AppendBytesField(out, 1, key);
   AppendBytesField(out, 2, value);
   return out;
}

std::string
ExternalFloatTensor(const std::string &name, std::uint64_t dim, std::uint64_t offset, const std::string &location = "")
{
   std::string out;
   AppendVarintField(out, 1, dim);
   AppendVarintField(out, 2, 1);
   AppendBytesField(out, 8, name);
   if (!location.empty())
      AppendBytesField(out, 13, StringEntry("location", location));
   AppendBytesField(out, 13, StringEntry("offset", std::to_string(offset)));
   AppendBytesField(out, 13, StringEntry("length", std::to_string(dim * sizeof(float))));
   AppendVarintField(out, 14, 1);
   return out;
}

void WriteModelFile(const std::string &fileName, const std::vector<std::string> &initializers,
                    const std::string &firstTensorName)
{
   std::string node;
   AppendBytesField(node, 1, firstTensorName);
   AppendBytesField(node, 2, "out");
   AppendBytesField(node, 4, "Identity");

   std::string output;
   AppendBytesField(output, 1, "out");

   std::string graph;
   AppendBytesField(graph, 1, node);
   AppendBytesField(graph, 2, "test_graph");
   for (const std::string &tensor : initializers)
      AppendBytesField(graph, 5, tensor);
   AppendBytesField(graph, 12, output);

   std::string model;
   AppendVarintField(model, 1, 8);
   AppendBytesField(model, 7, graph);

   std::ofstream file(fileName, std::ios::binary);
   file.write(model.data(), model.size());
   ASSERT_TRUE(file.good());
}

void WriteDataFile(const std::string &fileName, const std::vector<float> &values, std::size_t padding = 0)
{
   std::ofstream file(fileName, std::ios::binary);
   for (std::size_t i = 0; i < padding; ++i)
      file.put('\0');
   for (float value : values) {
      std::uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      for (int i = 0; i < 4; ++i)
         file.put(char((bits >> (8 * i)) & 0xff));
   }
   ASSERT_TRUE(file.good());
}

struct TestTensor {
   std::string name;
   std::vector<std::string> dims;
};

std::string FloatValueInfo(const TestTensor &tensor)
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

std::string
GenerateGemmCode(const std::string &fileName, const std::vector<TestTensor> &inputs, const TestTensor &output)
{
   std::string node;
   for (const TestTensor &input : inputs)
      AppendBytesField(node, 1, input.name);
   AppendBytesField(node, 2, output.name);
   AppendBytesField(node, 3, "gemm_0");
   AppendBytesField(node, 4, "Gemm");

   std::string graph;
   AppendBytesField(graph, 1, node);
   AppendBytesField(graph, 2, "test_graph");
   for (const TestTensor &input : inputs)
      AppendBytesField(graph, 11, FloatValueInfo(input));
   AppendBytesField(graph, 12, FloatValueInfo(output));

   std::string opset;
   AppendVarintField(opset, 2, 13);
   std::string model;
   AppendVarintField(model, 1, 10);
   AppendBytesField(model, 7, graph);
   AppendBytesField(model, 8, opset);

   std::ofstream file(fileName, std::ios::binary);
   file.write(model.data(), model.size());
   file.close();

   RModel rmodel = RModelParser_ONNX{}.Parse(fileName);
   rmodel.Generate(Options::kNoWeightFile);
   std::ostringstream code;
   rmodel.PrintGenerated(code);
   return code.str();
}

}

TEST(SOFIEParser, ExternalDataLocationRelativeToModelDirectory)
{
   std::filesystem::create_directories("extdata_models");
   const std::vector<float> values1{1.f, 2.f, 3.f, 4.f};
   const std::vector<float> values2{-5.f, 6.5f};
   WriteDataFile("extdata_models/weights1.bin", values1,  8);
   WriteDataFile("extdata_models/weights2.bin", values2);
   WriteModelFile("extdata_models/modelLoc.onnx",
                  {ExternalFloatTensor("w1", values1.size(),  8, "weights1.bin"),
                   ExternalFloatTensor("w2", values2.size(),  0, "weights2.bin")},
                  "w1");

   RModelParser_ONNX parser;
   RModel model = parser.Parse("extdata_models/modelLoc.onnx");
   EXPECT_EQ(model.GetTensorData<float>("w1"), values1);
   EXPECT_EQ(model.GetTensorData<float>("w2"), values2);
}

TEST(SOFIEParser, ExternalDataFileResolvedPerParsedModel)
{
   const std::vector<float> valuesA{10.f, 20.f, 30.f};
   const std::vector<float> valuesB{-1.f, -2.f, -3.f};
   WriteDataFile("extdataA.onnx.data", valuesA);
   WriteDataFile("extdataB.onnx.data", valuesB);
   WriteModelFile("extdataA.onnx", {ExternalFloatTensor("w", valuesA.size(),  0)}, "w");
   WriteModelFile("extdataB.onnx", {ExternalFloatTensor("w", valuesB.size(),  0)}, "w");

   RModelParser_ONNX parser;
   RModel modelA = parser.Parse("extdataA.onnx");
   EXPECT_EQ(modelA.GetTensorData<float>("w"), valuesA);
   RModel modelB = parser.Parse("extdataB.onnx");
   EXPECT_EQ(modelB.GetTensorData<float>("w"), valuesB);
}

TEST(SOFIEParser, SetExternalDataFileTakesPrecedenceOnce)
{
   const std::vector<float> valuesExplicit{5.f, 6.f};
   const std::vector<float> valuesLocation{7.f, 8.f};
   WriteDataFile("extdata_explicit.bin", valuesExplicit);
   WriteDataFile("extdata_location.bin", valuesLocation);
   WriteModelFile("extdataC.onnx",
                  {ExternalFloatTensor("w", valuesExplicit.size(),  0, "extdata_location.bin")}, "w");

   RModelParser_ONNX parser;
   parser.SetExternalDataFile("extdata_explicit.bin");
   RModel modelExplicit = parser.Parse("extdataC.onnx");
   EXPECT_EQ(modelExplicit.GetTensorData<float>("w"), valuesExplicit);

   RModel modelLocation = parser.Parse("extdataC.onnx");
   EXPECT_EQ(modelLocation.GetTensorData<float>("w"), valuesLocation);
}

TEST(SOFIEParser, MissingExternalDataFileThrows)
{
   WriteModelFile("extdataD.onnx", {ExternalFloatTensor("w", 2,  0, "extdata_does_not_exist.bin")}, "w");

   RModelParser_ONNX parser;
   EXPECT_THROW(parser.Parse("extdataD.onnx"), std::runtime_error);
}

TEST(SOFIEParser, GemmBroadcastsABiasThatDoesNotHaveTheOutputShape)
{
   const std::string code = GenerateGemmCode(
      "gemm_static_bias.onnx", {{"x", {"1", "10"}}, {"w", {"10", "32"}}, {"b", {"32"}}}, {"y", {"1", "32"}});

   EXPECT_NE(code.find("Copy(tensor_y"), std::string::npos);
   EXPECT_NE(code.find(",nullptr);"), std::string::npos);
}

TEST(SOFIEParser, GemmDoesNotBroadcastABiasThatHasTheOutputShape)
{
   const std::string code = GenerateGemmCode(
      "gemm_dynamic_bias.onnx", {{"a", {"N", "3"}}, {"w", {"3", "4"}}, {"c", {"N", "4"}}}, {"y", {"N", "4"}});

   EXPECT_EQ(code.find("Copy(tensor_y"), std::string::npos);
   EXPECT_NE(code.find(",tensor_c);"), std::string::npos);
}
