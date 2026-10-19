
#include <SOFIE/RModel.hxx>
#include <SOFIE/RModelParser_ONNX.hxx>

#include <gtest/gtest.h>

#include "common/onnx_proto_helpers.h"
#include "common/test_helpers.h"

#include <nlohmann/json.hpp>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace SOFIE;

namespace {

std::string
InlineFloatTensor(const std::string &name, const std::vector<std::uint64_t> &shape, const std::vector<float> &values)
{
   std::string out;
   for (std::uint64_t d : shape)
      AppendVarintField(out, 1, d);
   AppendVarintField(out, 2, 1);
   AppendBytesField(out, 8, name);
   std::string rawData;
   rawData.reserve(values.size() * sizeof(float));
   for (float value : values) {
      std::uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      for (int i = 0; i < 4; ++i)
         rawData.push_back(char((bits >> (8 * i)) & 0xff));
   }
   AppendBytesField(out, 9, rawData);
   return out;
}

std::string FloatValueInfo(const std::string &name, const std::vector<std::uint64_t> &shape)
{
   std::string shapeProto;
   for (std::uint64_t d : shape) {
      std::string dim;
      AppendVarintField(dim, 1, d);
      AppendBytesField(shapeProto, 1, dim);
   }
   std::string tensorType;
   AppendVarintField(tensorType, 1, 1);
   AppendBytesField(tensorType, 2, shapeProto);
   std::string type;
   AppendBytesField(type, 1, tensorType);
   std::string out;
   AppendBytesField(out, 1, name);
   AppendBytesField(out, 2, type);
   return out;
}

void WriteMulModelFile(const std::string &fileName, const std::vector<std::uint64_t> &shape,
                       const std::vector<float> &weights)
{
   std::string node;
   AppendBytesField(node, 1, "X");
   AppendBytesField(node, 1, "W");
   AppendBytesField(node, 2, "Y");
   AppendBytesField(node, 4, "Mul");

   std::string graph;
   AppendBytesField(graph, 1, node);
   AppendBytesField(graph, 2, "test_graph");
   AppendBytesField(graph, 5, InlineFloatTensor("W", shape, weights));
   AppendBytesField(graph, 11, FloatValueInfo("X", shape));
   AppendBytesField(graph, 12, FloatValueInfo("Y", shape));

   std::string model;
   AppendVarintField(model, 1, 8);
   AppendBytesField(model, 7, graph);

   std::ofstream file(fileName, std::ios::binary);
   file.write(model.data(), model.size());
   ASSERT_TRUE(file.good());
}

void WriteTwoInputMulModelFile(const std::string &fileName, const std::vector<std::uint64_t> &shape)
{
   std::string node;
   AppendBytesField(node, 1, "X");
   AppendBytesField(node, 1, "Y");
   AppendBytesField(node, 2, "Z");
   AppendBytesField(node, 4, "Mul");

   std::string graph;
   AppendBytesField(graph, 1, node);
   AppendBytesField(graph, 2, "test_graph");
   AppendBytesField(graph, 11, FloatValueInfo("X", shape));
   AppendBytesField(graph, 11, FloatValueInfo("Y", shape));
   AppendBytesField(graph, 12, FloatValueInfo("Z", shape));

   std::string model;
   AppendVarintField(model, 1, 8);
   AppendBytesField(model, 7, graph);

   std::ofstream file(fileName, std::ios::binary);
   file.write(model.data(), model.size());
   ASSERT_TRUE(file.good());
}

std::string GenerateModel(const std::string &modelFileName)
{
   RModelParser_ONNX parser;
   RModel model = parser.Parse(modelFileName);
   model.Generate(Options::kSafetensorsWeightFile);
   model.OutputGenerated();
   return model.GetName();
}

void DeclareModel(const std::string &modelName)
{
   std::ifstream header(modelName + ".hxx");
   ASSERT_TRUE(header.good()) << "failed to find header " << modelName << ".hxx";
}

std::string RunDriver(const std::string &modelName, const std::string &mainBody)
{
   static int counter = 0;
   const std::string name = "sofie_safetensors_driver_" + std::to_string(counter++);
   const std::string source = "#include <cstdio>\n#include <cstdlib>\n#include <fstream>\n#include <iterator>\n"
                              "#include <string>\n#include <vector>\n#include \"" +
                              modelName + ".hxx\"\nint main()\n{\n" + mainBody + "\n}\n";
   if (!compileGeneratedProgram(name, source))
      throw std::runtime_error("failed to compile the driver program for " + modelName);
   return runGeneratedProgram(name);
}

std::string PrintOutput()
{
   return "   for (float v : output) std::printf(\"%a\\n\", v);\n";
}

std::string StringLiteral(const std::string &s)
{
   std::string out = "\"";
   for (char c : s) {
      if (c == '\\' || c == '"')
         out.push_back('\\');
      out.push_back(c);
   }
   return out + "\"";
}

std::vector<float>
RunModel(const std::string &modelName, const std::string &dataFileName, const std::vector<float> &input)
{
   std::string body = "   std::vector<float> input = " + floatArrayLiteral(input) + ";\n";
   body += "   SOFIE_" + modelName + "::Session session(" + StringLiteral(dataFileName) + ");\n";
   body += "   std::vector<float> output = session.infer(input.data());\n" + PrintOutput();
   return parseFloatList(RunDriver(modelName, body));
}

std::vector<float> RunModelDefaultFile(const std::string &modelName, const std::vector<float> &input)
{
   std::string body = "   std::vector<float> input = " + floatArrayLiteral(input) + ";\n";
   body += "   SOFIE_" + modelName + "::Session session;\n";
   body += "   std::vector<float> output = session.infer(input.data());\n" + PrintOutput();
   return parseFloatList(RunDriver(modelName, body));
}

bool SessionConstructorThrows(const std::string &modelName, const std::string &dataFileName)
{
   std::string body = "   try { SOFIE_" + modelName + "::Session s(" + StringLiteral(dataFileName) +
                      "); } catch (const std::exception &) { std::printf(\"1\"); return 0; }\n"
                      "   std::printf(\"0\");\n";
   return RunDriver(modelName, body) == "1";
}

std::string ReadWholeFile(const std::string &fileName)
{
   std::ifstream f(fileName, std::ios::binary);
   if (!f)
      throw std::runtime_error("cannot open " + fileName);
   return std::string{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::string BlobFile(const std::string &blob)
{
   static int counter = 0;
   const std::string fileName = "sofie_safetensors_blob_" + std::to_string(counter++) + ".bin";
   std::ofstream out(fileName, std::ios::binary);
   out.write(blob.data(), blob.size());
   return fileName;
}

std::string BlobPreamble(const std::string &modelName, const std::string &blobFile)
{
   return "   std::ifstream blobStream(" + StringLiteral(blobFile) + ", std::ios::binary);\n"
          "   const std::string blob{std::istreambuf_iterator<char>(blobStream), std::istreambuf_iterator<char>()};\n"
          "   SOFIE_" + modelName + "::SafetensorsBlob safetensorsBlob{blob.data(), blob.size()};\n";
}

std::vector<float>
RunModelFromBlob(const std::string &modelName, const std::string &blob, const std::vector<float> &input)
{
   std::string body = "   std::vector<float> input = " + floatArrayLiteral(input) + ";\n";
   body += BlobPreamble(modelName, BlobFile(blob));
   body += "   SOFIE_" + modelName + "::Session session(safetensorsBlob);\n";
   body += "   std::vector<float> output = session.infer(input.data());\n" + PrintOutput();
   return parseFloatList(RunDriver(modelName, body));
}

std::vector<float> RunTwoInputModelFromBlob(const std::string &modelName, const std::string &blob,
                                            std::vector<float> &inputX, std::vector<float> &inputY)
{
   std::string body = "   std::vector<float> inputX = " + floatArrayLiteral(inputX) + ";\n";
   body += "   std::vector<float> inputY = " + floatArrayLiteral(inputY) + ";\n";
   body += BlobPreamble(modelName, BlobFile(blob));
   body += "   SOFIE_" + modelName + "::Session session(safetensorsBlob);\n";
   body += "   std::vector<float> output = session.infer(inputX.data(), inputY.data());\n" + PrintOutput();
   return parseFloatList(RunDriver(modelName, body));
}

std::string MakeSafetensorsPayload(const std::string &header, const std::string &payload)
{
   std::string out;
   for (int i = 0; i < 8; ++i)
      out.push_back(char((header.size() >> (8 * i)) & 0xff));
   out += header;
   out += payload;
   return out;
}

bool BlobSessionConstructorThrows(const std::string &modelName, const std::string &blob)
{
   std::string body = BlobPreamble(modelName, BlobFile(blob));
   body += "   try { SOFIE_" + modelName +
           "::Session s(safetensorsBlob); } catch (const std::exception &) { std::printf(\"1\"); return 0; }\n"
           "   std::printf(\"0\");\n";
   return RunDriver(modelName, body) == "1";
}

const std::vector<float> kWeights{1.5f, -2.25f, 1.0e-42f, std::numeric_limits<float>::infinity(), 0.0f, 123456.75f};
const std::vector<float> kInput{0.5f, -1.0f, 3.0f, 1.0f, -8.0f, 0.25f};

std::vector<float> ExpectedOutput()
{
   std::vector<float> out(kWeights.size());
   for (std::size_t i = 0; i < out.size(); ++i)
      out[i] = kWeights[i] * kInput[i];
   return out;
}

}

#define SOFIE_SKIP_ON_BIG_ENDIAN \
   if (std::endian::native != std::endian::little) \
   GTEST_SKIP() << "safetensors weights are little-endian only"

TEST(SOFIESafetensors, FileLayoutAndPayload)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   WriteMulModelFile("safetensors_mul.onnx", {2, 3}, kWeights);
   const std::string name = GenerateModel("safetensors_mul.onnx");
   ASSERT_EQ(name, "safetensors_mul");

   std::ifstream f("safetensors_mul.safetensors", std::ios::binary);
   ASSERT_TRUE(f.is_open());
   std::uint64_t headerSize = 0;
   char sizestr[8];
   f.read(sizestr, 8);
   ASSERT_TRUE(f.good());
   for (int i = 0; i < 8; ++i)
      headerSize |= std::uint64_t(static_cast<unsigned char>(sizestr[i])) << (8 * i);

   std::string headerStr(headerSize, '\0');
   f.read(headerStr.data(), headerStr.size());
   ASSERT_TRUE(f.good());
   const auto header = nlohmann::json::parse(headerStr);

   ASSERT_EQ(header.size(), 1u);
   ASSERT_TRUE(header.contains("tensor_W"));
   const auto &entry = header["tensor_W"];
   EXPECT_EQ(entry["dtype"], "F32");
   EXPECT_EQ(entry["shape"], (nlohmann::json::array_t{2, 3}));
   EXPECT_EQ(entry["data_offsets"], (nlohmann::json::array_t{0, kWeights.size() * sizeof(float)}));

   std::vector<char> payload(kWeights.size() * sizeof(float));
   f.read(payload.data(), payload.size());
   ASSERT_TRUE(f.good());
   EXPECT_EQ(std::memcmp(payload.data(), kWeights.data(), payload.size()), 0);
   EXPECT_EQ(f.peek(), EOF);
}

TEST(SOFIESafetensors, DeterministicOutput)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   WriteMulModelFile("safetensors_mul.onnx", {2, 3}, kWeights);
   GenerateModel("safetensors_mul.onnx");
   std::ifstream first("safetensors_mul.safetensors", std::ios::binary);
   const std::string content1{std::istreambuf_iterator<char>(first), std::istreambuf_iterator<char>()};
   GenerateModel("safetensors_mul.onnx");
   std::ifstream second("safetensors_mul.safetensors", std::ios::binary);
   const std::string content2{std::istreambuf_iterator<char>(second), std::istreambuf_iterator<char>()};
   EXPECT_EQ(content1, content2);
}

TEST(SOFIESafetensors, InferenceMatchesReference)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   DeclareModel("safetensors_mul");

   const std::vector<float> output = RunModel("safetensors_mul", "safetensors_mul.safetensors", kInput);
   const std::vector<float> expected = ExpectedOutput();

   ASSERT_EQ(output.size(), expected.size());
   for (std::size_t i = 0; i < expected.size(); ++i)
      EXPECT_EQ(output[i], expected[i]) << "at output index " << i;
}

TEST(SOFIESafetensors, DefaultWeightFileName)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   EXPECT_EQ(RunModelDefaultFile("safetensors_mul", kInput), ExpectedOutput());
}

TEST(SOFIESafetensors, MissingFileThrows)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   EXPECT_TRUE(SessionConstructorThrows("safetensors_mul", "safetensors_mul_bogus.safetensors"));
}

TEST(SOFIESafetensors, WrongDtypeThrows)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   std::ifstream in("safetensors_mul.safetensors", std::ios::binary);
   std::string content{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
   in.close();
   const std::size_t pos = content.find("\"F32\"");
   ASSERT_NE(pos, std::string::npos);
   content.replace(pos, 5, "\"F64\"");
   std::ofstream out("safetensors_mul_bad_dtype.safetensors", std::ios::binary);
   out.write(content.data(), content.size());
   out.close();

   EXPECT_TRUE(SessionConstructorThrows("safetensors_mul", "safetensors_mul_bad_dtype.safetensors"));
}

TEST(SOFIESafetensors, ThirdPartyFileLayout)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   const std::string headerStr = R"({
   "__metadata__": {
      "generator": "hand-written \"test\" file\ngenerated for SOFIE",
      "format": "pt"
   },
   "tensor_W":
   {
      "data_offsets" : [ 0, 24 ],
      "shape" : [ 3, 2 ],
      "dtype" : "F32"
   }
})";
   std::ofstream out("safetensors_mul_thirdparty.safetensors", std::ios::binary);
   for (int i = 0; i < 8; ++i)
      out.put(char((headerStr.size() >> (8 * i)) & 0xff));
   out << headerStr;
   for (float value : kWeights) {
      std::uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      for (int i = 0; i < 4; ++i)
         out.put(char((bits >> (8 * i)) & 0xff));
   }
   out.close();

   const std::vector<float> output = RunModel("safetensors_mul", "safetensors_mul_thirdparty.safetensors", kInput);
   EXPECT_EQ(output, ExpectedOutput());
}

TEST(SOFIESafetensors, MalformedJsonThrows)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   std::string headerStr = R"({"tensor_W": {"dtype": "F32", "data_offsets": [0, 24},)";
   std::ofstream out("safetensors_mul_malformed.safetensors", std::ios::binary);
   for (int i = 0; i < 8; ++i)
      out.put(char((headerStr.size() >> (8 * i)) & 0xff));
   out << headerStr;
   out.close();

   EXPECT_TRUE(SessionConstructorThrows("safetensors_mul", "safetensors_mul_malformed.safetensors"));
}

TEST(SOFIESafetensors, InferenceFromMemoryBlob)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   const std::string blob = ReadWholeFile("safetensors_mul.safetensors");
   const std::vector<float> output = RunModelFromBlob("safetensors_mul", blob, kInput);
   EXPECT_EQ(output, ExpectedOutput());
}

TEST(SOFIESafetensors, InMemoryWriterMatchesFile)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   WriteMulModelFile("safetensors_mul.onnx", {2, 3}, kWeights);
   RModelParser_ONNX parser;
   RModel model = parser.Parse("safetensors_mul.onnx");
   model.Generate(Options::kSafetensorsWeightFile);
   EXPECT_EQ(model.WriteInitializedTensorsToBuffer(), ReadWholeFile("safetensors_mul.safetensors"));
}

TEST(SOFIESafetensors, TruncatedBlobThrows)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   const std::string full = ReadWholeFile("safetensors_mul.safetensors");
   ASSERT_GT(full.size(), 10u);
   EXPECT_TRUE(BlobSessionConstructorThrows("safetensors_mul", full.substr(0, full.size() - 10)));
}

TEST(SOFIESafetensors, WeightlessModelFromBlob)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   WriteTwoInputMulModelFile("safetensors_weightless.onnx", {2, 3});
   ASSERT_EQ(GenerateModel("safetensors_weightless.onnx"), "safetensors_weightless");
   DeclareModel("safetensors_weightless");
   std::vector<float> inX{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
   std::vector<float> inY{2.0f, -1.0f, 0.5f, 3.0f, -4.0f, 7.0f};
   std::vector<float> expected(inX.size());
   for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = inX[i] * inY[i];
   EXPECT_EQ(RunTwoInputModelFromBlob("safetensors_weightless", "", inX, inY), expected);
}

TEST(SOFIESafetensors, DeepNestingThrows)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   std::string header = R"({"tensor_W":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"__metadata__":)";
   header += std::string(10000, '[') + std::string(10000, ']') + "}";
   EXPECT_TRUE(BlobSessionConstructorThrows("safetensors_mul", MakeSafetensorsPayload(header, std::string(24, '\0'))));
}

TEST(SOFIESafetensors, ModerateNestingAccepted)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   std::string header = R"({"tensor_W":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"__metadata__":)";
   header += std::string(32, '[') + std::string(32, ']') + "}";
   EXPECT_FALSE(BlobSessionConstructorThrows("safetensors_mul", MakeSafetensorsPayload(header, std::string(24, '\0'))));
}

TEST(SOFIESafetensors, WrappingOffsetsThrow)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   const std::string header =
      R"({"tensor_W":{"dtype":"F32","shape":[2,3],"data_offsets":[0,18446744073709551600]}})";
   EXPECT_TRUE(BlobSessionConstructorThrows("safetensors_mul", MakeSafetensorsPayload(header, std::string(24, '\0'))));
}

TEST(SOFIESafetensors, NonIntegerOffsetsThrow)
{
   SOFIE_SKIP_ON_BIG_ENDIAN;
   const std::string header = R"({"tensor_W":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24.0]}})";
   EXPECT_TRUE(BlobSessionConstructorThrows("safetensors_mul", MakeSafetensorsPayload(header, std::string(24, '\0'))));
}
