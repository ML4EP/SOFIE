#include "SOFIE/RGPUModel.hxx"
#include "SOFIE/RModelParser_ONNX.hxx"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <array>

#ifdef SOFIE_TEST_GPU_ARCH
#ifdef SOFIE_TEST_HIP
#include <hip/hip_runtime.h>
#define SOFIE_GPU(name) hip##name
#else
#include <cuda_runtime.h>
#define SOFIE_GPU(name) cuda##name
#endif
#endif

namespace {
const std::array<const char *, 5> cases{"dense_raw", "dense_typed", "dense_external", "attention", "selection"};
std::filesystem::path Fixture(const std::string &name)
{
   return std::filesystem::path(SOFIE_FP16_MODELS_DIR) / name;
}
}

TEST(SOFIEFloat16Onnx, ParsesAndGeneratesFullGraphs)
{
   for (const auto *name : cases) {
      SCOPED_TRACE(name);
      SOFIE::RModelParser_ONNX parser;
      auto model = parser.Parse(Fixture(std::string(name) + ".onnx").string());
      model.SetExternalWorkspace();
      EXPECT_NO_THROW(model.GenerateGPU_ALPAKA(SOFIE::Options::kNoWeightFile));
      EXPECT_EQ(model.GetTensorType("input"), SOFIE::ETensorType::FLOAT16);
      EXPECT_EQ(model.GetTensorType("output"), SOFIE::ETensorType::FLOAT16);
      EXPECT_NE(model.ReturnGenerated().find("ViewConstH1D"), std::string::npos);
      if (std::string(name).find("dense_") == 0) {
         EXPECT_NE(model.ReturnGenerated().find(
            "static_cast<const SOFIE::Float16*>(alpaka::getPtrNative(deviceBuf_bias))"), std::string::npos);
         EXPECT_EQ(model.ReturnGenerated().find(
            "static_cast<const float*>(alpaka::getPtrNative(deviceBuf_bias))"), std::string::npos);
      }
   }
}

TEST(SOFIEFloat16Onnx, RejectsInvalidInitializerBits)
{
   SOFIE::RModelParser_ONNX parser;
   EXPECT_THROW(parser.Parse(Fixture("invalid_bits.onnx").string()), std::runtime_error);
}

#ifdef SOFIE_TEST_GPU_ARCH
namespace {
struct Buffer {
   void *data = nullptr;
   explicit Buffer(std::size_t bytes)
   {
      if (SOFIE_GPU(Malloc)(&data, bytes) != SOFIE_GPU(Success))
         throw std::runtime_error("GPU allocation failed");
   }
   ~Buffer() { SOFIE_GPU(Free)(data); }
};
struct Stream {
   SOFIE_GPU(Stream_t) value{};
   Stream()
   {
      if (SOFIE_GPU(StreamCreate)(&value) != SOFIE_GPU(Success))
         throw std::runtime_error("GPU stream creation failed");
   }
   ~Stream() { SOFIE_GPU(StreamSynchronize)(value); SOFIE_GPU(StreamDestroy)(value); }
};
std::vector<SOFIE::Float16> ReadHalf(const std::filesystem::path &path, std::size_t elements)
{
   std::ifstream input(path, std::ios::binary);
   std::vector<SOFIE::Float16> result(elements);
   for (auto &value : result) {
      unsigned char bytes[2];
      if (!input.read(reinterpret_cast<char *>(bytes), 2))
         throw std::runtime_error("Cannot read FP16 fixture " + path.string());
      value = SOFIE::Float16::FromBits(bytes[0] | (std::uint16_t(bytes[1]) << 8));
   }
   return result;
}
}

TEST(SOFIEFloat16Onnx, MatchesReferenceOnBorrowedStreamAndReboundArena)
{
   int device = 0;
   ASSERT_EQ(SOFIE_GPU(GetDevice)(&device), SOFIE_GPU(Success));
   for (const auto *name : cases) {
      SCOPED_TRACE(name);
      SOFIE::RModelParser_ONNX parser;
      SOFIE::RGPUModel model(parser.Parse(Fixture(std::string(name) + ".onnx").string()));
      ASSERT_EQ(model.GetPrecision(), SOFIE::RGPUModel::Precision::Float16);
#ifdef SOFIE_TEST_HIP
      model.Compile({SOFIE::RGPUModel::Backend::HIP, "", SOFIE_TEST_GPU_ARCH});
#else
      model.Compile({SOFIE::RGPUModel::Backend::CUDA, "", SOFIE_TEST_GPU_ARCH});
#endif
      Stream stream;
      auto session = model.CreateSession(stream.value, device, 4);
      Buffer input(model.InputSize() * 4 * 2), output(model.OutputSize() * 4 * 2);
      Buffer arena(session->WorkspaceSize() + 256), replacement(session->WorkspaceSize() + 256);
      for (auto *memory : {&arena, &replacement}) {
         ASSERT_EQ(SOFIE_GPU(MemsetAsync)(memory->data, 0x5a, session->WorkspaceSize() + 256, stream.value), SOFIE_GPU(Success));
         session->SetWorkspace(memory->data, session->WorkspaceSize());
         for (const std::size_t batch : {4u, 1u, 3u}) {
            const auto prefix = std::string(name) + "_" + std::to_string(batch);
            const auto values = ReadHalf(Fixture(prefix + ".input"), batch * model.InputSize());
            const auto expected = ReadHalf(Fixture(prefix + ".expected"), batch * model.OutputSize());
            std::vector<SOFIE::Float16> actual(expected.size());
            ASSERT_EQ(SOFIE_GPU(MemcpyAsync)(input.data, values.data(), values.size() * 2,
               SOFIE_GPU(MemcpyHostToDevice), stream.value), SOFIE_GPU(Success));
            session->Infer(input.data, output.data, batch);
            ASSERT_EQ(SOFIE_GPU(MemcpyAsync)(actual.data(), output.data, actual.size() * 2,
               SOFIE_GPU(MemcpyDeviceToHost), stream.value), SOFIE_GPU(Success));
            ASSERT_EQ(SOFIE_GPU(StreamSynchronize)(stream.value), SOFIE_GPU(Success));
            for (std::size_t index = 0; index < expected.size(); ++index)
               EXPECT_NEAR(float(actual[index]), float(expected[index]), 0.01f);
         }
         std::array<unsigned char, 256> guard{};
         ASSERT_EQ(SOFIE_GPU(Memcpy)(guard.data(), static_cast<char *>(memory->data) + session->WorkspaceSize(),
            guard.size(), SOFIE_GPU(MemcpyDeviceToHost)), SOFIE_GPU(Success));
         for (const auto value : guard) EXPECT_EQ(value, 0x5a);
      }
   }
}
#endif
