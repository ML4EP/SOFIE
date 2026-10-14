#include "SOFIE/RGPUModel.hxx"
#include "SOFIE/ROperator_Relu.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>

#ifdef SOFIE_TEST_HIP
#include <hip/hip_runtime.h>
#define SOFIE_GPU(name) hip##name
#else
#include <cuda_runtime.h>
#define SOFIE_GPU(name) cuda##name
#endif

namespace {
struct DeviceBuffer {
   void *data = nullptr;
   explicit DeviceBuffer(std::size_t bytes)
   {
      if (SOFIE_GPU(Malloc)(&data, bytes) != SOFIE_GPU(Success))
         throw std::runtime_error("Device allocation failed");
   }
   ~DeviceBuffer() { SOFIE_GPU(Free)(data); }
};
struct Stream {
   SOFIE_GPU(Stream_t) value{};
   Stream() { if (SOFIE_GPU(StreamCreate)(&value) != SOFIE_GPU(Success)) throw std::runtime_error("Stream creation failed"); }
   ~Stream() { SOFIE_GPU(StreamSynchronize)(value); SOFIE_GPU(StreamDestroy)(value); }
};
}

template<class Value> void CheckWorkspace()
{
   int device = 0;
   ASSERT_EQ(SOFIE_GPU(GetDevice)(&device), SOFIE_GPU(Success));
   SOFIE::RModel graph("relu_workspace", "test\n");
   graph.AddInputTensorInfo("input", SOFIE::GetTemplatedType(Value{}), std::vector<SOFIE::Dim>{SOFIE::Dim("batch"), SOFIE::Dim(3)});
   graph.AddInputTensorName("input");
   std::array<Value, 9> weights{1, 0, 0, 0, 1, 0, 0, 0, 1};
   graph.AddInitializedTensor("weights", SOFIE::GetTemplatedType(Value{}), {3, 3}, weights.data());
   graph.AddOperator(std::make_unique<SOFIE::ROperator_Gemm<Value>>(1.f, 0.f, 0, 0, "input", "weights", "hidden"));
   graph.AddOperator(std::make_unique<SOFIE::ROperator_Relu<Value>>("hidden", "output"));
   graph.AddOutputTensorNameList({"output"});
   SOFIE::RGPUModel model(std::move(graph));
#ifdef SOFIE_TEST_HIP
   model.Compile({SOFIE::RGPUModel::Backend::HIP, "", SOFIE_TEST_GPU_ARCH});
#else
   model.Compile({SOFIE::RGPUModel::Backend::CUDA, "", SOFIE_TEST_GPU_ARCH});
#endif
   Stream stream;
   auto session = model.CreateSession(stream.value, device, 4);
   ASSERT_GT(session->WorkspaceSize(), 0);
   ASSERT_EQ(session->WorkspaceSize() % 256, 0);
   DeviceBuffer firstArena(session->WorkspaceSize() + 256);
   DeviceBuffer secondArena(session->WorkspaceSize() + 256);
   DeviceBuffer input(sizeof(Value) * 12);
   DeviceBuffer output(sizeof(Value) * 12);
   const std::array<Value, 12> values{-1, 2, -3, 4, -5, 6, 7, -8, 9, -10, 11, 12};
   ASSERT_EQ(SOFIE_GPU(MemcpyAsync)(input.data, values.data(), sizeof(values), SOFIE_GPU(MemcpyHostToDevice), stream.value), SOFIE_GPU(Success));
   EXPECT_THROW(session->Infer(input.data, output.data, 4), std::runtime_error);
   EXPECT_THROW(session->SetWorkspace(nullptr, session->WorkspaceSize()), std::runtime_error);
   EXPECT_THROW(session->SetWorkspace(firstArena.data, session->WorkspaceSize() - 1), std::runtime_error);
   EXPECT_THROW(session->SetWorkspace(static_cast<char*>(firstArena.data) + 1, session->WorkspaceSize()), std::runtime_error);
   for (auto *arena : {&firstArena, &secondArena}) {
      ASSERT_EQ(SOFIE_GPU(MemsetAsync)(arena->data, 0x5a, session->WorkspaceSize() + 256, stream.value), SOFIE_GPU(Success));
      session->SetWorkspace(arena->data, session->WorkspaceSize());
      for (std::size_t batch : {4u, 1u, 3u}) {
         ASSERT_EQ(SOFIE_GPU(MemsetAsync)(output.data, 0, sizeof(values), stream.value), SOFIE_GPU(Success));
         session->Infer(input.data, output.data, batch);
         std::array<Value, 12> result{};
         ASSERT_EQ(SOFIE_GPU(MemcpyAsync)(result.data(), output.data, sizeof(result), SOFIE_GPU(MemcpyDeviceToHost), stream.value), SOFIE_GPU(Success));
         ASSERT_EQ(SOFIE_GPU(StreamSynchronize)(stream.value), SOFIE_GPU(Success));
         for (std::size_t index = 0; index < result.size(); ++index)
            EXPECT_FLOAT_EQ(float(result[index]), index < batch * 3 ? std::max(0.f, float(values[index])) : 0.f);
      }
      std::array<std::uint8_t, 256> guard{};
      ASSERT_EQ(SOFIE_GPU(Memcpy)(guard.data(), static_cast<char*>(arena->data) + session->WorkspaceSize(), guard.size(), SOFIE_GPU(MemcpyDeviceToHost)), SOFIE_GPU(Success));
      for (auto value : guard) EXPECT_EQ(value, 0x5a);
   }
   EXPECT_THROW(session->Infer(input.data, output.data, 5), std::invalid_argument);
   EXPECT_NO_THROW(session->Infer(nullptr, nullptr, 0));
   auto second = model.CreateSession(stream.value, device, 4);
   second->SetWorkspace(firstArena.data, second->WorkspaceSize());
   second->Infer(input.data, output.data, 2);
   ASSERT_EQ(SOFIE_GPU(StreamSynchronize)(stream.value), SOFIE_GPU(Success));
}

TEST(SOFIERuntimeWorkspace, Float32) { CheckWorkspace<float>(); }
TEST(SOFIERuntimeWorkspace, Float16) { CheckWorkspace<SOFIE::Float16>(); }
