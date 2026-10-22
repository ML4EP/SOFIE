#include "TestAlpakaCommon.h"

#include <fstream>
#include <set>
#include <sstream>

#include "CustomMemoryChain_GPU_ALPAKA.hxx"
#include "CustomMemoryStreamChain_GPU_ALPAKA.hxx"
#include "CustomMemoryIf_GPU_ALPAKA.hxx"

namespace {

struct Provider {
   int allocations = 0;
   int deallocations = 0;
   std::size_t blockBytes = 0;
   void *block = nullptr;
   std::vector<std::size_t> assignedOffsets;
};

template <class MemoryProvider>
MemoryProvider MakeProvider(Provider &state, bool withAssign)
{
   MemoryProvider provider;
   provider.alloc = [&state](std::size_t bytes) -> void * {
      void *ptr = nullptr;
      EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
      ++state.allocations;
      state.blockBytes = bytes;
      state.block = ptr;
      return ptr;
   };
   provider.dealloc = [&state](void *ptr) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
      ++state.deallocations;
   };
   if (withAssign) {
      provider.assign = [&state](std::size_t offset) -> void * {
         state.assignedOffsets.push_back(offset);
         return static_cast<char *>(state.block) + offset;
      };
   }
   return provider;
}

float ChainReference()
{
   const float sigmoid = 1.f / (1.f + std::exp(-3.5f));
   return std::tanh(sigmoid);
}

std::string ReadFile(const std::string &name)
{
   std::ifstream in(name);
   std::stringstream ss;
   ss << in.rdbuf();
   return ss.str();
}

bool AllocatesWithAlpaka(const std::string &code)
{
   std::size_t pos = 0;
   while ((pos = code.find("alpaka::allocBuf<", pos)) != std::string::npos) {
      std::size_t end = code.find('\n', pos);
      if (code.substr(pos, end - pos).find(", Idx>(devAcc") != std::string::npos)
         return true;
      pos += 1;
   }
   return false;
}

}

class SofieAlpakaCustomMemoryTest : public SofieAlpakaTest {
protected:
   alpaka::Buf<alpaka::DevCudaRt, float, Dim, Idx> UploadOnes(std::size_t n)
   {
      auto h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{n}));
      float *p = reinterpret_cast<float *>(alpaka::getPtrNative(h));
      for (std::size_t i = 0; i < n; ++i)
         p[i] = 1.f;
      auto d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{n}));
      alpaka::memcpy(queue, d, h);
      alpaka::wait(queue);
      return d;
   }

   template <class Result>
   void CheckChain(Result &result)
   {
      auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{12}));
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
      float *res = reinterpret_cast<float *>(alpaka::getPtrNative(result_h));
      for (std::size_t i = 0; i < 12; ++i)
         EXPECT_LE(std::abs(res[i] - ChainReference()), DEFAULT_TOLERANCE);
   }
};

TEST_F(SofieAlpakaCustomMemoryTest, GeneratedCodeHasNoAlpakaDeviceAllocation)
{
   for (const char *header : {"CustomMemoryChain_GPU_ALPAKA.hxx", "CustomMemoryStreamChain_GPU_ALPAKA.hxx",
                              "CustomMemoryIf_GPU_ALPAKA.hxx"}) {
      const std::string code = ReadFile(header);
      ASSERT_FALSE(code.empty()) << header;
      EXPECT_FALSE(AllocatesWithAlpaka(code)) << header;
   }
}

TEST_F(SofieAlpakaCustomMemoryTest, SingleBlockAllocatedAndAssigned)
{
   using Session = SOFIE_CustomMemoryChain::Session<alpaka::TagGpuCudaRt>;
   Provider state;
   auto x_d = UploadOnes(15);
   {
      auto provider = MakeProvider<SOFIE_CustomMemoryChain::MemoryProvider>(state, true);
      Session session(provider, "CustomMemoryChain_GPU_ALPAKA.dat");
      EXPECT_EQ(state.allocations, 1);
      EXPECT_EQ(state.deallocations, 0);
      EXPECT_EQ(session.GetIntermediateMemoryPoolSize(), state.blockBytes);

      EXPECT_GE(state.assignedOffsets.size(), 4u);
      for (std::size_t offset : state.assignedOffsets) {
         EXPECT_EQ(offset % 256, 0u);
         EXPECT_LT(offset, state.blockBytes);
      }
      std::set<std::size_t> distinct(state.assignedOffsets.begin(), state.assignedOffsets.end());
      EXPECT_GE(distinct.size(), 3u);

      auto result = session.infer(x_d);
      CheckChain(result);
   }
   EXPECT_EQ(state.deallocations, 1);
}

TEST_F(SofieAlpakaCustomMemoryTest, AssignIsOptional)
{
   using Session = SOFIE_CustomMemoryChain::Session<alpaka::TagGpuCudaRt>;
   Provider state;
   auto x_d = UploadOnes(15);
   {
      auto provider = MakeProvider<SOFIE_CustomMemoryChain::MemoryProvider>(state, false);
      Session session(provider, "CustomMemoryChain_GPU_ALPAKA.dat");
      EXPECT_EQ(state.allocations, 1);
      auto result = session.infer(x_d);
      CheckChain(result);
   }
   EXPECT_EQ(state.deallocations, 1);
}

TEST_F(SofieAlpakaCustomMemoryTest, AssignedPointerIsUsed)
{
   using Session = SOFIE_CustomMemoryChain::Session<alpaka::TagGpuCudaRt>;
   auto x_d = UploadOnes(15);
   char *arena = nullptr;
   ASSERT_EQ(cudaMalloc(&arena, 1 << 20), cudaSuccess);
   SOFIE_CustomMemoryChain::MemoryProvider provider;
   std::size_t blockBytes = 0;
   int assignCalls = 0;
   provider.alloc = [&](std::size_t bytes) -> void * {
      blockBytes = bytes;
      return arena + 4096;
   };
   provider.assign = [&](std::size_t offset) -> void * {
      ++assignCalls;
      return arena + 4096 + offset;
   };
   {
      Session session(provider, "CustomMemoryChain_GPU_ALPAKA.dat");
      EXPECT_GE(assignCalls, 4);
      EXPECT_LE(blockBytes, (1u << 20) - 4096);
      auto result = session.infer(x_d);
      CheckChain(result);
   }
   provider.assign = [](std::size_t) -> void * { return nullptr; };
   EXPECT_THROW(Session(provider, "CustomMemoryChain_GPU_ALPAKA.dat"), std::runtime_error);
   ASSERT_EQ(cudaFree(arena), cudaSuccess);
}

TEST_F(SofieAlpakaCustomMemoryTest, MissingAllocIsRejected)
{
   using Session = SOFIE_CustomMemoryChain::Session<alpaka::TagGpuCudaRt>;
   SOFIE_CustomMemoryChain::MemoryProvider provider;
   EXPECT_THROW(Session(provider, "CustomMemoryChain_GPU_ALPAKA.dat"), std::runtime_error);
}

TEST_F(SofieAlpakaCustomMemoryTest, ExternalStreamAndCustomMemory)
{
   using Session = SOFIE_CustomMemoryStreamChain::Session<alpaka::TagGpuCudaRt>;
   cudaStream_t stream;
   ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
   Provider state;
   auto x_d = UploadOnes(15);
   {
      auto provider = MakeProvider<SOFIE_CustomMemoryStreamChain::MemoryProvider>(state, true);
      Session session(stream, provider, "CustomMemoryStreamChain_GPU_ALPAKA.dat");
      auto result = session.infer(x_d);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      CheckChain(result);
   }
   EXPECT_EQ(state.allocations, 1);
   EXPECT_EQ(state.deallocations, 1);
   ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(SofieAlpakaCustomMemoryTest, SubGraphSessionsUseTheProvider)
{
   using Session = SOFIE_CustomMemoryIf::Session<alpaka::TagGpuCudaRt>;
   Provider state;
   auto x_d = UploadOnes(15);
   auto xh = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{15}));
   float *xp = reinterpret_cast<float *>(alpaka::getPtrNative(xh));
   for (std::size_t i = 0; i < 15; ++i)
      xp[i] = 0.1f * static_cast<float>(i) - 0.5f;
   alpaka::memcpy(queue, x_d, xh);
   alpaka::wait(queue);
   {
      auto provider = MakeProvider<SOFIE_CustomMemoryIf::MemoryProvider>(state, true);
      Session session(provider, "");
      EXPECT_GE(state.allocations, 1);
      for (uint8_t cond : {1, 0}) {
         auto ch = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{1}));
         *reinterpret_cast<uint8_t *>(alpaka::getPtrNative(ch)) = cond;
         auto c_d = alpaka::allocBuf<uint8_t, Idx>(device, Ext1D::all(Idx{1}));
         alpaka::memcpy(queue, c_d, ch);
         alpaka::wait(queue);
         auto y = session.infer(c_d, x_d);
         auto yh = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{15}));
         alpaka::memcpy(queue, yh, y);
         alpaka::wait(queue);
         float *yp = reinterpret_cast<float *>(alpaka::getPtrNative(yh));
         for (std::size_t i = 0; i < 15; ++i) {
            const float ref = cond ? xp[i] * xp[i] : 1.f / (1.f + std::exp(xp[i]));
            EXPECT_LE(std::abs(yp[i] - ref), DEFAULT_TOLERANCE);
         }
      }
   }
   EXPECT_EQ(state.allocations, state.deallocations);
}
