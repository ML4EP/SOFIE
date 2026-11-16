#include "TestAlpakaCommon.h"

#include <fstream>
#include <set>
#include <sstream>

#include "CustomMemoryChain_GPU_ALPAKA.hxx"
#include "CustomMemoryStreamChain_GPU_ALPAKA.hxx"
#include "CustomMemoryIf_GPU_ALPAKA.hxx"
#include "CustomMemoryDebugChain_GPU_ALPAKA.hxx"
#include "DebugDefaultChain_GPU_ALPAKA.hxx"
#include "DebugStreamChain_GPU_ALPAKA.hxx"

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
      EXPECT_EQ(TestRtMalloc(&ptr, bytes), TestRtSuccess);
      ++state.allocations;
      state.blockBytes = bytes;
      state.block = ptr;
      return ptr;
   };
   provider.dealloc = [&state](void *ptr) {
      EXPECT_EQ(TestRtFree(ptr), TestRtSuccess);
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
   alpaka::Buf<TestDev, float, Dim, Idx> UploadOnes(std::size_t n)
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
   using Session = SOFIE_CustomMemoryChain::Session<TestTag>;
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
   using Session = SOFIE_CustomMemoryChain::Session<TestTag>;
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
   using Session = SOFIE_CustomMemoryChain::Session<TestTag>;
   auto x_d = UploadOnes(15);
   char *arena = nullptr;
   ASSERT_EQ(TestRtMalloc(&arena, 1 << 20), TestRtSuccess);
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
   ASSERT_EQ(TestRtFree(arena), TestRtSuccess);
}

TEST_F(SofieAlpakaCustomMemoryTest, MissingAllocIsRejected)
{
   using Session = SOFIE_CustomMemoryChain::Session<TestTag>;
   SOFIE_CustomMemoryChain::MemoryProvider provider;
   EXPECT_THROW(Session(provider, "CustomMemoryChain_GPU_ALPAKA.dat"), std::runtime_error);
}

TEST_F(SofieAlpakaCustomMemoryTest, ExternalStreamAndCustomMemory)
{
   using Session = SOFIE_CustomMemoryStreamChain::Session<TestTag>;
   TestStream stream;
   ASSERT_EQ(TestRtStreamCreateNonBlocking(&stream), TestRtSuccess);
   Provider state;
   auto x_d = UploadOnes(15);
   {
      auto provider = MakeProvider<SOFIE_CustomMemoryStreamChain::MemoryProvider>(state, true);
      Session session(stream, provider, "CustomMemoryStreamChain_GPU_ALPAKA.dat");
      auto result = session.infer(x_d);
      ASSERT_EQ(TestRtStreamSynchronize(stream), TestRtSuccess);
      CheckChain(result);
   }
   EXPECT_EQ(state.allocations, 1);
   EXPECT_EQ(state.deallocations, 1);
   ASSERT_EQ(TestRtStreamDestroy(stream), TestRtSuccess);
}

TEST_F(SofieAlpakaCustomMemoryTest, SubGraphSessionsUseTheProvider)
{
   using Session = SOFIE_CustomMemoryIf::Session<TestTag>;
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

TEST_F(SofieAlpakaCustomMemoryTest, DebugModeReportsTheRegions)
{
   using Session = SOFIE_CustomMemoryDebugChain::Session<TestTag>;
   Provider state;
   auto x_d = UploadOnes(15);
   auto provider = MakeProvider<SOFIE_CustomMemoryDebugChain::MemoryProvider>(state, true);
   Session session(provider, "CustomMemoryDebugChain_GPU_ALPAKA.dat");

   auto const &regions = session.GetMemoryRegions();
   ASSERT_GE(regions.size(), 4u);
   std::size_t padding = 0;
   for (auto const &r : regions) {
      EXPECT_EQ(r.reservedBytes % 256, 0u) << r.name;
      EXPECT_GE(r.reservedBytes, r.requestedBytes) << r.name;
      padding += r.reservedBytes - r.requestedBytes;
   }
   EXPECT_EQ(session.GetPaddingBytes(), padding);
   EXPECT_GT(padding, 0u);
   EXPECT_NO_THROW(session.ValidateMemory());

   const std::string report = session.GetMemoryReport();
   EXPECT_NE(report.find("managed by the session: " + std::to_string(state.blockBytes)), std::string::npos) << report;
   EXPECT_NE(report.find("\nW  "), std::string::npos) << report;

   auto result = session.infer(x_d);
   CheckChain(result);
}

TEST_F(SofieAlpakaCustomMemoryTest, DebugModeRejectsBadAssignments)
{
   using Session = SOFIE_CustomMemoryDebugChain::Session<TestTag>;
   Provider state;
   auto provider = MakeProvider<SOFIE_CustomMemoryDebugChain::MemoryProvider>(state, false);

   provider.assign = [&state](std::size_t) -> void * { return state.block; };
   EXPECT_THROW(Session(provider, "CustomMemoryDebugChain_GPU_ALPAKA.dat"), std::runtime_error);

   provider.assign = [&state](std::size_t offset) -> void * { return static_cast<char *>(state.block) + offset + 4; };
   EXPECT_THROW(Session(provider, "CustomMemoryDebugChain_GPU_ALPAKA.dat"), std::runtime_error);
}

TEST_F(SofieAlpakaCustomMemoryTest, DebugModeTracesTheSession)
{
   using Session = SOFIE_CustomMemoryDebugChain::Session<TestTag>;
   Provider state;
   auto x_d = UploadOnes(15);
   auto provider = MakeProvider<SOFIE_CustomMemoryDebugChain::MemoryProvider>(state, true);

   testing::internal::CaptureStdout();
   {
      Session session(provider, "CustomMemoryDebugChain_GPU_ALPAKA.dat");
      auto result = session.infer(x_d);
   }
   const std::string log = testing::internal::GetCapturedStdout();
   for (const char *expected : {"[SOFIE debug] constructing the session of model CustomMemoryDebugChain",
                                "reading the weights from file", "planned device memory block", "memory block allocated through the provider",
                                "tensor W placed at offset", "uploading the weights of tensor W", "kernel objects instantiated with the session",
                                "adding the BLAS configuration of operator 0", "the session of model CustomMemoryDebugChain is constructed",
                                "running the inference of model CustomMemoryDebugChain", "launching operator", "kernel launch(es)",
                                "are enqueued", "released through the provider"})
      EXPECT_NE(log.find(expected), std::string::npos) << expected << "\n" << log;

   SOFIE_CustomMemoryDebugChain::DebugLogEnabled() = false;
   testing::internal::CaptureStdout();
   {
      Session session(provider, "CustomMemoryDebugChain_GPU_ALPAKA.dat");
   }
   EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
   SOFIE_CustomMemoryDebugChain::DebugLogEnabled() = true;
}

TEST_F(SofieAlpakaCustomMemoryTest, DebugModeWithoutCustomMemoryTracesTheAlpakaAllocations)
{
   using Session = SOFIE_DebugDefaultChain::Session<TestTag>;
   auto x_d = UploadOnes(15);
   testing::internal::CaptureStdout();
   {
      Session session("DebugDefaultChain_GPU_ALPAKA.dat");
      auto result = session.infer(x_d);
      CheckChain(result);
   }
   const std::string log = testing::internal::GetCapturedStdout();
   for (const char *expected : {"device buffer deviceBuf_W allocated by alpaka: 80 bytes", "allocated the unified pool",
                                "placed in the pool at offset", "launching operator"})
      EXPECT_NE(log.find(expected), std::string::npos) << expected << "\n" << log;
}

TEST_F(SofieAlpakaCustomMemoryTest, DebugModeReportsTheMemoryWithoutCustomMemory)
{
   using Session = SOFIE_DebugDefaultChain::Session<TestTag>;
   auto x_d = UploadOnes(15);
   Session session("DebugDefaultChain_GPU_ALPAKA.dat");

   auto const &regions = session.GetMemoryRegions();
   std::set<std::string> origins;
   for (auto const &r : regions)
      origins.insert(r.origin);
   EXPECT_TRUE(origins.count("alpaka"));
   EXPECT_TRUE(origins.count("pool"));
   EXPECT_GT(session.GetPaddingBytes(), 0u);
   EXPECT_NO_THROW(session.ValidateMemory());

   const std::string report = session.GetMemoryReport();
   EXPECT_NE(report.find("\nW  alpaka"), std::string::npos) << report;
   EXPECT_NE(report.find("pool  "), std::string::npos) << report;
   EXPECT_NE(report.find("managed by the session: " + std::to_string(session.GetIntermediateMemoryPoolSize())), std::string::npos) << report;

   auto result = session.infer(x_d);
   CheckChain(result);
}

TEST_F(SofieAlpakaCustomMemoryTest, DebugModeWithAnExternalStream)
{
   using Session = SOFIE_DebugStreamChain::Session<TestTag>;
   TestStream stream;
   ASSERT_EQ(TestRtStreamCreateNonBlocking(&stream), TestRtSuccess);
   auto x_d = UploadOnes(15);
   testing::internal::CaptureStdout();
   {
      Session session(stream, "DebugStreamChain_GPU_ALPAKA.dat");
      EXPECT_NO_THROW(session.ValidateMemory());
      auto result = session.infer(x_d);
      ASSERT_EQ(TestRtStreamSynchronize(stream), TestRtSuccess);
      CheckChain(result);
   }
   const std::string log = testing::internal::GetCapturedStdout();
   EXPECT_NE(log.find("launching operator"), std::string::npos) << log;
   ASSERT_EQ(TestRtStreamDestroy(stream), TestRtSuccess);
}
