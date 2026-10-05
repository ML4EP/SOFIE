#include "TestAlpakaCommon.h"

#include "IfSimple_FromONNX_GPU_ALPAKA.hxx"
#include "IfTwoOutputs_FromONNX_GPU_ALPAKA.hxx"

namespace {

// copy the first n elements of a device buffer (or view) to the host
template <typename TBuf>
std::vector<float> toHost(alpaka::DevCpu const &host, alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> &queue,
                          TBuf const &deviceBuf, std::size_t n)
{
   auto hostBuf = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{n}));
   alpaka::memcpy(queue, hostBuf, deviceBuf);
   alpaka::wait(queue);
   const float *p = alpaka::getPtrNative(hostBuf);
   return std::vector<float>(p, p + n);
}

} // namespace

// If operator (subgraphs): both branches read the outer-scope input X
TEST_F(SofieAlpakaTest, IfSimpleThen)
{
   std::vector<float> x = {1, -2, 3, -4, 5, -6};
   uint8_t cond = 1;
   auto cond_d = makeDeviceBuf<uint8_t>(host, device, queue, &cond, 1);
   auto x_d = makeDeviceBuf<float>(host, device, queue, x.data(), x.size());

   SOFIE_IfSimple::Session<alpaka::TagGpuCudaRt> session;
   auto result = session.infer(cond_d, x_d);
   auto out = toHost(host, queue, result, x.size());

   std::vector<float> correct = {1, 4, 9, 16, 25, 36};
   for (size_t i = 0; i < correct.size(); ++i)
      EXPECT_FLOAT_EQ(out[i], correct[i]) << "i=" << i;
}

TEST_F(SofieAlpakaTest, IfSimpleElse)
{
   std::vector<float> x = {1, -2, 3, -4, 5, -6};
   uint8_t cond = 0;
   auto cond_d = makeDeviceBuf<uint8_t>(host, device, queue, &cond, 1);
   auto x_d = makeDeviceBuf<float>(host, device, queue, x.data(), x.size());

   SOFIE_IfSimple::Session<alpaka::TagGpuCudaRt> session;
   auto result = session.infer(cond_d, x_d);
   auto out = toHost(host, queue, result, x.size());

   std::vector<float> correct = {-1, 2, -3, 4, -5, 6};
   for (size_t i = 0; i < correct.size(); ++i)
      EXPECT_FLOAT_EQ(out[i], correct[i]) << "i=" << i;
}

// the same session runs one branch after the other, depending on the condition of each call
TEST_F(SofieAlpakaTest, IfSimpleSwitchBranch)
{
   std::vector<float> x = {1, -2, 3, -4, 5, -6};
   auto x_d = makeDeviceBuf<float>(host, device, queue, x.data(), x.size());

   SOFIE_IfSimple::Session<alpaka::TagGpuCudaRt> session;
   for (uint8_t cond : {1, 0, 0, 1}) {
      auto cond_d = makeDeviceBuf<uint8_t>(host, device, queue, &cond, 1);
      auto result = session.infer(cond_d, x_d);
      auto out = toHost(host, queue, result, x.size());
      for (size_t i = 0; i < x.size(); ++i)
         EXPECT_FLOAT_EQ(out[i], cond ? x[i] * x[i] : -x[i]) << "cond=" << int(cond) << " i=" << i;
   }
}

// two outputs per branch, branch-local initializers and an operator reading the If outputs
TEST_F(SofieAlpakaTest, IfTwoOutputsThen)
{
   std::vector<float> x = {1, -2, 3, -4};
   uint8_t cond = 1;
   auto cond_d = makeDeviceBuf<uint8_t>(host, device, queue, &cond, 1);
   auto x_d = makeDeviceBuf<float>(host, device, queue, x.data(), x.size());

   SOFIE_IfTwoOutputs::Session<alpaka::TagGpuCudaRt> session;
   auto result = session.infer(cond_d, x_d);
   auto out = toHost(host, queue, result, x.size());

   std::vector<float> correct = {4, -3, 10, -6};
   for (size_t i = 0; i < correct.size(); ++i)
      EXPECT_FLOAT_EQ(out[i], correct[i]) << "i=" << i;
}

TEST_F(SofieAlpakaTest, IfTwoOutputsElse)
{
   std::vector<float> x = {1, -2, 3, -4};
   uint8_t cond = 0;
   auto cond_d = makeDeviceBuf<uint8_t>(host, device, queue, &cond, 1);
   auto x_d = makeDeviceBuf<float>(host, device, queue, x.data(), x.size());

   SOFIE_IfTwoOutputs::Session<alpaka::TagGpuCudaRt> session;
   auto result = session.infer(cond_d, x_d);
   auto out = toHost(host, queue, result, x.size());

   std::vector<float> correct = {1, 0, 5, 0};
   for (size_t i = 0; i < correct.size(); ++i)
      EXPECT_FLOAT_EQ(out[i], correct[i]) << "i=" << i;
}
