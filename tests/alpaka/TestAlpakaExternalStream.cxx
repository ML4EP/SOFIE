#include "TestAlpakaCommon.h"

#include "ExternalStreamGemmRelu_GPU_ALPAKA.hxx"

TEST_F(SofieAlpakaTest, ExternalStream)
{
   TestStream stream;
   ASSERT_EQ(TestRtStreamCreateNonBlocking(&stream), TestRtSuccess);

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{15}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));
   for (Idx i = 0; i < 15; ++i) {
      A_ptr[i] = 1.0;
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{15}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{12}));

   {
      SOFIE_ExternalStreamGemmRelu::Session<TestTag> session(stream, "ExternalStreamGemmRelu_GPU_ALPAKA.dat");
      static_assert(std::is_same_v<decltype(session)::NativeStream, TestStream>);
      auto result = session.infer(A_d);
      ASSERT_EQ(TestRtStreamSynchronize(stream), TestRtSuccess);

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   ASSERT_EQ(TestRtStreamDestroy(stream), TestRtSuccess);

   float *res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   for (size_t i = 0; i < 12; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - 3.5f), DEFAULT_TOLERANCE);
   }
}
