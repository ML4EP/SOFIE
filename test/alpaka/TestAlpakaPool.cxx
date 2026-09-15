#include "TestAlpakaCommon.h"

#include "MaxPool1d_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/MaxPool1d.ref.hxx"
#include "MaxPool2d_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/MaxPool2d.ref.hxx"
#include "MaxPool3d_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/MaxPool3d.ref.hxx"
#include "AvgPool_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/AvgPool.ref.hxx"
#include "AvgPoolPad_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/AvgPoolPad.ref.hxx"
#include "AvgPoolCountIncludePad_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/AvgPoolCountIncludePad.ref.hxx"
#include "GlobalAvgPool2d_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/GlobalAvgPool2d.ref.hxx"
#include "DynamicMaxPool_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicPad_FromONNX_GPU_ALPAKA.hxx"

TEST_F(SofieAlpakaTest, DynamicMaxPool)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    for (std::size_t N : {std::size_t(1), std::size_t(3)}) {
        std::vector<float> input(N * 16);
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 17) - 8.0f;

        auto input_d = makeDeviceBuf<float>(host, device, queue, input.data(), input.size());
        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N * 4}));
        {
            SOFIE_DynamicMaxPool::Session<alpaka::TagGpuCudaRt> session("DynamicMaxPool_FromONNX_GPU_ALPAKA.dat", N);
            auto result = session.infer(N, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n) {
            const float* img = input.data() + n * 16;
            float expected[4];
            for (int oh = 0; oh < 2; ++oh)
                for (int ow = 0; ow < 2; ++ow) {
                    float m = -1e30f;
                    for (int kh = 0; kh < 2; ++kh)
                        for (int kw = 0; kw < 2; ++kw)
                            m = std::max(m, img[(oh * 2 + kh) * 4 + (ow * 2 + kw)]);
                    expected[oh * 2 + ow] = m;
                }
            for (int i = 0; i < 4; ++i)
                EXPECT_LE(std::abs(res[n * 4 + i] - expected[i]), TOLERANCE) << "n=" << n << " i=" << i << " N=" << N;
        }
    }
}

// Pad on a tensor with dynamic dims [N, 3, n_pf], pads = (1,0,2 ; 1,0,3), constant_value = -1
TEST_F(SofieAlpakaTest, DynamicPad)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    constexpr std::size_t C = 3;
    constexpr int64_t padBeforeN = 1, padAfterN = 1;
    constexpr int64_t padBeforeC = 0, padAfterC = 0;
    constexpr int64_t padBeforeP = 2, padAfterP = 3;
    constexpr float CONST_VALUE = -1.0f;

    const std::size_t Ns[] = {1, 8};
    const std::size_t Ps[] = {1, 5};   // n_pf
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], P = Ps[t];
        const std::size_t inSize = N * C * P;
        const std::size_t outN = N + padBeforeN + padAfterN;
        const std::size_t outP = P + padBeforeP + padAfterP;
        const std::size_t outSize = outN * C * outP;

        std::vector<float> input(inSize);
        for (std::size_t i = 0; i < inSize; ++i) input[i] = static_cast<float>(i % 10) - 5.0f;

        auto input_d = makeDeviceBuf<float>(host, device, queue, input.data(), inSize);
        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));

        {
            SOFIE_DynamicPad::Session<alpaka::TagGpuCudaRt> session("DynamicPad_FromONNX_GPU_ALPAKA.dat", N, P);
            auto result = session.infer(N, P, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < outN; ++n)
            for (std::size_t c = 0; c < C; ++c)
                for (std::size_t p = 0; p < outP; ++p) {
                    float expected = CONST_VALUE;
                    bool interior = (n >= static_cast<std::size_t>(padBeforeN)) && (n < static_cast<std::size_t>(padBeforeN) + N) &&
                                    (c >= static_cast<std::size_t>(padBeforeC)) && (c < static_cast<std::size_t>(padBeforeC) + C) &&
                                    (p >= static_cast<std::size_t>(padBeforeP)) && (p < static_cast<std::size_t>(padBeforeP) + P);
                    if (interior) {
                        std::size_t in_n = n - padBeforeN, in_c = c - padBeforeC, in_p = p - padBeforeP;
                        expected = input[in_n * C * P + in_c * P + in_p];
                    }
                    std::size_t idx = n * C * outP + c * outP + p;
                    EXPECT_LE(std::abs(res[idx] - expected), TOLERANCE)
                        << "N=" << N << " P=" << P << " n=" << n << " c=" << c << " p=" << p;
                }
    }
}

TEST_F(SofieAlpakaTest, MaxPool2d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({
       0.6266,  0.1656,  0.2753, -0.4558, -1.4592,  0.9285, -1.3410,  1.3223, -0.5936, -1.3648,
      -0.2989,  0.5901, -0.8845, -0.0433,  0.8314, -1.7159, -0.5765,  0.8678,  1.0257,  0.7847,
      -0.3421, -1.2364, -0.5805,  0.4421,  1.2184,  0.5043,  1.6823, -1.0483, -2.2798, -1.8927,
       0.7716,  0.0405,  0.3121, -0.3011, -0.3266, -1.9660,  1.0837,  0.2317,  0.9084, -0.3285,
      -0.9398, -0.2065, -0.9499, -0.9739, -0.1288, -0.1375, -1.2612,  0.8810,  0.8506,  0.4455
   });

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(MaxPool2d_ExpectedOutput::output) / sizeof(float)}));

   {
      SOFIE_MaxPool2d::Session<alpaka::TagGpuCudaRt> session("MaxPool2d_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = MaxPool2d_ExpectedOutput::output;
   constexpr size_t nOut_maxpool2d = sizeof(MaxPool2d_ExpectedOutput::output) / sizeof(float);

   for (size_t i = 0; i < nOut_maxpool2d; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

TEST_F(SofieAlpakaTest, MaxPool1d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({
       0.0907,  0.1029,  0.8143,  1.4497, -0.7785,  0.3825, -0.3764,  1.5785, -0.0835,  0.1622,
       1.5867,  0.9823, -0.8821,  0.4439, -0.1378, -0.2273, -0.0198, -2.0230,  0.0905,  0.6674,
      -1.4290, -1.3100, -0.9439, -0.0833, -0.1919,  0.6886,  0.9389, -1.2914, -1.3584, -2.0341,
      -0.3269,  0.1704,  1.1776,  1.3972, -1.8874, -1.5334,  1.1541,  0.3011,  0.6569, -2.3504,
       0.4033,  0.1142,  2.2846, -1.3948, -0.8573,  0.5756, -1.0864,  0.2283,  0.8947,  1.7627,
      -0.1657,  0.0649, -1.6066,  0.4162, -1.1525, -0.8184,  1.1324, -1.1086,  0.1061,  1.0071
   }); // took from reference output

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(MaxPool1d_ExpectedOutput::output) / sizeof(float)}));

   {
      SOFIE_MaxPool1d::Session<alpaka::TagGpuCudaRt> session("MaxPool1d_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = MaxPool1d_ExpectedOutput::output;
   constexpr size_t nOut_maxpool1d = sizeof(MaxPool1d_ExpectedOutput::output) / sizeof(float);
   for (size_t i = 0; i < nOut_maxpool1d; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

TEST_F(SofieAlpakaTest, MaxPool3d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({
      -2.6496,  1.0476, -0.5153,  0.3771,  0.4129, -0.3077, -0.8717, -0.8040, -0.3525,
      -0.1765, -0.3364,  0.8737, -0.2381, -0.8297,  0.4666,  0.6984, -0.6760,  0.6298,
       1.3833,  0.1101,  0.2039, -0.5477,  0.2341,  0.9181,  0.3842,  0.2428,  1.7924
   });// took from reference output

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(MaxPool3d_ExpectedOutput::output) / sizeof(float)}));

   {
      SOFIE_MaxPool3d::Session<alpaka::TagGpuCudaRt> session("MaxPool3d_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = MaxPool3d_ExpectedOutput::output;
   constexpr size_t nOut_maxpool3d = sizeof(MaxPool3d_ExpectedOutput::output) / sizeof(float);

   for (size_t i = 0; i < nOut_maxpool3d; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

// AveragePool with no padding (kernel 3x2, stride [2,1]); this re uses the existing AvgPool model and CPU reference

TEST_F(SofieAlpakaTest, AvgPool)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({
      0.4764, -0.1976,  1.6506, -0.2421,  0.6412,  1.9985,  0.3938,
            0.1347,  0.2204, -0.7503,
           0.2139,  0.7285, -0.0210, -0.4585, -1.5333, -0.4772,  0.5560,
            0.6323, -2.5372,  1.4906,
          -1.1062, -0.9703,  0.2366, -0.9184,  0.3014,  0.7985, -0.6841,
           -2.2854, -2.7728, -1.2806,
          -1.0947, -0.5990, -0.3033, -1.9042, -0.5403,  0.2332,  0.9215,
           -0.1549,  0.0557, -0.5567,
          -1.4971,  0.5386, -0.2922,  0.4860, -0.3973, -0.4624,  0.4514,
            0.2385,  0.3783, -1.0500
   });

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(AvgPool_ExpectedOutput::output) / sizeof(float)}));

   {
      SOFIE_AvgPool::Session<alpaka::TagGpuCudaRt> session("AvgPool_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = AvgPool_ExpectedOutput::output;
   constexpr size_t nOut_avgpool = sizeof(AvgPool_ExpectedOutput::output) / sizeof(float);

   for (size_t i = 0; i < nOut_avgpool; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

// AveragePool 3x3, pads 1 all round, count_include_pad=0 (default)

TEST_F(SofieAlpakaTest, AvgPoolPad)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(16);
   for (size_t i = 0; i < input.size(); ++i) input[i] = float(i);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(AvgPoolPad_ExpectedOutput::output) / sizeof(float)}));

   {
      SOFIE_AvgPoolPad::Session<alpaka::TagGpuCudaRt> session("AvgPoolPad_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = AvgPoolPad_ExpectedOutput::output;
   constexpr size_t nOut_avgpoolpad = sizeof(AvgPoolPad_ExpectedOutput::output) / sizeof(float);

   for (size_t i = 0; i < nOut_avgpoolpad; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

// Same model as AvgPoolPad but count_include_pad = 1, so the divisor is the full
// kernel area (kh*kw). The border values differ from AvgPoolPad, which is what this test pins down

TEST_F(SofieAlpakaTest, AvgPoolCountIncludePad)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(16);
   for (size_t i = 0; i < input.size(); ++i) input[i] = float(i);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(AvgPoolCountIncludePad_ExpectedOutput::output) / sizeof(float)}));

   {
      SOFIE_AvgPoolCountIncludePad::Session<alpaka::TagGpuCudaRt> session("AvgPoolCountIncludePad_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = AvgPoolCountIncludePad_ExpectedOutput::output;
   constexpr size_t nOut_avgpoolinc = sizeof(AvgPoolCountIncludePad_ExpectedOutput::output) / sizeof(float);

   for (size_t i = 0; i < nOut_avgpoolinc; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

// GlobalAveragePool: one output per channel = the mean of the whole channel.
TEST_F(SofieAlpakaTest, GlobalAvgPool2d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(18);
   for (size_t i = 0; i < input.size(); ++i) input[i] = float(i);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(GlobalAvgPool2d_ExpectedOutput::output) / sizeof(float)}));

   {
      SOFIE_GlobalAvgPool2d::Session<alpaka::TagGpuCudaRt> session("GlobalAvgPool2d_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = GlobalAvgPool2d_ExpectedOutput::output;
   constexpr size_t nOut_globalavg = sizeof(GlobalAvgPool2d_ExpectedOutput::output) / sizeof(float);

   for (size_t i = 0; i < nOut_globalavg; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}
