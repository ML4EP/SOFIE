#include "TestAlpakaCommon.h"

#include "RMSNorm_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicRMSNorm_FromONNX_GPU_ALPAKA.hxx"
#include "GroupNorm_FromONNX_GPU_ALPAKA.hxx"
#include "GroupNormBig_FromONNX_GPU_ALPAKA.hxx"
#include "CumSum_FromONNX_GPU_ALPAKA.hxx"
#include "CumSumExclusive_FromONNX_GPU_ALPAKA.hxx"

#include "input_models/references/RMSNorm.ref.hxx"
#include "input_models/references/RMSNorm_input.ref.hxx"
#include "input_models/references/GroupNorm.ref.hxx"
#include "input_models/references/GroupNorm_input.ref.hxx"
#include "input_models/references/CumSum.ref.hxx"
#include "input_models/references/CumSum_input.ref.hxx"
#include "input_models/references/CumSumExclusive.ref.hxx"
#include "input_models/references/CumSumExclusive_input.ref.hxx"


TEST_F(SofieAlpakaTest, RMSNorm)
{
    constexpr float TOLERANCE = 1e-4f;

    const float* input = RMSNorm_Input::data;
    constexpr std::size_t inputSize  = 2 * 4;
    constexpr std::size_t outputSize = inputSize;

    auto input_d = makeDeviceBuf<float>(host, device, queue, input, inputSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_RMSNorm::Session<alpaka::TagGpuCudaRt> session("RMSNorm_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = RMSNorm_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(RMSNorm_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, DynamicRMSNorm)
{
    constexpr float TOLERANCE = 1e-4f;
    constexpr std::size_t D = 4;
    const float scale[D] = {1.0f, 1.0f, 1.0f, 1.0f};

    for (std::size_t N : {std::size_t(1), std::size_t(6)}) {
        std::vector<float> input(N * D);
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = (static_cast<float>(i % 9) - 4.0f) * 0.5f;

        auto input_d = makeDeviceBuf<float>(host, device, queue, input.data(), input.size());
        auto scale_d = makeDeviceBuf<float>(host, device, queue, scale, D);
        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N * D}));

        {
            SOFIE_DynamicRMSNorm::Session<alpaka::TagGpuCudaRt> session("DynamicRMSNorm_FromONNX_GPU_ALPAKA.dat", N);
            auto result = session.infer(N, input_d, scale_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        const float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n) {
            double sumSq = 0;
            for (std::size_t d = 0; d < D; ++d) sumSq += double(input[n * D + d]) * input[n * D + d];
            float invRms = static_cast<float>(1.0 / std::sqrt(sumSq / D + 1e-5));
            for (std::size_t d = 0; d < D; ++d) {
                float expected = scale[d] * input[n * D + d] * invRms;
                EXPECT_LE(std::abs(res[n * D + d] - expected), TOLERANCE) << "n=" << n << " d=" << d << " N=" << N;
            }
        }
    }
}

TEST_F(SofieAlpakaTest, RMSNormResetState)
{
    SOFIE_RMSNorm::Session<alpaka::TagGpuCudaRt> session("RMSNorm_FromONNX_GPU_ALPAKA.dat");
    // resetState must be callable; for a stateless operator it is a no-op
    EXPECT_NO_THROW(session.resetState(session.queue));
}

TEST_F(SofieAlpakaTest, GroupNorm)
{
    constexpr float TOLERANCE = 1e-4f;

    const float* input = GroupNorm_Input::data;
    constexpr std::size_t inputSize  = 1 * 4 * 2;
    constexpr std::size_t outputSize = inputSize;

    auto input_d  = makeDeviceBuf<float>(host, device, queue, input, inputSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_GroupNorm::Session<alpaka::TagGpuCudaRt> session("GroupNorm_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = GroupNorm_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(GroupNorm_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, GroupNormBig)
{
    constexpr float TOLERANCE = 1e-3f;
    constexpr std::size_t N = 2, C = 4, H = 20, W = 20, G = 2;
    constexpr std::size_t spatial = H * W, gs = C / G;
    constexpr std::size_t size = N * C * spatial;

    std::vector<float> x(size);
    for (std::size_t i = 0; i < size; ++i) x[i] = (static_cast<float>(i % 23) - 11.0f) * 0.37f;
    float scale[C] = {1.1f, 0.9f, 1.3f, 0.8f};
    float bias[C]  = {0.1f, -0.2f, 0.3f, -0.1f};

    auto x_d     = makeDeviceBuf<float>(host, device, queue, x.data(), size);
    auto scale_d = makeDeviceBuf<float>(host, device, queue, scale, C);
    auto bias_d  = makeDeviceBuf<float>(host, device, queue, bias, C);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{size}));

    {
        SOFIE_GroupNormBig::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(x_d, scale_d, bias_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    std::vector<float> expected(size);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t g = 0; g < G; ++g) {
            double sum = 0;
            for (std::size_t ci = 0; ci < gs; ++ci) {
                std::size_t c = g * gs + ci;
                for (std::size_t s = 0; s < spatial; ++s)
                    sum += x[n * C * spatial + c * spatial + s];
            }
            double mean = sum / (gs * spatial);
            double varsum = 0;
            for (std::size_t ci = 0; ci < gs; ++ci) {
                std::size_t c = g * gs + ci;
                for (std::size_t s = 0; s < spatial; ++s) {
                    double d = x[n * C * spatial + c * spatial + s] - mean;
                    varsum += d * d;
                }
            }
            double var = varsum / (gs * spatial);
            double inv_std = 1.0 / std::sqrt(var + 1e-5);
            for (std::size_t ci = 0; ci < gs; ++ci) {
                std::size_t c = g * gs + ci;
                for (std::size_t s = 0; s < spatial; ++s) {
                    std::size_t idx = n * C * spatial + c * spatial + s;
                    expected[idx] = static_cast<float>((x[idx] - mean) * inv_std) * scale[c] + bias[c];
                }
            }
        }
    }

    const float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (std::size_t i = 0; i < size; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, CumSumInclusive)
{
    constexpr float TOLERANCE = 1e-4f;

    const float* input = CumSum_Input::data;
    constexpr std::size_t inputSize  = 2 * 5;
    constexpr std::size_t outputSize = inputSize;

    auto input_d  = makeDeviceBuf<float>(host, device, queue, input, inputSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_CumSum::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = CumSum_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(CumSum_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, CumSumExclusive)
{
    constexpr float TOLERANCE = 1e-4f;

    const float* input = CumSumExclusive_Input::data;
    constexpr std::size_t inputSize  = 2 * 5;
    constexpr std::size_t outputSize = inputSize;

    auto input_d  = makeDeviceBuf<float>(host, device, queue, input, inputSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_CumSumExclusive::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = CumSumExclusive_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(CumSumExclusive_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}
