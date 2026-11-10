#ifndef SOFIE_TEST_QUANTIZATIONALPAKATESTFIXTURE
#define SOFIE_TEST_QUANTIZATIONALPAKATESTFIXTURE

// Alpaka/CUDA session fixture for the quantization tests, with the deterministic
// int8 dense-linear reference the DenseLinear sessions are checked against.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>

#include "gtest/gtest.h"

using Idx = std::size_t;
using Dim = alpaka::DimInt<1>;
using Ext1D = alpaka::Vec<Dim, Idx>;

struct QuantizedLinearTest {
   Idx m;
   Idx k;
   Idx n;
   bool matMul;
   bool hasBias;
   bool perChannelWeight;
};

inline std::int8_t QuantizedLinearTestInputValue(Idx index)
{
   return static_cast<std::int8_t>(((index * 5 + index / 7) % 31) - 15);
}

inline std::int8_t QuantizedLinearTestWeightValue(Idx index, Idx n)
{
   return static_cast<std::int8_t>(((index * 3 + index / 11 + n) % 29) - 14);
}

inline std::vector<std::int8_t> MakeQuantizedLinearTestInput(const QuantizedLinearTest &test)
{
   std::vector<std::int8_t> input(test.m * test.k);
   for (Idx i = 0; i < input.size(); ++i)
      input[i] = QuantizedLinearTestInputValue(i);
   return input;
}

inline std::vector<std::int8_t> MakeQuantizedLinearTestExpected(const QuantizedLinearTest &test,
                                                                const std::vector<std::int8_t> &input)
{
   std::vector<std::int8_t> output(test.m * test.n);
   constexpr int scaleNumerators[] = {3, 4, 5, 6};
   for (Idx row = 0; row < test.m; ++row) {
      for (Idx column = 0; column < test.n; ++column) {
         std::int32_t accumulator = 0;
         for (Idx inner = 0; inner < test.k; ++inner) {
            const Idx weightIndex = test.matMul ? inner * test.n + column : column * test.k + inner;
            accumulator += static_cast<std::int32_t>(input[row * test.k + inner]) *
                           static_cast<std::int32_t>(QuantizedLinearTestWeightValue(weightIndex, test.n));
         }
         if (test.hasBias)
            accumulator += static_cast<std::int32_t>((column * 3) % 17) - 8;
         const int scaleNumerator = test.perChannelWeight ? scaleNumerators[column % 4] : 4;
         const auto quantized = static_cast<long>(std::nearbyint(
            static_cast<double>(accumulator) * static_cast<double>(scaleNumerator) / 128.0));
         output[row * test.n + column] = static_cast<std::int8_t>(std::clamp(quantized, -128L, 127L));
      }
   }
   return output;
}

class QuantizationAlpakaTest : public ::testing::Test {
protected:
    // Shared devices and platforms
    alpaka::PlatformCpu hostPlatform;
    alpaka::DevCpu host;
    alpaka::PlatformCudaRt platform;
    alpaka::DevCudaRt device;
    alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue;

    template <typename TModel>
    void ExpectQuantizedLinearInt8Output(TModel &model, const std::vector<std::int8_t> &expectedOutput,
                                         auto &&...inputs)
    {
        const Idx outputSize = expectedOutput.size();
        auto result_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(outputSize));
        auto result = model.infer(std::forward<decltype(inputs)>(inputs)...);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);

        const auto *res_ptr = reinterpret_cast<const std::int8_t *>(alpaka::getPtrNative(result_h));
        for (Idx i = 0; i < outputSize; ++i) {
            EXPECT_EQ(static_cast<int>(res_ptr[i]), static_cast<int>(expectedOutput[i])) << "i=" << i;
        }
    }

    auto CopyQuantizedInputToDevice(const std::vector<std::int8_t> &input)
    {
        const Idx inputSize = input.size();
        auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(inputSize));
        std::int8_t *input_ptr = reinterpret_cast<std::int8_t *>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inputSize; ++i)
            input_ptr[i] = input[i];

        auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(inputSize));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);
        return input_d;
    }

    template <typename TModel>
    void RunQuantizedLinearInt8(const char *weightFile, const QuantizedLinearTest &test)
    {
        const auto input = MakeQuantizedLinearTestInput(test);
        const auto expectedOutput = MakeQuantizedLinearTestExpected(test, input);
        auto input_d = CopyQuantizedInputToDevice(input);
        TModel model(weightFile);
        ExpectQuantizedLinearInt8Output(model, expectedOutput, input_d);
    }

    // Runs the dense layer behind a float graph boundary (QONNX float I/O): feeds
    // codes * inputScale and checks the output against codes * outputScale.
    template <typename TModel>
    void RunQuantizedLinearFloat(const char *weightFile, const QuantizedLinearTest &test, float inputScale,
                                 float outputScale)
    {
        const auto codes = MakeQuantizedLinearTestInput(test);
        const auto expectedCodes = MakeQuantizedLinearTestExpected(test, codes);

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx(codes.size())));
        for (Idx i = 0; i < codes.size(); ++i)
            alpaka::getPtrNative(input_h)[i] = static_cast<float>(codes[i]) * inputScale;
        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx(codes.size())));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        TModel model(weightFile);
        auto result = model.infer(input_d);
        alpaka::wait(queue);
        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx(expectedCodes.size())));
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);

        const float *res = alpaka::getPtrNative(result_h);
        int mismatches = 0;
        for (Idx i = 0; i < expectedCodes.size() && mismatches < 5; ++i) {
            const float expected = static_cast<float>(expectedCodes[i]) * outputScale;
            if (res[i] != expected) {
                ++mismatches;
                EXPECT_EQ(res[i], expected) << "i=" << i;
            }
        }
        EXPECT_EQ(mismatches, 0);
    }


    QuantizationAlpakaTest() 
        : hostPlatform{}
        , host(alpaka::getDevByIdx(hostPlatform, 0u))
        , platform{}
        , device(alpaka::getDevByIdx(platform, 0u))
        , queue(device)
    {
    }

    void SetUp() override {
        cudaDeviceSynchronize();
    }

    void TearDown() override {
        alpaka::wait(queue);
        cudaDeviceSynchronize();
    }

    ~QuantizationAlpakaTest() override {
        cudaDeviceSynchronize();
    }
};

#endif // SOFIE_TEST_QUANTIZATIONALPAKATESTFIXTURE
