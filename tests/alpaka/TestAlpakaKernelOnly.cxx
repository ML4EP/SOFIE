// Exercises Options::kKernelOnly: the generated header contains just the
// kernel struct and the sofie_workdiv helper without any session impl,
// so the kernel is launched directly.
// With Options::kStridedInput the header also defines sofie_strided_layout and sofie_strided_offset, and the
// kernels reading a graph input take its layout (logical shape and strides) as an argument.

#include "TestAlpakaCommon.h"
#include "common/strided_input_test_utils.h"
#include "LeakyRelu_KernelOnly_GPU_ALPAKA.hxx"
#include "KernelOnlyStridedChain_KernelOnly_GPU_ALPAKA.hxx"
#include "KernelOnlyStridedBinary_KernelOnly_GPU_ALPAKA.hxx"

using Acc = alpaka::TagToAcc<alpaka::TagGpuCudaRt, Dim, Idx>;

TEST_F(SofieAlpakaTest, KernelOnlyLeakyRelu)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    constexpr float alpha = 0.1f;
    std::vector<float> input({1.0f, -2.0f, 3.0f, -0.5f, 0.0f, -4.0f});
    const size_t n = input.size();

    auto input_d = makeDeviceBuf<float>(host, device, queue, input.data(), n);
    auto output_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{n}));

    auto workDiv = sofie_workdiv<Dim, Idx>(Ext1D::all(Idx{n}));
    auto task = alpaka::createTaskKernel<Acc>(workDiv, SOFIE_LeakyReluKernelOnly::LeakyReluKernel{},
                                               alpaka::getPtrNative(input_d), alpaka::getPtrNative(output_d),
                                               n, alpha);
    alpaka::enqueue(queue, task);
    alpaka::wait(queue);
    cudaDeviceSynchronize();

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{n}));
    alpaka::memcpy(queue, result_h, output_d);
    alpaka::wait(queue);

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < n; ++i) {
        float expected = input[i] >= 0.0f ? input[i] : alpha * input[i];
        EXPECT_LE(std::abs(res_ptr[i] - expected), TOLERANCE) << "i=" << i;
    }
}

// Neg reads the 3x5 input through its strides (the helpers are generated with the kernels), Sigmoid reads the
// contiguous output of Neg
TEST_F(SofieAlpakaTest, KernelOnlyStridedChain)
{
    // row padding, transposed and contiguous views
    for (const auto &strides : std::vector<std::vector<size_t>>{{8, 1}, {1, 3}, {5, 1}}) {
        StridedBuffer<> in({3, 5}, strides);
        constexpr size_t n = 15;
        auto in_d = makeDeviceBuf<float>(host, device, queue, in.storage.data(), in.storage.size());
        auto neg_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{n}));
        auto out_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{n}));

        sofie_strided_layout<2> layout{{3, 5}, {strides[0], strides[1]}};
        auto workDiv = sofie_workdiv<Dim, Idx>(Ext1D::all(Idx{n}));
        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, SOFIE_KernelOnlyStridedChain::UnaryNegStridedKernel{},
                                                              alpaka::getPtrNative(in_d), alpaka::getPtrNative(neg_d),
                                                              n, layout));
        alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, SOFIE_KernelOnlyStridedChain::SigmoidKernel{},
                                                              alpaka::getPtrNative(neg_d), alpaka::getPtrNative(out_d), n));
        auto out = toHost(host, queue, out_d, n);

        for (size_t i = 0; i < n; ++i)
            EXPECT_NEAR(out[i], 1.f / (1.f + std::exp(in.logical[i])), 1e-5f) << "i=" << i;
    }
}

// A - B with both inputs strided; B (3x1) is broadcast over the first and last dimensions of A (2x3x4): its
// layout has the shape of the output and a zero stride for the broadcast dimensions
TEST_F(SofieAlpakaTest, KernelOnlyStridedBinary)
{
    StridedBuffer<> a({2, 3, 4}, {1, 8, 2});
    StridedBuffer<> b({3, 1}, {2, 7}, [](size_t k) { return 1.5f * float(k) + 0.25f; });
    constexpr size_t n = 24;
    auto a_d = makeDeviceBuf<float>(host, device, queue, a.storage.data(), a.storage.size());
    auto b_d = makeDeviceBuf<float>(host, device, queue, b.storage.data(), b.storage.size());
    auto out_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{n}));

    sofie_strided_layout<3> layoutA{{2, 3, 4}, {1, 8, 2}};
    sofie_strided_layout<3> layoutB{{2, 3, 4}, {0, 2, 0}};
    auto workDiv = sofie_workdiv<Dim, Idx>(Ext1D::all(Idx{n}));
    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv, SOFIE_KernelOnlyStridedBinary::Binary0SubKernel{},
                                                          alpaka::getPtrNative(a_d), alpaka::getPtrNative(b_d),
                                                          alpaka::getPtrNative(out_d), layoutA, layoutB, n));
    auto out = toHost(host, queue, out_d, n);

    for (size_t k = 0; k < n; ++k)
        EXPECT_FLOAT_EQ(out[k], a.logical[k] - b.logical[(k / 4) % 3]) << "k=" << k;
}

// the offset function of the generated header: element idx (row-major logical index) of a view with these strides
TEST(KernelOnlyStridedOffset, MatchesLayout)
{
    sofie_strided_layout<3> layout{{2, 3, 4}, {1, 8, 2}};
    for (size_t idx = 0; idx < 24; ++idx) {
        size_t i = idx / 12, j = (idx / 4) % 3, l = idx % 4;
        EXPECT_EQ(sofie_strided_offset(layout, idx), i * 1 + j * 8 + l * 2) << "idx=" << idx;
    }
}
