// Quantization test binary: checks GPU numerics of generated quantized sessions against
// exact host references. The emitFromONNXAlpaka fixture step generates the headers; the
// numerics tests are #included so the suite is one CUDA translation unit.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "QuantizationAlpakaTestFixture.hxx"

#include "QONNX_QuantConv_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantConv_FromONNX_GPU_ALPAKA.hxx"

#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>

#include "gtest/gtest.h"

TEST_F(QuantizationAlpakaTest, ConvolutionFrontendsEquivalent)
{
   // Checks that a QONNX and a Q/DQ 1x1 NCHW Conv match an independent host reference.
   // QONNX outputs the dequantized float grid; Q/DQ outputs the INT8 carrier.
   constexpr Idx channels = 64;
   constexpr Idx width = 256;
   constexpr double inputScale = 0.125;
   constexpr double weightScale = 0.0625;
   constexpr double outputScale = 0.25;
   constexpr Idx elements = channels * width;

   std::vector<std::int8_t> carrier(elements);
   std::vector<float> fakeQuantInput(elements);
   for (Idx channel = 0; channel < channels; ++channel) {
      for (Idx position = 0; position < width; ++position) {
         const auto value = static_cast<std::int8_t>(
            static_cast<int>((channel * 7 + position * 3) % 31) - 15);
         carrier[channel * width + position] = value;
         fakeQuantInput[channel * width + position] =
            static_cast<float>(static_cast<double>(value) * inputScale);
      }
   }

   std::vector<std::int8_t> expectedCarrier(elements);
   std::vector<float> expectedFakeQuant(elements);
   for (Idx outputChannel = 0; outputChannel < channels; ++outputChannel) {
      const Idx adjacent = (outputChannel + 1) % channels;
      for (Idx position = 0; position < width; ++position) {
         const auto accumulator =
            8 * static_cast<int>(carrier[outputChannel * width + position]) -
            2 * static_cast<int>(carrier[adjacent * width + position]);
         const auto quantized = static_cast<long>(std::nearbyint(
            static_cast<double>(accumulator) * inputScale * weightScale / outputScale));
         const auto clamped = static_cast<std::int8_t>(
            std::clamp(quantized, -128L, 127L));
         expectedCarrier[outputChannel * width + position] = clamped;
         expectedFakeQuant[outputChannel * width + position] =
            static_cast<float>(static_cast<double>(clamped) * outputScale);
      }
   }

   auto qonnxInput_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(elements));
   std::copy(fakeQuantInput.begin(), fakeQuantInput.end(),
             alpaka::getPtrNative(qonnxInput_h));
   auto qonnxInput_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(elements));
   alpaka::memcpy(queue, qonnxInput_d, qonnxInput_h);

   auto qdqInput_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(elements));
   std::copy(carrier.begin(), carrier.end(), alpaka::getPtrNative(qdqInput_h));
   auto qdqInput_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(elements));
   alpaka::memcpy(queue, qdqInput_d, qdqInput_h);
   alpaka::wait(queue);

   SOFIE_QONNX_QuantConv::Session<alpaka::TagGpuCudaRt> qonnx(
      "QONNX_QuantConv_FromONNX_GPU_ALPAKA.dat");
   SOFIE_ONNX_QDQ_QuantConv::Session<alpaka::TagGpuCudaRt> qdq(
      "ONNX_QDQ_QuantConv_FromONNX_GPU_ALPAKA.dat");
   auto qonnxOutput_d = qonnx.infer(qonnxInput_d);
   auto qdqOutput_d = qdq.infer(qdqInput_d);
   alpaka::wait(queue);
   ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

   auto qonnxOutput_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(elements));
   auto qdqOutput_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(elements));
   alpaka::memcpy(queue, qonnxOutput_h, qonnxOutput_d);
   alpaka::memcpy(queue, qdqOutput_h, qdqOutput_d);
   alpaka::wait(queue);

   const auto *qonnxActual = alpaka::getPtrNative(qonnxOutput_h);
   const auto *qdqActual = alpaka::getPtrNative(qdqOutput_h);
   for (Idx index = 0; index < elements; ++index) {
      EXPECT_FLOAT_EQ(qonnxActual[index], expectedFakeQuant[index]) << "index=" << index;
      EXPECT_EQ(static_cast<int>(qdqActual[index]),
                static_cast<int>(expectedCarrier[index])) << "index=" << index;
   }
}

#include "TestQuantizationNumerics.cxx"
