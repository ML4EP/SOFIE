#pragma once

// BatchNormalization models for the parser tests (cpu/TestSofieParser.cxx), written as protobuf
// directly so they can also describe models that SOFIE must reject.

#include "common/onnx_proto_helpers.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct BatchNormModel {
   int numInputs = 5;                       // x, scale, bias, mean, var
   std::vector<std::string> extraOutputs;   // running mean and variance outputs of a training node
   int trainingMode = -1;                   // -1: no training_mode attribute
   bool withRelu = false;                   // followed by a Relu, which the parser fuses
};

// x[1,3,2,2] -> BatchNormalization (-> Relu) -> y
inline void WriteBatchNormModel(const std::string &fileName, const BatchNormModel &m)
{
   const std::vector<std::string> inputNames{"x", "scale", "bias", "mean", "var"};

   std::string bn;
   for (int i = 0; i < m.numInputs; ++i)
      AppendBytesField(bn, 1, inputNames[i]);
   AppendBytesField(bn, 2, m.withRelu ? "bn" : "y");
   for (const std::string &out : m.extraOutputs)
      AppendBytesField(bn, 2, out);
   AppendBytesField(bn, 3, "bn_0");
   AppendBytesField(bn, 4, "BatchNormalization");
   if (m.trainingMode >= 0)
      AppendBytesField(bn, 5, IntAttribute("training_mode", m.trainingMode));

   std::string graph;
   AppendBytesField(graph, 1, bn);
   if (m.withRelu) {
      std::string relu;
      AppendBytesField(relu, 1, "bn");
      AppendBytesField(relu, 2, "y");
      AppendBytesField(relu, 3, "relu_0");
      AppendBytesField(relu, 4, "Relu");
      AppendBytesField(graph, 1, relu);
   }
   AppendBytesField(graph, 2, "test_graph");
   AppendBytesField(graph, 5, FloatTensor("scale", {1.5f, 0.5f, 2.f}));
   AppendBytesField(graph, 5, FloatTensor("bias", {0.1f, -0.2f, 0.3f}));
   AppendBytesField(graph, 5, FloatTensor("mean", {0.f, 1.f, -1.f}));
   AppendBytesField(graph, 5, FloatTensor("var", {1.f, 4.f, 0.25f}));
   AppendBytesField(graph, 11, FloatValueInfo({"x", {"1", "3", "2", "2"}}));
   AppendBytesField(graph, 12, FloatValueInfo({"y", {"1", "3", "2", "2"}}));
   for (const std::string &out : m.extraOutputs)
      AppendBytesField(graph, 12, FloatValueInfo({out, {"3"}}));

   std::string opset;
   AppendVarintField(opset, 2, 15);
   std::string model;
   AppendVarintField(model, 1, 10);
   AppendBytesField(model, 7, graph);
   AppendBytesField(model, 8, opset);

   std::ofstream file(fileName, std::ios::binary);
   file.write(model.data(), model.size());
   if (!file.good())
      throw std::runtime_error("cannot write the test model " + fileName);
}
