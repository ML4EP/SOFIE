#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/operators/ROperator_BatchNormalization.hxx"
#include "onnx.hxx"
namespace SOFIE {

ParserFuncSignature ParseBatchNormalization = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("SOFIE ONNX Parser BatchNorm op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);
   float fepsilon = 1e-05;
   float fmomentum = 0.9;
   std::size_t ftraining_mode = 0;
   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      const std::string &attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "epsilon")
         fepsilon = nodeproto.attribute(i).f();
      else if (attribute_name == "momentum")
         fmomentum = nodeproto.attribute(i).f();
      else if (attribute_name == "training_mode")
         ftraining_mode = nodeproto.attribute(i).i();
   }
   // In training mode Y is normalized with the statistics of the batch, and the node can also
   // output the updated running mean and variance (the only way to ask for training before
   // opset 14). SOFIE only implements inference, which uses the input mean and variance.
   bool training_outputs = false;
   for (int_t i = 1; i < nodeproto.output_size(); i++)
      training_outputs |= !nodeproto.output(i).empty();
   if (ftraining_mode != 0 || training_outputs) {
      throw std::runtime_error("SOFIE ONNX Parser BatchNormalization op " + nodeproto.name() +
                               " is in training mode (training_mode=1 or running mean and variance outputs),"
                               " only inference is supported");
   }

   switch (input_type) {
   case ETensorType::FLOAT:
      if (nodeproto.input_size() == 5) {
         op.reset(new ROperator_BatchNormalization<float>(fepsilon, fmomentum, ftraining_mode, nodeproto.input(0),
                                                          nodeproto.input(1), nodeproto.input(2), nodeproto.input(3),
                                                          nodeproto.input(4), output_name));
      } else {
         throw std::runtime_error("SOFIE ONNX Parser BatchNormalization op requires exactly 5 inputs, got " +
                                  std::to_string(nodeproto.input_size()));
      }
      break;
   default:
      throw std::runtime_error("SOFIE - Unsupported - Operator BatchNorm does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
