#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_Swish.hxx"
#include "onnx.hxx"
namespace SOFIE {

ParserFuncSignature ParseSwish = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("SOFIE ONNX Parser Swish op has input tensor" + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;

   float attr_alpha = 1;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "alpha")
         attr_alpha = nodeproto.attribute(i).f();
   }

   if (attr_alpha != 1.0) {
      throw std::runtime_error("SOFIE - Unsupported - Operator Swish does not yet support alpha != 1");
   }

   std::string output_name = nodeproto.output(0);

   op.reset(new ROperator_Swish(input_name, output_name));

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

}
