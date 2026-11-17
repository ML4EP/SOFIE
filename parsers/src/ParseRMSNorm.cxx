#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/operators/ROperator_RMSNorm.hxx"
#include "onnx.hxx"

namespace SOFIE {

ParserFuncSignature ParseRMSNorm = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
   -> std::unique_ptr<ROperator>
{
   if (nodeproto.input_size() < 2)
      throw std::runtime_error("SOFIE ParseRMSNorm: need at least 2 inputs (X, scale)");

   const std::string &xName     = nodeproto.input(0);
   const std::string &scaleName = nodeproto.input(1);
   const std::string &yName     = nodeproto.output(0);

   float   epsilon    = 1e-5f;
   int     axis       = -1;
   int64_t stash_type = 1;

   for (const auto &attr : nodeproto.attribute()) {
      if (attr.name() == "epsilon")         epsilon    = attr.f();
      else if (attr.name() == "axis")       axis       = static_cast<int>(attr.i());
      else if (attr.name() == "stash_type") stash_type = attr.i();
   }
   // the kernels accumulate in float, which is what stash_type = 1 (the ONNX default) asks for
   if (stash_type != 1)
      throw std::runtime_error("SOFIE ParseRMSNorm: only stash_type = 1 is supported, got " +
                               std::to_string(stash_type));

   ETensorType inputType = ETensorType::UNDEFINED;
   if (parser.IsRegisteredTensorType(xName))
      inputType = parser.GetTensorType(xName);
   else
      throw std::runtime_error("SOFIE ParseRMSNorm: input tensor " + xName + " type not registered");

   if (!parser.IsRegisteredTensorType(yName))
      parser.RegisterTensorType(yName, inputType);

   switch (inputType) {
   case ETensorType::FLOAT:
      return std::make_unique<ROperator_RMSNorm<float>>(axis, epsilon, xName, scaleName, yName);
   default:
      throw std::runtime_error("SOFIE ParseRMSNorm: unsupported input type");
   }
};

} // namespace SOFIE
