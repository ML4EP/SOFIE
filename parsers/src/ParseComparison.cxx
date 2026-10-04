#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_Comparison.hxx"
#include "onnx.hxx"
namespace SOFIE {

template <EComparisonOperator Op>
std::unique_ptr<ROperator> ParseComparison(RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
{
   ETensorType input_type = ETensorType::UNDEFINED;

   for (int i = 0; i < 2; ++i) {
      auto input_name = nodeproto.input(i);
      if (parser.IsRegisteredTensorType(input_name)) {
         // according to ONNX both inputs have same type
         if (i == 0)
            input_type = parser.GetTensorType(input_name);
         else
            if (input_type != parser.GetTensorType(input_name)) {
               throw
                  std::runtime_error("SOFIE ONNX parser Comparison op has input tensors of different types");
            }
      } else {
         throw std::runtime_error("SOFIE ONNX Parser Comparison op has input tensor " + input_name +
                                  " but its type is not yet registered");
      }
   }


   std::string output_name = nodeproto.output(0);

   std::unique_ptr<ROperator> op;
   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Comparison<float, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::INT64:
      op.reset(new ROperator_Comparison<int64_t, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::INT32:
      op.reset(new ROperator_Comparison<int32_t, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   default:
      throw std::runtime_error("SOFIE - Unsupported - Comparison Operator does not yet support input type " +
                               ConvertTypeToString(input_type));
   }

   // Infer the output type
   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, ETensorType::BOOL);
   }

   return op;
};

void RegisterComparisonParsers(RModelParser_ONNX &parser)
{
   parser.RegisterOperator("Equal", ParseComparison<EComparisonOperator::Eq>);
   parser.RegisterOperator("Less", ParseComparison<EComparisonOperator::Less>);
   parser.RegisterOperator("LessOrEqual", ParseComparison<EComparisonOperator::LessEq>);
   parser.RegisterOperator("Greater", ParseComparison<EComparisonOperator::Greater>);
   parser.RegisterOperator("GreaterOrEqual", ParseComparison<EComparisonOperator::GreaterEq>);
}

} // namespace SOFIE
