#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/operators/ROperator_BatchNormalization.hxx"
#include "onnx.hxx"
namespace SOFIE {

ParserFuseFuncSignature ParseFuseBatchnormRelu = [](RModelParser_ONNX &parser, const onnx::NodeProto &batchnormnode,
                                                const onnx::NodeProto &relunode) {
        ETensorType input_type;

        auto input_name = batchnormnode.input(0);
        if (parser.IsRegisteredTensorType(input_name)) {
            input_type = parser.GetTensorType(input_name);
        } else {
            throw std::runtime_error("SOFIE ONNX Parser BatchNorm op has input tensor " + input_name +
                                    " but its type is not yet registered");
        }
        
        std::unique_ptr<ROperator> op;
        std::string output_name = relunode.output(0);
        float fepsilon = 1e-05;
        float fmomentum = 0.9;
        std::size_t ftraining_mode = 0;
        for (int_t i = 0; i < batchnormnode.attribute_size(); i++) {
           const std::string &attribute_name = batchnormnode.attribute(i).name();
           if (attribute_name == "epsilon")
              fepsilon = batchnormnode.attribute(i).f();
           else if (attribute_name == "momentum")
              fmomentum = batchnormnode.attribute(i).f();
           else if (attribute_name == "training_mode")
              ftraining_mode = batchnormnode.attribute(i).i();
        }
        // same check as in ParseBatchNormalization: only inference is supported
        bool training_outputs = false;
        for (int_t i = 1; i < batchnormnode.output_size(); i++)
           training_outputs |= !batchnormnode.output(i).empty();
        if (ftraining_mode != 0 || training_outputs) {
           throw std::runtime_error("SOFIE ONNX Parser BatchNormalization op " + batchnormnode.name() +
                                    " is in training mode (training_mode=1 or running mean and variance outputs),"
                                    " only inference is supported");
        }

        switch (input_type) {
        case ETensorType::FLOAT:
            if (batchnormnode.input_size() == 5) {
                op.reset(new ROperator_BatchNormalization<float>(fepsilon, fmomentum, ftraining_mode, batchnormnode.input(0),
                                                                batchnormnode.input(1), batchnormnode.input(2), batchnormnode.input(3),
                                                                batchnormnode.input(4), output_name, EActivationType::RELU));
            } else {
                throw std::runtime_error("SOFIE ONNX Parser BatchNormalization op requires exactly 5 inputs, got " +
                                         std::to_string(batchnormnode.input_size()));
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

