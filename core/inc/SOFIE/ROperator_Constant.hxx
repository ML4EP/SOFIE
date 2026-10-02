#ifndef SOFIE_ROPERATOR_Constant
#define SOFIE_ROPERATOR_Constant

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <iomanip>
#include <limits>

namespace SOFIE{

template<typename T>
class ROperator_Constant final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::vector<Dim> fDimShape;  // used for dynamic ConstantOfShape
   std::vector<T> fValues;
   std::string fAttrType;
   bool fIsConstantOfShape = false;
   bool fRuntimeShape = false;
   std::vector<std::string> fRuntimeDims;

   std::string ValueLiteral(T v) const {
      std::ostringstream s;
      if (fAttrType == "float" || fAttrType == "double") {
         s << std::setprecision(fAttrType == "float" ? std::numeric_limits<float>::max_digits10
                                                       : std::numeric_limits<double>::max_digits10)
           << v;
         std::string str = s.str();
         if (str.find_first_of(".eE") == std::string::npos)
            str += ".0";
         if (fAttrType == "float") str += "f";
         return str;
      }
      if (fAttrType == "bool")
         s << (v ? "true" : "false");
      else
         s << v;
      return s.str();
   }

public:
   ROperator_Constant(){}

   ROperator_Constant(const std::string & type, const std::vector<T> & values, const std::vector<size_t> & shape, std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)),
      fNY(UTILITY::Clean_name(nameY)),
      fShape(shape),
      fValues(values),
      fAttrType(type)
      {
         fInputTensorNames = fNX.empty() ? std::vector<std::string>{} : std::vector<std::string>{fNX};
         fOutputTensorNames = {fNY};
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      size_t length = 1;
      if (!fNX.empty()) {
         // case of ConstantOfShape (since no inputs in case of Constant operator)
         fIsConstantOfShape  = true;
         if (model.CheckIfTensorAlreadyExist(fNX) == false){
           throw std::runtime_error("SOFIE ConstantOfShape Op Input Tensor is not found in model");
         }
         if (model.IsShapeTensor(fNX)) {
            // Input is a shape tensor (symbolic dimensions) — output will be a dynamic tensor
            // whose shape is determined at runtime from the symbolic values.
            const auto & dimVals = model.GetShapeTensorValues(fNX);
            std::vector<Dim> outShape;
            for (const auto & d : dimVals)
               outShape.push_back(d);
            if (fValues.size() != 1)
               throw std::runtime_error("SOFIE ConstantOfShape Op value Tensor has invalid size " + std::to_string(fValues.size()));
            // Register as a dynamic intermediate tensor — values will be filled at runtime.
            model.AddIntermediateTensor(fNY, GetTemplatedType(T()), outShape);
            // Store shape for code generation (use fShape for rank, values = 0 for symbolic dims)
            fShape.resize(outShape.size());
            for (size_t i = 0; i < outShape.size(); i++)
               fShape[i] = outShape[i].isParam ? 0 : outShape[i].dim;
            // Store symbolic lengths/shape for Generate()
            fDimShape = outShape;
            fIsOutputConstant = false;  // cannot be constant since shape is dynamic
            return;
         }

         if (!model.IsInitializedTensor(fNX)) {
            auto inputShape = model.GetTensorShape(fNX);
            if (inputShape.size() != 1)
               throw std::runtime_error("SOFIE ConstantOfShape Op Input Tensor must have rank 1");

            size_t rank = ConvertShapeToLength(inputShape);
            fRuntimeShape = true;
            fRuntimeDims.resize(rank);
            fDimShape.resize(rank);

            for (size_t i = 0; i < rank; ++i) {
               fRuntimeDims[i] = fNY + "_dim_" + std::to_string(i);
               fDimShape[i] = Dim{fRuntimeDims[i], size_t(-1)};
               model.RegisterInternalDynamicParam(fRuntimeDims[i]);
            }

            if (fValues.size() != 1)
               throw std::runtime_error("SOFIE ConstantOfShape Op value Tensor has invalid size " + std::to_string(fValues.size()));

            model.AddDynamicTensor(fNY, GetTemplatedType(T{}), fDimShape);
            fIsOutputConstant = false;
            return;
         }

         // get output shape from input values:
         // can work only if input is a constant or initialized tensor
         auto dptr = model.GetInitializedTensorData(fNX);
         auto input_tensor = static_cast<int64_t *>(dptr.get());
         auto input_shape = model.GetTensorShape(fNX);
         if (input_shape.size() > 1 )
            throw std::runtime_error("SOFIE ConstantOfShape Op Input Tensor has invalid shape");
         if (input_tensor != nullptr && !input_shape.empty()) {
            fShape = std::vector<size_t> (input_shape[0]);
            for (size_t i = 0; i < fShape.size(); i++)
               fShape[i] = input_tensor[i];
         } else
            fShape = {1};  // scalar case

         length = ConvertShapeToLength(fShape);
         if (fValues.size() != 1)
            throw std::runtime_error("SOFIE ConstantOfShape Op value Tensor has invalid size " + std::to_string(fValues.size()));

         T value = fValues[0];
         fValues = std::vector<T>(length, value);

      } else {
         // case of constant operator
         // in case of standard constant the shape is provided as input
         length = ConvertShapeToLength(fShape);
         if (length != fValues.size())
            throw std::runtime_error("SOFIE Constant Op has invalid shape : " + ConvertShapeToString(fShape) +
                                 " with " + std::to_string(fValues.size()) + " values");
      }

      // we need to create an initialized tensor of type constant to flag to not save it in a weight file
      // but keep its initialization in the generated code. The values might also be needed in initializing the
      // following operators using as input Constant or ConstantOfShape
       // resize fValues to shape length
      model.AddConstantTensor(fNY, fShape, fValues);
      if (model.Verbose()) {
         std::cout << "adding constant tensor " << fNY << " with shape " << ConvertShapeToString(fShape)
         << " and values [";
         for (auto v : fValues) std::cout << " " << v;
         std::cout << "]" << std::endl;
      }
   }

   std::string Generate(std::string /* OpName */) override {
      if (!fIsConstantOfShape || fIsOutputConstant)
         return "//---------------------------------------\n";
      std::stringstream out;
      std::string length = ConvertDimShapeToLength(fDimShape);
      out << "\n//------ CONSTANTOFSHAPE (dynamic)\n";
      out << SP << "for (size_t i = 0; i < " << length << "; i++)\n";
      out << SP << SP << "tensor_" << fNY << "[i] = " << ValueLiteral(fValues[0]) << ";\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      if (!fIsConstantOfShape || fIsOutputConstant) return "";
      opName = "op_" + opName;
      std::string kname = "ConstantOfShapeFillKernel_" + opName;
      std::string op;
      op  = "\n//------ CONSTANTOFSHAPE_FILL_KERNEL_ALPAKA (dynamic)\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, T* __restrict__ out, T const value, std::size_t const totalElements) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "auto const stride = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "for (std::size_t i = idx; i < totalElements; i += stride) out[i] = value;\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      if (!fIsConstantOfShape || fIsOutputConstant) return "";
      opName = "op_" + opName;
      return SP + "ConstantOfShapeFillKernel_" + opName + " constantOfShapeFillKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      if (!fIsConstantOfShape || fIsOutputConstant)
         return "//---------------------------------------\n";
      opName = "op_" + opName;

      std::stringstream out;
      out << "\n//------ CONSTANTOFSHAPE_GPU_ALPAKA (dynamic)\n";

      if (fRuntimeShape) {
         out << SP << "auto shapeHost_" << opName << " = alpaka::allocBuf<int64_t, Idx>(hostAcc, Ext1D::all(Idx{"
             << fRuntimeDims.size() << "}));\n";
         out << SP << "alpaka::memcpy(queue, shapeHost_" << opName << ", deviceBuf_" << fNX << ");\n";
         out << SP << "alpaka::wait(queue);\n";
         out << SP << "auto* shapePtr_" << opName << " = alpaka::getPtrNative(shapeHost_" << opName << ");\n";
         for (size_t i = 0; i < fRuntimeDims.size(); i++)
            out << SP << "size_t " << fRuntimeDims[i] << " = static_cast<size_t>(shapePtr_" << opName << "[" << i << "]);\n";
         out << SP << "deviceBuf_" << fNY << " = alpaka::allocBuf<" << ConvertTypeToString(GetTemplatedType(T()))
             << ", Idx>(devAcc, Ext1D::all(Idx{static_cast<Idx>(" << ConvertDimShapeToLength(fDimShape) << ")}));\n";
      }

      std::string length = ConvertDimShapeToLength(fDimShape);
      out << SP << "auto const elementsPerGrid_" << opName << " = Vec::all(Idx{static_cast<Idx>(" << length << ")});\n";
      out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", constantOfShapeFillKernel_" << opName
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", static_cast<" << ConvertTypeToString(GetTemplatedType(T())) << ">(" << ValueLiteral(fValues[0]) << ")"
          << ", static_cast<std::size_t>(" << length << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }
};

}//SOFIE


#endif //SOFIE_ROPERATOR_Constant
