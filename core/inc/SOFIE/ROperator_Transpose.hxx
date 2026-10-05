#ifndef SOFIE_ROPERATOR_TRANSPOSE
#define SOFIE_ROPERATOR_TRANSPOSE

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <cassert>


namespace SOFIE{



class ROperator_Transpose final : public ROperator
{

private:

   std::vector<int64_t> fAttrPerm;

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;
   bool fDynamicOutput = false;
   ETensorType fType = ETensorType::FLOAT;

public:

   ROperator_Transpose(){}
   ROperator_Transpose(std::vector<int64_t> attr_perm, std::string nameData, std::string nameOutput):
      fAttrPerm(attr_perm), fNX(UTILITY::Clean_name(nameData)), fNY(UTILITY::Clean_name(nameOutput)) {
            fInputTensorNames = { fNX };
            fOutputTensorNames = { fNY };
   }


   template<class T>
   void ProcessInitializedTensor(RModel& model) {
      auto shapeX = ConvertShapeToInt(fShapeX);
      auto shapeY = ConvertShapeToInt(fShapeY);
      fIsOutputConstant = true;

      auto inStrides = UTILITY::ComputeStrideFromShape(shapeX);
      auto outStrides = UTILITY::ComputeStrideFromShape(shapeY);
      size_t length = ConvertShapeToLength(shapeY);
      auto inputData = static_cast<T *>(model.GetInitializedTensorData(fNX).get());
      size_t dim = fShapeX.size();
      std::vector<size_t> outputIdx(dim);
      std::vector<T> outputData(length);
      for (size_t i = 0; i < length; i++) {
         outputIdx[0] = i / outStrides[0];
         for (size_t j = 1; j < dim; j++) {
            outputIdx[j] = (i % outStrides[j - 1]) / outStrides[j];
         }
         size_t inputIndex = 0;
         for (size_t j = 0; j < dim; j++) {
            int k = std::find(fAttrPerm.begin(), fAttrPerm.end(), j) - fAttrPerm.begin();
            inputIndex += outputIdx[k] * inStrides[j];
         }
         outputData[i] = inputData[inputIndex];
      }
      model.AddConstantTensor<T>(fNY, shapeY, outputData.data());
      if (model.Verbose()) {
         std::cout << "Transpose: output is a constant tensor " << ConvertShapeToString(shapeY) << " : "
                   << ConvertValuesToString(outputData) << std::endl;
      }
   }

   bool SupportsStridedInput() const override { return true; }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
         std::cout<<"Input tensor for transpose: "<<fNX<<'\n';
         throw std::runtime_error("SOFIE Tranpose Op Input Tensor is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShapeX.empty();
      if (fAttrPerm.empty()){
         fAttrPerm.reserve(fShapeX.size());
         for (int i = fShapeX.size() - 1; i >= 0; i--){
            fAttrPerm.push_back(i);
         }
      }

      if (fAttrPerm.size() != fShapeX.size() )
         throw std::runtime_error("SOFIE Tranpose Op - Invalid axes attributes");

      fShapeY.resize(fAttrPerm.size());
      for (size_t i = 0; i < fAttrPerm.size(); i++){
         fShapeY[i] = fShapeX[fAttrPerm[i]];
      }

      if (model.IsInitializedTensor(fNX) ) {
         auto type = model.GetTensorType(fNX);
         switch(type) {
            case ETensorType::FLOAT:
               ProcessInitializedTensor<float>(model);
               break;
            case ETensorType::INT64:
               ProcessInitializedTensor<int64_t>(model);
               break;
            case ETensorType::BOOL:
               ProcessInitializedTensor<uint8_t>(model);
               break;
            case ETensorType::UINT8:
               ProcessInitializedTensor<uint8_t>(model);
               break;
            default:
               std::cout << "Transpose - no support for initialized tensor of type " << ConvertTypeToString(type) << std::endl;
         }
         return;
      }
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      fDynamicOutput = model.IsDynamicTensor(fNY);
      fType = model.GetTensorType(fNX);
      if (model.Verbose()) {
         std::cout << "Transpose ---> " << fNY << " " <<  ConvertDimShapeToString(fShapeY) << std::endl;
      }
   }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) return "";  //no op for constant tensors
      opName = "op_" + opName;
      if (fShapeX.empty() || fShapeY.empty()){
         throw std::runtime_error("SOFIE Transpose Op called to Generate without being initialized first");
      }
      auto stridesX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto stridesY = UTILITY::ComputeStrideFromShape(fShapeY);

      auto intShapeX = ConvertShapeToInt(fShapeX);
      size_t rank = fShapeX.size();
      bool isDynamic = (intShapeX.empty() && rank > 0);

      std::string constQualifier = (isDynamic) ? "const" : "constexpr";

      std::stringstream out;

      out << SP << "///------- Transpose operator " << opName << ConvertDimShapeToString(fShapeX)
                  << " --> " << ConvertDimShapeToString(fShapeY) << std::endl;


      out << SP << "{\n";
      if (fHasStridedInput) {
         // input strides given to the Session
         out << GenerateInputStrideArray(opName + "_strX", fNX, fShapeX);
      } else {
         out << SP << SP << "// Pre-baked input strides (row-major)\n";
         out << SP << SP << constQualifier << " size_t " << opName << "_strX[] = {";
         for (size_t i = 0; i < rank; ++i)
            out << stridesX[i] << (i + 1 < rank ? ", " : "");
         out << "};\n";
      }

      out << SP << SP << "// Pre-baked output strides (row-major)\n";
      out << SP << SP << constQualifier << " size_t " << opName << "_strY[] = {";
      for (size_t i = 0; i < rank; ++i)
         out << stridesY[i] << (i + 1 < rank ? ", " : "");
      out << "};\n\n";

      // the inner stride of a strided input is only known at run time: no contiguous copy of the inner block
      bool innerContiguous = !fHasStridedInput && (fAttrPerm.back() == (int64_t) (rank - 1));
      size_t outerRank    = innerContiguous ? rank - 1 : rank;
      size_t  innerSize    = innerContiguous ? (isDynamic ? 0 : intShapeX[fAttrPerm[rank - 1]])
                             : 1;

      if (innerContiguous && !isDynamic && innerSize > 1) {
         out << SP << SP
             << "// Fast path: last permuted axis is contiguous in source\n";
         out << SP << SP
             << "// Inner " << innerSize << " elements copied with pointer arithmetic\n";

         EmitNestedLoops(out, outerRank, fShapeY);

         out << SP << SP << SP << "size_t src_off = ";
         for (size_t i = 0; i < outerRank; ++i) {
            out << "idx_" << i << " * " << opName << "_strX["
                << fAttrPerm[i] << "]";
            if (i + 1 < outerRank) out << " + ";
         }
         out << ";\n";

         out << SP << SP << SP << "size_t dst_off = ";
         for (size_t i = 0; i < outerRank; ++i) {
            out << "idx_" << i << " * " << opName << "_strY[" << i << "]";
            if (i + 1 < outerRank) out << " + ";
         }
         out << ";\n";

         out << SP << SP << SP
             << "std::copy(tensor_" << fNX << " + src_off, "
             << "tensor_" << fNX << " + src_off + " << innerSize << ", "
             << "tensor_" << fNY << " + dst_off);\n";

         CloseNestedLoops(out, outerRank);

      } else {

         out << SP << SP << "// General N-D transpose\n";

         EmitNestedLoops(out, rank, fShapeY);

         out << SP << SP << SP << "size_t src_idx = ";
         for (size_t i = 0; i < rank; ++i) {
            out << "idx_" << i << " * " << opName << "_strX[" << fAttrPerm[i] << "]";
            if (i + 1 < rank) out << " + ";
         }
         out << ";\n";

         out << SP << SP << SP << "size_t dst_idx = ";
         for (size_t i = 0; i < rank; ++i) {
            out << "idx_" << i << " * " << opName << "_strY[" << i << "]";
            if (i + 1 < rank) out << " + ";
         }
         out << ";\n";

         out << SP << SP << SP
             << "tensor_" << fNY << "[dst_idx] = "
             << "tensor_" << fNX << "[src_idx];\n";

         CloseNestedLoops(out, rank);

      }

      out << SP << "}\n";
      return out.str();
   }


   std::string Generate_GPU_Kernel_ALPAKA(std::string OpName, const std::vector<std::string> &dynParamNames) override {
      if (fIsOutputConstant) return "";
      std::string op;
      OpName = "op_" + OpName;

      const auto &dimShapeData = fShapeX;
      const auto &dimShapeOutput = fShapeY;
      const size_t rank = dimShapeData.size();

      op = "\n//------ TRANSPOSE_KERNEL_ALPAKA\n";
      op += SP + "struct TransposeKernel_" + OpName + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output,";
      for (auto &p : dynParamNames)
         op += "const std::size_t " + p + ",";
      if (fHasStridedInput)
         op += "sofie_strided_layout<" + std::to_string(rank) + "> const layoutX,";
      op += "const std::size_t totalElements) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= totalElements) return;\n";
      op += SP + SP + SP + "std::size_t input_idx = 0;\n";
      op += SP + SP + SP + "std::size_t remaining = idx;\n";
      op += SP + SP + SP + "std::size_t coord;\n";

      auto inputStrides  = UTILITY::ComputeStrideFromShape(dimShapeData);
      auto outputStrides = UTILITY::ComputeStrideFromShape(dimShapeOutput);

      for (size_t k = 0; k < rank; k++) {
         op += SP + SP + SP + SP + "coord = remaining / ("
               + outputStrides[k].GetVal() + ");\n";
         op += SP + SP + SP + SP + "remaining = remaining - coord * ("
               + outputStrides[k].GetVal() + ");\n";
         op += SP + SP + SP + SP + "input_idx += coord * ("
               + (fHasStridedInput ? "layoutX.stride[" + std::to_string(fAttrPerm[k]) + "]"
                                   : inputStrides[fAttrPerm[k]].GetVal()) + ");\n";
      }

      op += SP + SP + SP + "output[idx] = input[input_idx];\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string OpName) override {
      if (fIsOutputConstant) return "";
      return SP + "TransposeKernel_op_" + OpName + " transposeKernel_" + OpName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName, const std::vector<std::string> &dynParamNames) override {
      if (fIsOutputConstant) return "";
      const auto &dimShapeData = fShapeX;
      const auto &dimShapeOutput = fShapeY;
      if (dimShapeOutput.empty())
         throw std::runtime_error("SOFIE Operator Transpose called to Generate without being initialized first");

      std::string length = ConvertDimShapeToLength(dimShapeOutput);
      std::string inputBuffer = "deviceBuf_" + fNX;
      std::string outputBuffer = "deviceBuf_" + fNY;

      std::stringstream out;
      out << "\n//------ TRANSPOSE_GPU_ALPAKA\n";

      if (fDynamicOutput && !IsOutputPooled(fNY)) {
         out << SP << outputBuffer << " = alpaka::allocBuf<"
             << ConvertTypeToString(fType) << ", Idx>(devAcc, Ext1D::all(Idx{"
             << "static_cast<Idx>(" << length << ")}));\n";
      }

      if (fHasStridedInput)
         out << GenerateStridedBroadcastLayout("op_" + OpName + "_X", fNX, fShapeX, fShapeX.size(), fShapeX);

      out << SP << "auto const elementsPerThread_" << fNY << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_" << fNY << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
         << ", transposeKernel_" << OpName << ", alpaka::getPtrNative(" << inputBuffer
         << "), alpaka::getPtrNative(" << outputBuffer << ")";
      for (auto &p : dynParamNames)
         out << ", static_cast<std::size_t>(" << p << ")";
      if (fHasStridedInput)
         out << ", layout_op_" << OpName << "_X";
      out << ", static_cast<Idx>(" << length << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }

   EFusionMappingType GetFusionMappingType() const override
   {
      if (fIsOutputConstant || fHasStridedInput || fAttrPerm.empty())
         return EFusionMappingType::Unsupported;

      return EFusionMappingType::Shuffle;
   }

   bool SupportsFusionTypes(const std::vector<ETensorType> &inputTypes, ETensorType outputType) const override
   {
      return inputTypes.size() == 1 && inputTypes[0] == outputType;
   }

   std::string GetFusionExpr(const std::vector<std::string> &inputs) const override
   {
      if (GetFusionMappingType() != EFusionMappingType::Shuffle || inputs.size() != 1)
         return "";

      return inputs[0];
   }

   std::string GetFusionInputIndexExpr(size_t inputIndex, const std::string &outputIndex,
                                    const std::vector<Dim> &inputShape,
                                    const std::vector<Dim> &outputShape) const override
   {
      if (inputIndex != 0 || GetFusionMappingType() != EFusionMappingType::Shuffle)
         return "";

      if (inputShape.size() != outputShape.size() || fAttrPerm.size() != outputShape.size())
         return "";

      const auto inputStrides = UTILITY::ComputeStrideFromShape(inputShape);
      const auto outputStrides = UTILITY::ComputeStrideFromShape(outputShape);

      std::string expression;

      for (size_t outputAxis = 0; outputAxis < outputShape.size(); ++outputAxis) {
         const auto inputAxisValue = fAttrPerm[outputAxis];

         if (inputAxisValue < 0 || static_cast<size_t>(inputAxisValue) >= inputShape.size())
            return "";

         const size_t inputAxis = static_cast<size_t>(inputAxisValue);

         if (!expression.empty())
            expression += " + ";

         expression += "(((" + outputIndex + ") / " + outputStrides[outputAxis].GetVal() + ") % " +
              outputShape[outputAxis].GetVal() + ") * " +
              inputStrides[inputAxis].GetVal();
      }

      return "(" + expression + ")";
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_TRANSPOSE
