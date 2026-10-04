#ifndef SOFIE_ROPERATOR_Tile
#define SOFIE_ROPERATOR_Tile

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename T>
class ROperator_Tile final : public ROperator
{

private:

   std::string fNRepeats;
   std::string fNInput;
   std::string fNY;
   std::vector<Dim> fShapeInput;
   std::vector<Dim> fShapeY;
   std::string fType;
   bool fHasDynamicTiledAxis = false;  // true if any output dim is a "(param * repeat)" expression

public:
   ROperator_Tile(){}
   ROperator_Tile(std::string nameRepeat, std::string nameInput, std::string nameY):
      fNRepeats(UTILITY::Clean_name(nameRepeat)),fNInput(UTILITY::Clean_name(nameInput)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNInput };
         fOutputTensorNames = { fNY };
      }

   std::vector<Dim> DoShapeInference(const std::vector<Dim> & input, const std::vector<size_t> repeat)  {
      std::vector<Dim> ret = input;
      for(size_t i=0; i < repeat.size(); i++) {
         if (repeat[i] != 1) {
            if (ret[i].isParam) {
               ret[i] = Dim{ std::string("(" + ret[i].GetVal() + ")*" + std::to_string(repeat[i])), static_cast<size_t>(-1) };
               fHasDynamicTiledAxis = true;
            } else {
               ret[i]=Dim { ret[i].dim *repeat[i] };
            }
         }
      }
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNInput) == false){
        throw std::runtime_error("SOFIE Tile Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNRepeats) == false){
        throw std::runtime_error("SOFIE Tile Op Input Tensor is not found in model");
      }
      fShapeInput=model.GetDimTensorShape(fNInput);

      // if repeats vector is not initialized we cannot deduce shape of output
      // not support for time being this case
      if (!model.IsInitializedTensor(fNRepeats))
         throw std::runtime_error("SOFIE Tile Op: non-initialized repeats input is not supported");

      auto repptr       = model.GetInitializedTensorData(fNRepeats);
      auto repeats_data = static_cast<int64_t*>(repptr.get());
      if (repeats_data == nullptr)
         throw std::runtime_error("SOFIE Tile Op: failed to retrieve repeats tensor data");

      auto repeats_shape = model.GetTensorShape(fNRepeats);
      if (repeats_shape.size() != 1)
         throw std::runtime_error("SOFIE Tile Op: repeats tensor must be 1D");

      size_t num_elements = repeats_shape[0];
      std::vector<size_t> repeats_vector(num_elements);
      std::copy(repeats_data, repeats_data + num_elements, repeats_vector.begin());

      fShapeY = DoShapeInference(fShapeInput,repeats_vector);
      fType = ConvertTypeToString(model.GetTensorType(fNInput));

      model.SetNotWritableInitializedTensor(fNRepeats);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInput), fShapeY);

      if (model.Verbose())
         std::cout <<  "Tile: " << fNInput << " " << ConvertDimShapeToString(fShapeInput) << " -> " << fNY << " with shape " << ConvertDimShapeToString(fShapeY)
            << " given repeats " << ConvertShapeToString(repeats_vector) << std::endl;
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeY.empty()) {
            throw std::runtime_error("SOFIE Tile Op called to Generate without being initialized first");
      }

      std::stringstream out;
      out << "///-------- Tile operator " << OpName << "\n";
      out << "{\n";

      const int rank = fShapeInput.size();

      out << SP << "const size_t input_shape[" << rank << "] = " << ConvertDimShapeToString(fShapeInput) << ";\n";
      out << SP << "const size_t output_shape[" << rank << "] = " << ConvertDimShapeToString(fShapeY) << ";\n\n";

      out << SP << "size_t input_strides[" << rank << "];\n";
      out << SP << "input_strides[" << rank - 1 << "] = 1;\n";
      out << SP << "for (int i = " << rank - 2 << "; i >= 0; --i) {\n";
      out << SP << SP << "input_strides[i] = input_strides[i+1] * input_shape[i+1];\n";
      out << SP << "}\n\n";

      out << SP << "size_t out_idx = 0;\n";
      std::string indent = SP;
      for (int i = 0; i < rank; ++i) {
         out << indent << "for (size_t o" << i << " = 0, ic" << i << " = 0; o" << i
             << " < output_shape[" << i << "]; ++o" << i << ") {\n";
         indent += SP;
         out << indent << "const size_t in_off" << i << " = "
             << (i == 0 ? std::string() : "in_off" + std::to_string(i - 1) + " + ")
             << "ic" << i << " * input_strides[" << i << "];\n";
      }
      out << indent << "tensor_" << fNY << "[out_idx++] = tensor_" << fNInput << "[in_off" << rank - 1 << "];\n";
      for (int i = rank - 1; i >= 0; --i) {
         out << indent << "if (++ic" << i << " == input_shape[" << i << "]) ic" << i << " = 0;\n";
         indent.resize(indent.size() - SP.size());
         out << indent << "}\n";
      }

      out << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Operator Tile called to Generate without being initialized first");

      const std::size_t D = fShapeInput.size();

      auto inputStrides  = UTILITY::ComputeStrideFromShape(fShapeInput);
      auto outputStrides = UTILITY::ComputeStrideFromShape(fShapeY);

      std::string kname = "TileKernel_" + opName;

      std::string op;
      op  = "\n//------ TILE_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ input,\n";
      op += SP + SP + SP + "T* __restrict__ output,\n";
      for (auto &p : dynParamNames)
         op += SP + SP + SP + "std::size_t const " + p + ",\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      EmitOutputCoordsFromThreadIdx(op, SP + SP + SP + SP, outputStrides, fShapeY);
      op += "\n";

      // Input index: tiling wraps each output coordinate back into the input shape
      op += SP + SP + SP + SP + "std::size_t const input_idx =\n";
      for (std::size_t d = 0; d < D; ++d) {
         op += SP + SP + SP + SP + SP
             + "(out_" + std::to_string(d) + " % (" + fShapeInput[d].GetVal() + "))"
             + " * (" + inputStrides[d].GetVal() + ")";
         op += (d + 1 < D) ? " +\n" : ";\n\n";
      }

      op += SP + SP + SP + SP + "output[elem_idx] = input[input_idx];\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "TileKernel_" + opName;
      return SP + kname + " tileKernel_" + opName + ";\n";
   }

   std::string GenerateInitCode_GPU_ALPAKA() override {
      if (!fHasDynamicTiledAxis) return "";
      if (IsOutputPooled(fNY)) return "";
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Operator Tile called to Generate without being initialized first");

      std::string totalElements = ConvertDimShapeToLength(fShapeY);
      return SP + "deviceBuf_" + fNY + " = alpaka::allocBuf<" + fType + ", Idx>(devAcc, Ext1D::all(Idx{static_cast<Idx>(" + totalElements + ")}));\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Operator Tile called to Generate without being initialized first");

      std::string totalElements = ConvertDimShapeToLength(fShapeY);
      std::string kname = "tileKernel_" + opName;

      std::string args =
          "alpaka::getPtrNative(deviceBuf_" + fNInput + "), "
          + "alpaka::getPtrNative(deviceBuf_" + fNY + ")";
      for (auto &p : dynParamNames)
         args += ", static_cast<std::size_t>(" + p + ")";
      args += ", static_cast<Idx>(" + totalElements + ")";

      std::stringstream out;
      out << "\n//------ TILE_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{" << totalElements << "});\n";
      out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", " << kname << ", " << args << ");\n";
      out << SP <<"alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   EFusionMappingType GetFusionMappingType() const override
   {
      if (fShapeInput.empty() || fShapeY.empty())
         return EFusionMappingType::Unsupported;

      return EFusionMappingType::Shuffle;
   }

   std::vector<size_t> GetFusionDataInputIndices() const override
   {
      return {1};
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

   std::string GetFusionInputIndexExpr(size_t inputIndex, const std::string &outputIndex, const std::vector<Dim> &inputShape, const std::vector<Dim> &outputShape) const override
   {
      if (inputIndex != 1 || GetFusionMappingType() != EFusionMappingType::Shuffle)
         return "";

      if (inputShape.size() != outputShape.size())
         return "";

      const auto inputStrides = UTILITY::ComputeStrideFromShape(inputShape);
      const auto outputStrides = UTILITY::ComputeStrideFromShape(outputShape);
      std::string expression;

      for (size_t d = 0; d < outputShape.size(); ++d) {
         if (!expression.empty())
            expression += " + ";

         expression += "(((((" + outputIndex + ") / " + outputStrides[d].GetVal() + ") % " + outputShape[d].GetVal() + ") % " + inputShape[d].GetVal() + ") * " + inputStrides[d].GetVal() + ")";
      }

      return "(" + expression + ")";
   }
};

}//SOFIE

#endif //SOFIE_ROPERATOR_Tile
