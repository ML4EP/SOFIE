#ifndef SOFIE_ROPERATOR_Tile
#define SOFIE_ROPERATOR_Tile

#include "SOFIE/utils/SOFIE_common.hxx"
#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename T>
class ROperator_Tile final : public ROperator_ALPAKA
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
      fHasStridedInput = model.IsStridedInputTensor(fNInput) && !fShapeInput.empty();

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

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeY.empty()) {
            throw std::runtime_error("SOFIE Tile Op called to Generate without being initialized first");
      }

      std::stringstream out;
      out << "///-------- Tile operator " << OpName << "\n";
      out << "{\n";

      const int rank = fShapeInput.size();

      out << "const size_t input_shape[" << rank << "] = " << ConvertDimShapeToString(fShapeInput) << ";\n";
      out << "const size_t output_shape[" << rank << "] = " << ConvertDimShapeToString(fShapeY) << ";\n\n";

      if (fHasStridedInput) {
         // input read through the strides given to the Session
         out << GenerateInputStrideArray("input_strides", fNInput, fShapeInput) << "\n";
      } else {
         out << "size_t input_strides[" << rank << "];\n";
         out << "input_strides[" << rank - 1 << "] = 1;\n";
         out << "for (int i = " << rank - 2 << "; i >= 0; --i) {\n";
         out << "input_strides[i] = input_strides[i+1] * input_shape[i+1];\n";
         out << "}\n\n";
      }

      out << "size_t out_idx = 0;\n";
      for (int i = 0; i < rank; ++i) {
         out << "for (size_t o" << i << " = 0, ic" << i << " = 0; o" << i
             << " < output_shape[" << i << "]; ++o" << i << ") {\n";
         out << "const size_t in_off" << i << " = "
             << (i == 0 ? std::string() : "in_off" + std::to_string(i - 1) + " + ")
             << "ic" << i << " * input_strides[" << i << "];\n";
      }
      out << "tensor_" << fNY << "[out_idx++] = tensor_" << fNInput << "[in_off" << rank - 1 << "];\n";
      for (int i = rank - 1; i >= 0; --i) {
         out << "if (++ic" << i << " == input_shape[" << i << "]) ic" << i << " = 0;\n";
         out << "}\n";
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
      // input read through the strides given to the Session: the kernel receives them in a layout
      if (fHasStridedInput)
         for (std::size_t d = 0; d < D; ++d)
            inputStrides[d] = Dim{"layoutX.stride[" + std::to_string(d) + "]", static_cast<std::size_t>(-1)};

      std::string kname = "TileKernel_" + opName;

      std::string op;
      op  = "\n//------ TILE_KERNEL_ALPAKA\n";
      op += "struct " + kname + " {\n";
      op += "template<typename TAcc, typename T>\n";
      op += "ALPAKA_FN_ACC void operator()(\n";
      op += "TAcc const& acc,\n";
      op += "T const* __restrict__ input,\n";
      op += "T* __restrict__ output,\n";
      for (auto &p : dynParamNames)
         op += "std::size_t const " + p + ",\n";
      if (fHasStridedInput)
         op += "sofie_strided_layout<" + std::to_string(D) + "> const layoutX,\n";
      op += "std::size_t const totalElements) const {\n\n";

      op += "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += "if (global_thread_idx >= totalElements) return;\n";
      op += "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      EmitOutputCoordsFromThreadIdx(op, outputStrides, fShapeY);
      op += "\n";

      // Input index: tiling wraps each output coordinate back into the input shape
      op += "std::size_t const input_idx =\n";
      for (std::size_t d = 0; d < D; ++d) {
         op += "(out_" + std::to_string(d) + " % (" + fShapeInput[d].GetVal() + ")) * (" + inputStrides[d].GetVal() + ")";
         op += (d + 1 < D) ? " +\n" : ";\n\n";
      }

      op += "output[elem_idx] = input[input_idx];\n";
      op += "}\n";
      op += "}\n";
      op += "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "TileKernel_" + opName;
      return kname + " tileKernel_" + opName + ";\n";
   }

   std::string GenerateInitCode_GPU_ALPAKA() override {
      if (!fHasDynamicTiledAxis) return "";
      if (IsOutputPooled(fNY)) return "";
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Operator Tile called to Generate without being initialized first");

      std::string totalElements = ConvertDimShapeToLength(fShapeY);
      return "deviceBuf_" + fNY + " = alpaka::allocBuf<" + fType + ", Idx>(devAcc, Ext1D::all(Idx{static_cast<Idx>(" + totalElements + ")}));\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Operator Tile called to Generate without being initialized first");

      std::string totalElements = ConvertDimShapeToLength(fShapeY);
      std::string kname = "tileKernel_" + opName;

      std::string args =
          "alpaka::getPtrNative(deviceBuf_" + fNInput + "), alpaka::getPtrNative(deviceBuf_" + fNY + ")";
      for (auto &p : dynParamNames)
         args += ", static_cast<std::size_t>(" + p + ")";
      if (fHasStridedInput)
         args += ", layout_" + opName + "_X";
      args += ", static_cast<Idx>(" + totalElements + ")";

      std::stringstream out;
      out << "\n//------ TILE_GPU_ALPAKA\n";
      if (fHasStridedInput)
         out << GenerateStridedBroadcastLayout(opName + "_X", fNInput, fShapeInput, fShapeInput.size(), fShapeInput);
      out << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
      out << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{" << totalElements << "});\n";
      out << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", " << kname << ", " << args << ");\n";
      out << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   EFusionMappingType GetFusionMappingType() const override
   {
      if (fHasStridedInput || fShapeInput.empty() || fShapeY.empty())
         return EFusionMappingType::Unsupported;

      return EFusionMappingType::Shuffle;
   }

   std::vector<size_t> GetFusionDataInputIndices() const override
   {
      return {0};
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
      if (inputIndex != 0 || GetFusionMappingType() != EFusionMappingType::Shuffle)
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
