#ifndef SOFIE_ROPERATOR_GatherND
#define SOFIE_ROPERATOR_GatherND

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>
namespace SOFIE{

class ROperator_GatherND final : public ROperator
{
private:

   size_t fBatchDims = 0;
   std::string fNX;
   std::string fNIndices;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeIndices;
   std::vector<Dim> fShapeY;

   std::vector<int64_t> fIndices;

   std::string fType;

   // data and indices tensors which are graph inputs read through their strides
   bool fStridedX = false;
   bool fStridedIndices = false;

   static std::string sz(const std::string &e) { return "static_cast<std::size_t>(" + e + ")"; }

public:
   ROperator_GatherND(){}
   ROperator_GatherND(int batch_dims, std::string nameX, std::string nameIndices, std::string nameY):
      fBatchDims(batch_dims), fNX(UTILITY::Clean_name(nameX)), fNIndices(UTILITY::Clean_name(nameIndices)), fNY(UTILITY::Clean_name(nameY)) {
         fInputTensorNames = { fNX, fNIndices };
         fOutputTensorNames = { fNY };
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("SOFIE GatherND Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      if (model.Verbose())
         std::cout << "GatherND - initial shape " << ConvertDimShapeToString(fShapeX) << " shape of indices "
               << ConvertDimShapeToString(model.GetDimTensorShape(fNIndices)) << std::endl;
      fShapeIndices = model.GetDimTensorShape(fNIndices);
      fStridedX = model.IsStridedInputTensor(fNX) && !fShapeX.empty();
      fStridedIndices = model.IsStridedInputTensor(fNIndices) && !fShapeIndices.empty();
      fHasStridedInput = fStridedX || fStridedIndices;
      size_t q = fShapeIndices.size();
      size_t r = fShapeX.size();

      if (q < 1) {
         throw std::runtime_error("SOFIE GatherND : rank of Indices is < 1");
      }
      if (r < 1) {
         throw std::runtime_error("SOFIE GatherND : rank of input tensor is < 1");
      }
      if (fBatchDims >= std::min(q,r)) {
         throw std::runtime_error("SOFIE GatherND : invalid batch dim value");
      }
      if (fBatchDims > 0) {
         for (size_t i = 0; i < fBatchDims; i++) {
            if (fShapeX[i] != fShapeIndices[i]) {
               std::cout << " input shape " << ConvertDimShapeToString(fShapeX) << " "
                         << " index shape " << ConvertDimShapeToString(fShapeIndices) << std::endl;
               throw std::runtime_error("SOFIE GatherND : invalid input or index shape for " + std::to_string(i));
            }
         }
      }

      if (fShapeIndices.back().isParam)
         throw std::runtime_error("SOFIE GatherND : Index_shape(-1) is not known");

      size_t last_index_shape = fShapeIndices.back().dim;
      if (last_index_shape < 1 || last_index_shape > r - fBatchDims) {
         throw std::runtime_error("SOFIE GatherND : Index_shape(-1) has wrong value " +
            std::to_string(last_index_shape));
      }

      size_t output_rank = r + q -1 - last_index_shape - fBatchDims;
      fShapeY = std::vector<Dim>(fShapeIndices.begin(), fShapeIndices.end() - 1);
      fShapeY.insert(fShapeY.end(), fShapeX.begin() + fBatchDims + last_index_shape, fShapeX.end());
      if (fShapeY.size() != output_rank) {
         std::cout << " input shape " << ConvertDimShapeToString(fShapeX) << " "
                         << " index shape " << ConvertDimShapeToString(fShapeIndices)
                         << " output shape " << ConvertDimShapeToString(fShapeY)
                         << " and output rank should be " << output_rank << std::endl;
         throw std::runtime_error("SOFIE GatherND : Something is wrong in initialization ");
      }

      if (!fIsOutputConstant) {
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
         fType = ConvertTypeToString(model.GetTensorType(fNX));
         if (model.Verbose())
               std::cout <<  "GatherND: input " << fNX << " " << ConvertDimShapeToString(fShapeX) << " indices " << fNIndices << ConvertDimShapeToString(fShapeIndices)
                         << " -> " << fNY << " with shape " << ConvertDimShapeToString(fShapeY) << std::endl;
      }





   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) {
         return "//---------------------------------------\n";
      }
      opName = "op_" + opName;
      std::stringstream out;
      out << "//--------- GatherND " << opName << " --> " << ConvertDimShapeToString(fShapeY) << "\n";
      // inputs read through the strides given to the Session, from the logical (contiguous) index of their elements
      const std::string xAt = "xoff_" + opName + "_X", iAt = "xoff_" + opName + "_I";
      if (fStridedX)
         out << GenerateStridedOffsetLambda(opName + "_X", fNX, fShapeX);
      if (fStridedIndices)
         out << GenerateStridedOffsetLambda(opName + "_I", fNIndices, fShapeIndices);
      size_t r = fShapeX.size();
      size_t q = fShapeIndices.size();
      auto stridesX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto stridesY = UTILITY::ComputeStrideFromShape(fShapeY);
      auto stridesIndices = UTILITY::ComputeStrideFromShape(fShapeIndices);

      size_t ss = fShapeIndices.back().dim;

      out << SP << "{\n";
      std::string outIndex;
      std::string inIndex;
      std::string idIndex;
      for (size_t j = 0; j < fBatchDims; j++) {
         std::string index = "i_" + std::to_string(j);
         for (size_t k = 0; k <= j; k++) out << SP;
         out << "for (size_t " << index << " = 0; " << index << " < " << fShapeY[j] << "; " << index << "++) {\n";
         if (j > 0) {
            outIndex += " + ";
            inIndex += " + ";
            idIndex += " + ";
         }
         outIndex += index;
         if (stridesY[j].GetVal() != "1")
            outIndex += " * " + stridesY[j].GetVal();
         inIndex += index;
         if (stridesX[j].GetVal() != "1")
            inIndex += " * " + stridesX[j].GetVal();
         idIndex += index;
         if (stridesIndices[j].GetVal() != "1")
            idIndex += " * " + stridesIndices[j].GetVal();
      }
      for (size_t j = fBatchDims; j < q - 1; j++) {
         std::string index = "i_" + std::to_string(j);
         for (size_t k = 0; k <= j; k++) out << SP;
         out << "for (size_t " << index << " = 0; " << index << " < " << fShapeY[j] << "; " << index << "++) {\n";
         if (j > 0) {
            outIndex += " + ";
            idIndex += " + ";
         }
         outIndex += index;
         if (stridesY[j].GetVal() != "1")
            outIndex += " * " + stridesY[j].GetVal();
         idIndex += index;
         if (stridesIndices[j].GetVal() != "1")
            idIndex += " * " + stridesIndices[j].GetVal();
      }
      for (size_t l = 0; l < ss; l++) {
         std::string indexIndex =
            idIndex.empty() ? std::to_string(l) : (l > 0 ? idIndex + " + " + std::to_string(l) : idIndex);
         for (size_t k = 0; k <= q - 1; k++)
            out << SP;
         if (fStridedIndices)
            out << "int64_t index_" << l << " = tensor_" << fNIndices << "[" << iAt << "(" << indexIndex << ")];\n";
         else
            out << "int64_t index_" << l << " = tensor_" << fNIndices << "[" << indexIndex << "];\n";
         for (size_t k = 0; k <= q - 1; k++)
            out << SP;
         out << "if (index_" << l << " < 0) index_" << l << " += " << fShapeX[fBatchDims + l] << ";\n";
      }
      for (size_t k = 0; k <= q - 1; k++) out << SP;
      out << "size_t inputIndex = " << inIndex;
      for (size_t l = 0; l < ss; l++) {
         if (!inIndex.empty() || l > 0)
            out << " + ";
         out << "index_" << l;
         if (stridesX[fBatchDims + l].GetVal() != "1") out
             << " * " << stridesX[fBatchDims + l];
      }
      out << ";\n";
      for (size_t k = 0; k <= q - 1; k++) out << SP;
      if (ss == r - fBatchDims) {
         out << "tensor_" << fNY << "[" << outIndex << "] = "
             << "tensor_" << fNX << (fStridedX ? "[" + xAt + "(inputIndex)]" : std::string("[inputIndex]")) << ";\n";
      } else if (fStridedX) {
         // the slice is not contiguous in memory: copy it element by element
         out << "for (size_t s = 0; s < " << stridesX[fBatchDims + ss - 1] << "; s++)\n";
         for (size_t k = 0; k <= q; k++) out << SP;
         out << "tensor_" << fNY << "[" << outIndex << " + s] = tensor_" << fNX << "[" << xAt << "(inputIndex + s)];\n";
      } else {
         out << "std::copy(tensor_" << fNX << " + inputIndex, tensor_" << fNX << " + inputIndex + "
             << stridesX[fBatchDims + ss - 1] << ","
             << "tensor_" << fNY << "+" << outIndex << ");\n";
      }

      for (size_t j = q-1; j > 0; j--) {
         for (size_t k = 0; k <j; k++) out << SP;
         out << "}\n";
      }
      out << SP << "}\n";

      return out.str();
   }


   std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeY.empty())
         throw std::runtime_error("SOFIE GatherND called to Generate without being initialized first");

      size_t r = fShapeX.size();
      size_t q = fShapeIndices.size();
      size_t b = static_cast<size_t>(fBatchDims);
      size_t last_idx_dim = fShapeIndices.back().dim;

      auto stridesData    = UTILITY::ComputeStrideFromShape(fShapeX);
      auto stridesIndices = UTILITY::ComputeStrideFromShape(fShapeIndices);
      auto stridesY       = UTILITY::ComputeStrideFromShape(fShapeY);

      size_t Dy = fShapeY.size();

      std::string kname = "GatherNDKernel_" + opName;

      std::string op;
      op  = "\n//------ GATHERND_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ data,\n";
      op += SP + SP + SP + "int64_t const* __restrict__ indices,\n";
      op += SP + SP + SP + "T* __restrict__ output,\n";
      for (auto &p : dynParamNames)
         op += SP + SP + SP + "std::size_t const " + p + ",\n";
      if (fStridedX)
         op += SP + SP + SP + "sofie_strided_layout<" + std::to_string(r) + "> const layoutX,\n";
      if (fStridedIndices)
         op += SP + SP + SP + "sofie_strided_layout<" + std::to_string(q) + "> const layoutIndices,\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      for (size_t d = 0; d < Dy; ++d) {
         op += SP + SP + SP + SP + "std::size_t const oy_" + std::to_string(d)
             + " = (elem_idx / " + sz(stridesY[d].GetVal()) + ") % "
             + sz(fShapeY[d].GetVal()) + ";\n";
      }
      op += "\n";

      op += SP + SP + SP + SP + "std::size_t const idx_base =\n";
      // batch dims: oy_0..oy_{b-1} * stridesIndices[0..b-1]
      // outer idx dims: oy_b..oy_{b+(q-b-2)} * stridesIndices[b..q-2]
      bool first = true;
      for (size_t i = 0; i < q - 1; ++i) {
         op += SP + SP + SP + SP + SP
             + (first ? "" : "+ ")
             + "oy_" + std::to_string(i) + " * " + sz(stridesIndices[i].GetVal()) + "\n";
         first = false;
      }
      if (first) op += SP + SP + SP + SP + SP + "0u\n"; // q==1: scalar index tuple
      op += SP + SP + SP + SP + SP + ";\n\n";

      op += SP + SP + SP + SP + "std::size_t data_idx =\n";
      first = true;
      for (size_t i = 0; i < b; ++i) {
         op += SP + SP + SP + SP + SP
             + (first ? "" : "+ ")
             + "oy_" + std::to_string(i) + " * " + sz(stridesData[i].GetVal()) + "\n";
         first = false;
      }
      if (first) op += SP + SP + SP + SP + SP + "0u\n";
      op += SP + SP + SP + SP + SP + ";\n\n";

      op += SP + SP + SP + SP + "// Read " + std::to_string(last_idx_dim) + "-element index tuple\n";
      for (size_t k = 0; k < last_idx_dim; ++k) {
         size_t idx_offset = k;
         size_t data_axis  = b + k;
         op += SP + SP + SP + SP + "{\n";
         op += SP + SP + SP + SP + SP
             + "int64_t idx_val = "
             + (fStridedIndices
                   ? StridedKernelRead("indices", "layoutIndices", "idx_base + " + std::to_string(idx_offset) + "u")
                   : "indices[idx_base + " + std::to_string(idx_offset) + "u]")
             + ";\n";
         op += SP + SP + SP + SP + SP
             + "if (idx_val < 0) idx_val += "
             + fShapeX[data_axis].GetVal() + ";\n";
         op += SP + SP + SP + SP + SP
             + "data_idx += static_cast<std::size_t>(idx_val) * "
             + sz(stridesData[data_axis].GetVal()) + ";\n";
         op += SP + SP + SP + SP + "}\n";
      }
      op += "\n";

      size_t y_trailing_start = b + (q - b - 1);
      for (size_t i = b + last_idx_dim; i < r; ++i) {
         size_t oy_dim = y_trailing_start + (i - (b + last_idx_dim));
         op += SP + SP + SP + SP
             + "data_idx += oy_" + std::to_string(oy_dim)
             + " * " + sz(stridesData[i].GetVal()) + ";\n";
      }
      op += "\n";

      op += SP + SP + SP + SP + "output[elem_idx] = "
            + (fStridedX ? StridedKernelRead("data", "layoutX", "data_idx") : std::string("data[data_idx]")) + ";\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "GatherNDKernel_" + opName;
      return SP + kname + " gatherNDKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeY.empty())
         throw std::runtime_error("SOFIE GatherND called to Generate without being initialized first");

      std::string totalElements = ConvertDimShapeToLength(fShapeY);
      std::string kname = "gatherNDKernel_" + opName;

      std::stringstream out;
      out << "\n//------ GATHERND_GPU_ALPAKA\n";
      if (fStridedX)
         out << GenerateStridedBroadcastLayout(opName + "_X", fNX, fShapeX, fShapeX.size(), fShapeX);
      if (fStridedIndices)
         out << GenerateStridedBroadcastLayout(opName + "_I", fNIndices, fShapeIndices, fShapeIndices.size(),
                                               fShapeIndices);
      out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{static_cast<Idx>(" << totalElements << ")});\n";
      out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << opName
          << ", " << kname
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNIndices << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")";
      for (auto &p : dynParamNames)
         out << ", static_cast<std::size_t>(" << p << ")";
      if (fStridedX)
         out << ", layout_" << opName << "_X";
      if (fStridedIndices)
         out << ", layout_" << opName << "_I";
      out << ", static_cast<Idx>(" << totalElements << "));\n";
      out << SP <<"alpaka::wait(queue);\n";
      return out.str();
   }
};

}


#endif
