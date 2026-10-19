#ifndef SOFIE_ROPERATOR_ScatterND
#define SOFIE_ROPERATOR_ScatterND

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>
namespace SOFIE{

class ROperator_ScatterND final : public ROperator
{
private:


   std::string fNX;
   std::string fNI;
   std::string fNU;
   std::string fNY;
   std::string fReduction;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeI;
   std::vector<Dim> fShapeY;


   std::vector<int64_t> fIndices;

   std::string fType;

   // inputs which are graph inputs read through their strides
   bool fStridedX = false;
   bool fStridedI = false;
   bool fStridedU = false;
   std::vector<Dim> fShapeU;

   size_t fK = 0;
   std::string fSliceSizeExpr;
   std::string fNumOuterExpr;

   static std::string sz(const std::string &e) { return "static_cast<std::size_t>(" + e + ")"; }


public:
   ROperator_ScatterND(){}
   ROperator_ScatterND(const std::string & nameX, const std::string & nameI, const std::string & nameU, const std::string & nameY,
                        std::string reduction):
      fNX(UTILITY::Clean_name(nameX)), fNI(UTILITY::Clean_name(nameI)), fNU(UTILITY::Clean_name(nameU)),
      fNY(UTILITY::Clean_name(nameY)), fReduction(reduction)
   {
      fInputTensorNames = { fNX, fNI, fNU };
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel& model) override {

      if (!model.CheckIfTensorAlreadyExist(fNX)){
         throw std::runtime_error(std::string("SOFIE ScatterND Op Input Tensor ") + fNX + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNI)) {
         throw std::runtime_error(std::string("SOFIE ScatterND Op Input Tensor ") + fNI + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNU)) {
         throw std::runtime_error(std::string("SOFIE ScatterND Op Input Tensor ") + fNU + "is not found in model");
      }

      fShapeX = model.GetDimTensorShape(fNX);
      fShapeI = model.GetDimTensorShape(fNI);
      auto shapeU = model.GetDimTensorShape(fNU);
      fShapeU = shapeU;
      fStridedX = model.IsStridedInputTensor(fNX) && !fShapeX.empty();
      fStridedI = model.IsStridedInputTensor(fNI) && !fShapeI.empty();
      fStridedU = model.IsStridedInputTensor(fNU) && !shapeU.empty();
      fHasStridedInput = fStridedX || fStridedI || fStridedU;

      const size_t r = fShapeX.size();
      const size_t q = fShapeI.size();
      if (!(fShapeI.back().isParam) ) {
         const size_t k = fShapeI.back().dim;

         if (k > r)
            throw std::invalid_argument(
               "ScatterND: last dim of indices (" + std::to_string(k) +
               ") must be <= rank of data (" + std::to_string(r) + ")");

         int64_t expected_updates_rank = q - 1 + r - k;
         if ((int64_t) shapeU.size() != expected_updates_rank)
            throw std::invalid_argument("ScatterND: updates rank mismatch");
         fK = k;
         std::vector<Dim> outerShape(fShapeI.begin(), fShapeI.end() - 1);
         fNumOuterExpr = ConvertDimShapeToLength(outerShape);
         std::vector<Dim> sliceShape(fShapeX.begin() + k, fShapeX.end());
         fSliceSizeExpr = ConvertDimShapeToLength(sliceShape);
      } else {
         throw std::runtime_error("SOFIE ScatterND : Index_shape(-1) is not known. This case is not supported");
      }

      fShapeY = fShapeX;

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      if (model.Verbose()) {
         std::cout << "ScatterElements: input: " << ConvertDimShapeToString(fShapeX)
                                                << " indices " << ConvertDimShapeToString(fShapeI)
                                                << " update " <<  ConvertDimShapeToString(shapeU);
         std::cout << "\t----> " << ConvertDimShapeToString(fShapeY) << std::endl;
      }
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) {
         return "//---------------------------------------\n";
      }
      opName = "op_" + opName;
      std::stringstream out;
      out << "//--------- ScatterND " << opName << " --> " << ConvertDimShapeToString(fShapeY) << "\n";
      // inputs read through the strides given to the Session, from the logical (contiguous) index of their elements
      if (fStridedI)
         out << GenerateStridedOffsetLambda(opName + "_I", fNI, fShapeI);
      if (fStridedU)
         out << GenerateStridedOffsetLambda(opName + "_U", fNU, fShapeU);
      auto readI = [&](const std::string &index) {
         return fStridedI ? "tensor_" + fNI + "[xoff_" + opName + "_I(" + index + ")]" : "tensor_" + fNI + "[" + index + "]";
      };
      auto readU = [&](const std::string &index) {
         return fStridedU ? "tensor_" + fNU + "[xoff_" + opName + "_U(" + index + ")]" : "tensor_" + fNU + "[" + index + "]";
      };

      size_t r = fShapeX.size();

      auto stridesX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto stridesY = UTILITY::ComputeStrideFromShape(fShapeY);
      auto stridesI = UTILITY::ComputeStrideFromShape(fShapeI);

      size_t k = fShapeI.back().dim;

      std::vector<Dim> shapeIndFirst(fShapeI.begin(), fShapeI.begin()+ fShapeI.size()-1);
      auto num_index_tuples = ConvertDimShapeToLength(shapeIndFirst);

      std::vector<Dim> shapeSlice(fShapeX.begin()+k, fShapeX.end());
      auto slice_size = ConvertDimShapeToLength(shapeSlice);

      auto data_length = ConvertDimShapeToLength(fShapeX);

      out << SP << "// Step 1: copy input data to output\n";
      if (fStridedX)
         out << GenerateStridedUnaryLoop(opName + "_cp", fNX, fNY, fShapeX, [](const std::string &v) { return v; });
      else
         out << SP << "std::copy(tensor_" << fNX << ", tensor_" << fNX << " + " << data_length << ", tensor_" << fNY << ");\n";

      out << SP << "// Step 2: data strides (row-major)\n";
      out << SP << "size_t " << opName << "_data_strides[" << r << "] = {";
      for (size_t i = 0; i < r; ++i)
         out << stridesX[i] << (i + 1 < r ? ", " : "");
      out << "};\n\n";

      out << SP << "// Step 3: scatter updates into output\n";
      out << SP << "for (int64_t idx = 0; idx < " << num_index_tuples << "; idx++) {\n";

      out << SP << SP << "int64_t data_offset = 0;\n";
      for (size_t dim = 0; dim < k; ++dim) {
         out << SP << SP << "{\n";
         out << SP << SP << SP << "int64_t coord = " << readI("idx * " + std::to_string(k) + " + " + std::to_string(dim))
             << ";\n";
         out << SP << SP << SP << "if (coord < 0) coord += " << fShapeX[dim] << ";\n";
         out << SP << SP << SP << "data_offset += coord * "
               << opName << "_data_strides[" << dim << "];\n";
         out << SP << SP << "}\n";
      }

      out << SP << SP << "for (int64_t s = 0; s < " << slice_size << "; s++) {\n";
      out << SP << SP << SP << "auto upd = " << readU("idx * " + slice_size + " + s") << ";\n";

      if (fReduction.empty() || fReduction == "none") {
         out << SP << SP << SP << "tensor_" << fNY << "[data_offset + s] = upd;\n";
      } else if (fReduction == "add") {
         out << SP << SP << SP << "tensor_" << fNY<< "[data_offset + s] += upd;\n";
      } else if (fReduction == "mul") {
         out << SP << SP << SP << "tensor_" << fNY << "[data_offset + s] *= upd;\n";
      } else if (fReduction == "min") {
         out << SP << SP << SP << "tensor_" << fNY<< "[data_offset + s] = "
               << "std::min(tensor_" << fNY << "[data_offset + s], upd);\n";
      } else if (fReduction == "max") {
         out << SP << SP << SP << "tensor_" << fNY << "[data_offset + s] = "
            << "std::max(tensor_" << fNY << "[data_offset + s], upd);\n";
      } else {
         throw std::runtime_error(
            "SOFIE ScatterND: unsupported reduction '" + fReduction + "'");
      }

      out << SP << SP << "}\n";
      out << SP << "}\n";

      return out.str();
   }


   std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeY.empty())
         throw std::runtime_error("SOFIE ScatterND: Generate_GPU_Kernel_ALPAKA called before Initialize");

      std::string kname = "ScatterNDKernel_" + opName;
      auto stridesData = UTILITY::ComputeStrideFromShape(fShapeX);

      std::string op;
      // the data input is copied in the output through its strides, before the updates are scattered
      if (fStridedX)
         op += GenerateStridedUnaryKernel("ScatterNDCopyKernel_" + opName, "SCATTERND_COPY",
                                          [](const std::string &v) { return v; });
      op += "\n//------ SCATTERND_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T* Y,\n";
      op += SP + SP + SP + "int64_t const* indices,\n";
      op += SP + SP + SP + "T const* updates,\n";
      for (auto &p : dynParamNames)
         op += SP + SP + SP + "std::size_t const " + p + ",\n";
      if (fStridedI)
         op += SP + SP + SP + "sofie_strided_layout<" + std::to_string(fShapeI.size()) + "> const layoutIndices,\n";
      if (fStridedU)
         op += SP + SP + SP + "sofie_strided_layout<" + std::to_string(fShapeU.size()) + "> const layoutUpdates,\n";
      op += SP + SP + SP + "std::size_t const numOuter,\n";
      op += SP + SP + SP + "std::size_t const sliceSize) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      // One thread per outer update position (handles the full slice serially)
      op += SP + SP + SP + "for (std::size_t i = global_thread_idx; i < numOuter; i += grid_thread_extent) {\n";
      op += SP + SP + SP + SP + "std::size_t out_base = 0;\n";

      for (size_t j = 0; j < fK; ++j) {
         op += SP + SP + SP + SP + "{\n";
         op += SP + SP + SP + SP + SP
             + "int64_t idx = "
             + (fStridedI ? StridedKernelRead("indices", "layoutIndices", "i * " + std::to_string(fK) + "u + " + std::to_string(j) + "u")
                          : "indices[i * " + std::to_string(fK) + "u + " + std::to_string(j) + "u]")
             + ";\n";
         op += SP + SP + SP + SP + SP
             + "if (idx < 0) idx += " + fShapeX[j].GetVal() + ";\n";
         op += SP + SP + SP + SP + SP
             + "out_base += static_cast<std::size_t>(idx) * " + sz(stridesData[j].GetVal()) + ";\n";
         op += SP + SP + SP + SP + "}\n";
      }

      op += SP + SP + SP + SP + "for (std::size_t s = 0; s < sliceSize; ++s) {\n";
      op += SP + SP + SP + SP + SP + "std::size_t const out_idx = out_base + s;\n";
      op += SP + SP + SP + SP + SP + "std::size_t const upd_idx = " + std::string(fStridedU ? "sofie_strided_offset(layoutUpdates, i * sliceSize + s)" : "i * sliceSize + s") + ";\n";

      if (fReduction.empty() || fReduction == "none") {
         op += SP + SP + SP + SP + SP + "Y[out_idx] = updates[upd_idx];\n";
      } else if (fReduction == "add") {
         op += SP + SP + SP + SP + SP + "alpaka::atomicAdd(acc, &Y[out_idx], updates[upd_idx]);\n";
      } else if (fReduction == "mul") {
         op += SP + SP + SP + SP + SP + "alpaka::atomicMul(acc, &Y[out_idx], updates[upd_idx]);\n";
      } else if (fReduction == "max") {
         op += SP + SP + SP + SP + SP + "alpaka::atomicMax(acc, &Y[out_idx], updates[upd_idx]);\n";
      } else if (fReduction == "min") {
         op += SP + SP + SP + SP + SP + "alpaka::atomicMin(acc, &Y[out_idx], updates[upd_idx]);\n";
      }

      op += SP + SP + SP + SP + "}\n"; // slice loop
      op += SP + SP + SP + "}\n";      // outer loop
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "ScatterNDKernel_" + opName;
      std::string defs = SP + kname + " scatterNDKernel_" + opName + ";\n";
      if (fStridedX)
         defs += SP + "ScatterNDCopyKernel_" + opName + " scatterNDCopyKernel_" + opName + ";\n";
      return defs;
   }

   std::string Generate_GPU_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeY.empty())
         throw std::runtime_error("SOFIE ScatterND: Generate_GPU_ALPAKA called before Initialize");

      std::stringstream out;
      out << "\n//------ SCATTERND_GPU_ALPAKA\n";

      if (fStridedI)
         out << GenerateStridedBroadcastLayout(opName + "_I", fNI, fShapeI, fShapeI.size(), fShapeI);
      if (fStridedU)
         out << GenerateStridedBroadcastLayout(opName + "_U", fNU, fShapeU, fShapeU.size(), fShapeU);
      if (fStridedX)
         out << GenerateStridedUnaryLaunch(opName + "_copy", "scatterNDCopyKernel_" + opName, "SCATTERND_COPY", fNX, fNY,
                                           fShapeX);
      else
         out << SP << "alpaka::memcpy(queue, deviceBuf_" << fNY << ", deviceBuf_" << fNX << ");\n";

      out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(static_cast<Idx>(" << fNumOuterExpr << "));\n";
      out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", scatterNDKernel_" << opName
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNI << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNU << ")";
      for (auto &p : dynParamNames)
         out << ", static_cast<std::size_t>(" << p << ")";
      if (fStridedI)
         out << ", layout_" << opName << "_I";
      if (fStridedU)
         out << ", layout_" << opName << "_U";
      out << ", static_cast<Idx>(" << fNumOuterExpr << ")"
          << ", static_cast<Idx>(" << fSliceSizeExpr << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";

      return out.str();
   }
};

}


#endif
