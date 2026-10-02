#ifndef SOFIE_ROPERATOR_GATHERND
#define SOFIE_ROPERATOR_GATHERND

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>
#include <numeric>

namespace SOFIE {

class ROperator_GatherND final : public ROperator
{
private:

   int64_t fBatchDims = 0;

   std::string fNData;
   std::string fNIndices;
   std::string fNY;

   std::vector<Dim> fShapeData;
   std::vector<Dim> fShapeIndices;
   std::vector<Dim> fShapeY;

   std::string fType;

   // A size expression (a literal, or a symbolic dim name/expression from
   // Dim::GetVal()) wrapped as an explicit std::size_t cast.
   static std::string sz(const std::string &e) { return "static_cast<std::size_t>(" + e + ")"; }

public:
   ROperator_GatherND() {}
   ROperator_GatherND(int64_t batchDims,
                      std::string nameData,
                      std::string nameIndices,
                      std::string nameY)
      : fBatchDims(batchDims),
        fNData(UTILITY::Clean_name(nameData)),
        fNIndices(UTILITY::Clean_name(nameIndices)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames  = { fNData, fNIndices };
      fOutputTensorNames = { fNY };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return { input[0] };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      return { input[0] };
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNData))
         throw std::runtime_error("SOFIE GatherND: data tensor " + fNData + " not found in model");
      if (!model.CheckIfTensorAlreadyExist(fNIndices))
         throw std::runtime_error("SOFIE GatherND: indices tensor " + fNIndices + " not found in model");

      fShapeData    = model.GetDimTensorShape(fNData);
      fShapeIndices = model.GetDimTensorShape(fNIndices);

      size_t r = fShapeData.size();
      size_t q = fShapeIndices.size();
      size_t b = static_cast<size_t>(fBatchDims);

      if (r < 1)
         throw std::runtime_error("SOFIE GatherND: data rank must be >= 1");
      if (q < 1)
         throw std::runtime_error("SOFIE GatherND: indices rank must be >= 1");
      if (b >= std::min(q, r))
         throw std::runtime_error("SOFIE GatherND: batch_dims must be < min(q, r)");

      if (fShapeIndices.back().isParam)
         throw std::runtime_error("SOFIE GatherND: the last indices dimension (index tuple length) "
            "must be static - a dynamic index-tuple length is not supported");
      size_t last_idx_dim = fShapeIndices.back().dim;
      if (last_idx_dim > r - b)
         throw std::runtime_error("SOFIE GatherND: indices_shape[-1] must be <= r - batch_dims");

      for (size_t i = 0; i < b; ++i) {
         if (fShapeData[i] != fShapeIndices[i])
            throw std::runtime_error("SOFIE GatherND: first batch_dims dimensions of data and indices must match");
      }

      // Output shape: batch_dims + indices[0..q-2] + data[b + last_idx_dim .. r-1]
      // rank = b + (q - b - 1) + (r - b - last_idx_dim)
      //      = q + r - last_idx_dim - 1 - b
      fShapeY.clear();
      for (size_t i = 0; i < b; ++i)
         fShapeY.push_back(fShapeData[i]);
      for (size_t i = b; i + 1 < q; ++i)
         fShapeY.push_back(fShapeIndices[i]);
      for (size_t i = b + last_idx_dim; i < r; ++i)
         fShapeY.push_back(fShapeData[i]);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNData), fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNData));

      if (model.Verbose())
         std::cout << "GatherND: data " << ConvertDimShapeToString(fShapeData)
                   << " indices " << ConvertDimShapeToString(fShapeIndices)
                   << " batch_dims=" << fBatchDims
                   << " -> " << fNY << " " << ConvertDimShapeToString(fShapeY) << std::endl;
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeY.empty())
         throw std::runtime_error("SOFIE GatherND called to Generate without being initialized first");

      size_t r = fShapeData.size();
      size_t q = fShapeIndices.size();
      size_t b = static_cast<size_t>(fBatchDims);
      size_t last_idx_dim = fShapeIndices.back().dim;

      auto stridesData    = UTILITY::ComputeStrideFromShape(fShapeData);
      auto stridesIndices = UTILITY::ComputeStrideFromShape(fShapeIndices);
      auto stridesY       = UTILITY::ComputeStrideFromShape(fShapeY);

      std::string totalOutput = ConvertDimShapeToLength(fShapeY);

      std::stringstream out;
      out << SP << "//--------- GatherND operator " << opName << "\n";

      out << SP << "for (size_t out_idx = 0; out_idx < static_cast<size_t>(" << totalOutput << "); out_idx++) {\n";

      out << SP << SP << "size_t rem = out_idx;\n";
      size_t Dy = fShapeY.size();
      for (size_t d = 0; d < Dy; ++d) {
         out << SP << SP << "size_t oy_" << d << " = rem / " << stridesY[d].GetVal() << ";\n";
         out << SP << SP << "rem %= " << stridesY[d].GetVal() << ";\n";
      }

      out << SP << SP << "size_t idx_base = 0;\n";
      for (size_t i = 0; i < b; ++i)
         out << SP << SP << "idx_base += oy_" << i << " * " << stridesIndices[i].GetVal() << ";\n";
      for (size_t i = b; i + 1 < q; ++i)
         out << SP << SP << "idx_base += oy_" << i << " * " << stridesIndices[i].GetVal() << ";\n";

      out << SP << SP << "size_t data_idx = 0;\n";
      for (size_t i = 0; i < b; ++i)
         out << SP << SP << "data_idx += oy_" << i << " * " << stridesData[i].GetVal() << ";\n";

      out << SP << SP << "for (size_t k = 0; k < " << last_idx_dim << "; k++) {\n";
      out << SP << SP << SP << "int64_t idx_val = tensor_" << fNIndices
          << "[idx_base + k * " << stridesIndices[q - 1].GetVal() << "];\n";
      out << SP << SP << SP << "if (idx_val < 0) idx_val += " << "static_cast<int64_t>(tensor_"
          << fNData << "_shape[" << b << " + k]);\n";
      out << SP << SP << SP << "data_idx += static_cast<size_t>(idx_val) * " << "data_stride_b_plus_k_" << opName << "[k];\n";
      out << SP << SP << "}\n";

      // Accumulate trailing data dims from output coords
      // Y dims [b + (q-b-1) .. ] correspond to data dims [b + last_idx_dim .. r-1]
      size_t y_trailing_start = b + (q - b - 1);
      for (size_t i = b + last_idx_dim; i < r; ++i) {
         size_t oy_dim = y_trailing_start + (i - (b + last_idx_dim));
         out << SP << SP << "data_idx += oy_" << oy_dim << " * " << stridesData[i].GetVal() << ";\n";
      }

      out << SP << SP << "tensor_" << fNY << "[out_idx] = tensor_" << fNData << "[data_idx];\n";
      out << SP << "}\n";

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeY.empty())
         throw std::runtime_error("SOFIE GatherND called to Generate without being initialized first");

      size_t r = fShapeData.size();
      size_t q = fShapeIndices.size();
      size_t b = static_cast<size_t>(fBatchDims);
      size_t last_idx_dim = fShapeIndices.back().dim;

      auto stridesData    = UTILITY::ComputeStrideFromShape(fShapeData);
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
             + "int64_t idx_val = indices[idx_base + "
             + std::to_string(idx_offset) + "u];\n";
         op += SP + SP + SP + SP + SP
             + "if (idx_val < 0) idx_val += "
             + fShapeData[data_axis].GetVal() + ";\n";
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

      op += SP + SP + SP + SP + "output[elem_idx] = data[data_idx];\n";
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
      out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{static_cast<Idx>(" << totalElements << ")});\n";
      out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << opName
          << ", " << kname
          << ", alpaka::getPtrNative(deviceBuf_" << fNData << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNIndices << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")";
      for (auto &p : dynParamNames)
         out << ", static_cast<std::size_t>(" << p << ")";
      out << ", static_cast<Idx>(" << totalElements << "));\n";
      out << SP <<"alpaka::wait(queue);\n";
      return out.str();
   }
};

} // SOFIE

#endif // SOFIE_ROPERATOR_GATHERND
