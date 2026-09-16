#ifndef SOFIE_ROPERATOR_SCATTERND
#define SOFIE_ROPERATOR_SCATTERND

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>
#include <numeric>

namespace SOFIE {

class ROperator_ScatterND final : public ROperator {
private:
   std::string fNData;
   std::string fNIndices;
   std::string fNUpdates;
   std::string fNY;
   std::string fReduction;

   std::vector<Dim> fShapeData;
   std::vector<Dim> fShapeIndices;
   std::vector<Dim> fShapeY;

   size_t fK          = 0;
   std::string fSliceSize;   // host size expression
   std::string fNumOuter;    // host size expression

   std::string fType;

   static std::string sz(const std::string &e) { return "static_cast<std::size_t>(" + e + ")"; }

public:
   ROperator_ScatterND() {}

   ROperator_ScatterND(const std::string& nameData,
                       const std::string& nameIndices,
                       const std::string& nameUpdates,
                       const std::string& nameY,
                       const std::string& reduction)
      : fNData(UTILITY::Clean_name(nameData)),
        fNIndices(UTILITY::Clean_name(nameIndices)),
        fNUpdates(UTILITY::Clean_name(nameUpdates)),
        fNY(UTILITY::Clean_name(nameY)),
        fReduction(reduction)
   {
      fInputTensorNames  = { fNData, fNIndices, fNUpdates };
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
         throw std::runtime_error("SOFIE ScatterND: data tensor " + fNData + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNIndices))
         throw std::runtime_error("SOFIE ScatterND: indices tensor " + fNIndices + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNUpdates))
         throw std::runtime_error("SOFIE ScatterND: updates tensor " + fNUpdates + " not found");

      fShapeData    = model.GetDimTensorShape(fNData);
      fShapeIndices = model.GetDimTensorShape(fNIndices);

      size_t r = fShapeData.size();
      size_t q = fShapeIndices.size();

      if (r < 1)
         throw std::runtime_error("SOFIE ScatterND: data rank must be >= 1");
      if (q < 1)
         throw std::runtime_error("SOFIE ScatterND: indices rank must be >= 1");

      if (fShapeIndices.back().isParam)
         throw std::runtime_error("SOFIE ScatterND: the last indices dimension (index tuple length) "
            "must be static - a dynamic index-tuple length is not supported");
      fK = fShapeIndices.back().dim;
      if (fK > r)
         throw std::runtime_error("SOFIE ScatterND: indices.shape[-1] must be <= data rank");

      std::vector<Dim> outerShape(fShapeIndices.begin(), fShapeIndices.end() - 1);
      fNumOuter = ConvertDimShapeToLength(outerShape);

      std::vector<Dim> sliceShape(fShapeData.begin() + fK, fShapeData.end());
      fSliceSize = ConvertDimShapeToLength(sliceShape);

      fShapeY = fShapeData;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNData), fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNData));

      if (model.Verbose())
         std::cout << "ScatterND: data " << ConvertDimShapeToString(fShapeData)
                   << " indices " << ConvertDimShapeToString(fShapeIndices)
                   << " k=" << fK << " numOuter=" << fNumOuter
                   << " sliceSize=" << fSliceSize
                   << " reduction=" << (fReduction.empty() ? "none" : fReduction)
                   << " -> " << fNY << " " << ConvertDimShapeToString(fShapeY) << "\n";
   }

   std::string GenerateInitCode() override { return ""; }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) return "";
      if (fShapeY.empty())
         throw std::runtime_error("SOFIE ScatterND: Generate called before Initialize");

      std::string dataSize = ConvertDimShapeToLength(fShapeY);
      auto stridesData = UTILITY::ComputeStrideFromShape(fShapeData);

      std::stringstream out;
      out << SP << "//------- ScatterND " << opName << "\n";

      out << SP << "std::copy(tensor_" << fNData
          << ", tensor_" << fNData << " + static_cast<size_t>(" << dataSize << ")"
          << ", tensor_" << fNY << ");\n";

      out << SP << "for (std::size_t _i = 0; _i < static_cast<std::size_t>(" << fNumOuter << "); ++_i) {\n";
      out << SP << SP << "std::size_t _out_base = 0;\n";

      for (size_t j = 0; j < fK; ++j) {
         out << SP << SP << "{\n";
         out << SP << SP << SP << "int64_t _idx = tensor_" << fNIndices
             << "[_i * " << fK << " + " << j << "];\n";
         out << SP << SP << SP << "if (_idx < 0) _idx += "
             << "static_cast<int64_t>(" << fShapeData[j].GetVal() << ");\n";
         out << SP << SP << SP << "_out_base += static_cast<std::size_t>(_idx) * "
             << stridesData[j].GetVal() << ";\n";
         out << SP << SP << "}\n";
      }

      out << SP << SP << "for (std::size_t _s = 0; _s < static_cast<std::size_t>(" << fSliceSize << "); ++_s) {\n";
      out << SP << SP << SP << "std::size_t const _out_idx = _out_base + _s;\n";
      out << SP << SP << SP << "std::size_t const _upd_idx = _i * static_cast<std::size_t>(" << fSliceSize << ") + _s;\n";

      if (fReduction.empty() || fReduction == "none") {
         out << SP << SP << SP << "tensor_" << fNY << "[_out_idx] = tensor_" << fNUpdates << "[_upd_idx];\n";
      } else if (fReduction == "add") {
         out << SP << SP << SP << "tensor_" << fNY << "[_out_idx] += tensor_" << fNUpdates << "[_upd_idx];\n";
      } else if (fReduction == "mul") {
         out << SP << SP << SP << "tensor_" << fNY << "[_out_idx] *= tensor_" << fNUpdates << "[_upd_idx];\n";
      } else if (fReduction == "max") {
         out << SP << SP << SP << "tensor_" << fNY << "[_out_idx] = std::max(tensor_" << fNY
             << "[_out_idx], tensor_" << fNUpdates << "[_upd_idx]);\n";
      } else if (fReduction == "min") {
         out << SP << SP << SP << "tensor_" << fNY << "[_out_idx] = std::min(tensor_" << fNY
             << "[_out_idx], tensor_" << fNUpdates << "[_upd_idx]);\n";
      } else {
         throw std::runtime_error("SOFIE ScatterND: invalid reduction '" + fReduction + "'");
      }

      out << SP << SP << "}\n"; // slice loop
      out << SP << "}\n";       // outer loop
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeY.empty())
         throw std::runtime_error("SOFIE ScatterND: Generate_GPU_Kernel_ALPAKA called before Initialize");

      std::string kname = "ScatterNDKernel_" + opName;
      auto stridesData = UTILITY::ComputeStrideFromShape(fShapeData);

      std::string op;
      op  = "\n//------ SCATTERND_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T* Y,\n";
      op += SP + SP + SP + "int64_t const* indices,\n";
      op += SP + SP + SP + "T const* updates,\n";
      for (auto &p : dynParamNames)
         op += SP + SP + SP + "std::size_t const " + p + ",\n";
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
             + "int64_t idx = indices[i * " + std::to_string(fK) + "u + " + std::to_string(j) + "u];\n";
         op += SP + SP + SP + SP + SP
             + "if (idx < 0) idx += " + fShapeData[j].GetVal() + ";\n";
         op += SP + SP + SP + SP + SP
             + "out_base += static_cast<std::size_t>(idx) * " + sz(stridesData[j].GetVal()) + ";\n";
         op += SP + SP + SP + SP + "}\n";
      }

      op += SP + SP + SP + SP + "for (std::size_t s = 0; s < sliceSize; ++s) {\n";
      op += SP + SP + SP + SP + SP + "std::size_t const out_idx = out_base + s;\n";
      op += SP + SP + SP + SP + SP + "std::size_t const upd_idx = i * sliceSize + s;\n";

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
      return SP + kname + " scatterNDKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeY.empty())
         throw std::runtime_error("SOFIE ScatterND: Generate_GPU_ALPAKA called before Initialize");

      std::stringstream out;
      out << "\n//------ SCATTERND_GPU_ALPAKA\n";

      out << SP << "alpaka::memcpy(queue, deviceBuf_" << fNY << ", deviceBuf_" << fNData << ");\n";

      out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(static_cast<Idx>(" << fNumOuter << "));\n";
      out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", scatterNDKernel_" << opName
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNIndices << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNUpdates << ")";
      for (auto &p : dynParamNames)
         out << ", static_cast<std::size_t>(" << p << ")";
      out << ", static_cast<Idx>(" << fNumOuter << ")"
          << ", static_cast<Idx>(" << fSliceSize << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";

      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_SCATTERND
