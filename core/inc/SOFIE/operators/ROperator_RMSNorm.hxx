#ifndef SOFIE_ROPERATOR_RMSNORM
#define SOFIE_ROPERATOR_RMSNORM

#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/model/RModel.hxx"
#include "SOFIE/utils/SOFIE_common.hxx"
#include <sstream>
#include <string>

namespace SOFIE {

template <typename T>
class ROperator_RMSNorm : public ROperator_ALPAKA {
private:
   int fAttrAxis;
   float fAttrEpsilon;

   std::string fNX;
   std::string fNScale;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeScale;
   std::vector<Dim> fShapeY;

   size_t fAxis;
   size_t fRank;

   std::string fLength;
   std::string fNormLength;
   std::string fAxesLength;
   std::string fType;

   std::vector<Dim> fNormShape;
   std::vector<Dim> fAxesShape;

public:
   ROperator_RMSNorm() {}

   ROperator_RMSNorm(int axis, float epsilon,
                     const std::string &nameX,
                     const std::string &nameScale,
                     const std::string &nameY)
      : fAttrAxis(axis), fAttrEpsilon(epsilon),
        fNX(UTILITY::Clean_name(nameX)),
        fNScale(UTILITY::Clean_name(nameScale)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::RMSNORM;
      fInputTensorNames  = { fNX, fNScale };
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE RMSNorm: input tensor " + fNX + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNScale))
         throw std::runtime_error("SOFIE RMSNorm: scale tensor " + fNScale + " not found");

      fShapeX   = model.GetDimTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShapeX.empty();
      if (model.IsStridedInputTensor(fNScale))
         throw std::runtime_error("SOFIE RMSNorm - strided input is only supported for the data tensor X");
      fShapeY   = fShapeX;
      fShapeScale = model.GetDimTensorShape(fNScale);
      fType     = ConvertTypeToString(model.GetTensorType(fNX));
      fRank     = fShapeX.size();
      fAxis     = (fAttrAxis < 0) ? fRank + fAttrAxis : static_cast<size_t>(fAttrAxis);

      fAxesShape = std::vector<Dim>(fShapeX.begin(), fShapeX.begin() + fAxis);
      fNormShape = std::vector<Dim>(fShapeX.begin() + fAxis, fShapeX.end());
      fAxesLength = ConvertDimShapeToLength(fAxesShape);
      fNormLength = ConvertDimShapeToLength(fNormShape);
      fLength     = ConvertDimShapeToLength(fShapeX);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      model.AddNeededStdLib("cmath");
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE RMSNorm " + opName + " called to Generate without being initialized");

      std::vector<std::string> shape(fRank);
      for (size_t i = 0; i < fRank; ++i) shape[i] = fShapeX[i].GetVal();
      auto strides = UTILITY::ComputeStrideFromShape(fShapeX);

      std::stringstream out;
      out << "\n//---- RMSNorm operator " << opName << "\n";
      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      const std::string readX = fHasStridedInput ? "tensor_" + fNX + "[xoff_" + opName + "(idx)]" : "tensor_" + fNX + "[idx]";
      if (fHasStridedInput)
         out << GenerateStridedOffsetLambda(opName, fNX, fShapeX);

      for (size_t i = 0; i < fAxis; ++i)
         out << "for (size_t a_" << i << " = 0; a_" << i << " < " << shape[i] << "; ++a_" << i << ") {\n";

      out << "size_t row_base = ";
      if (fAxis == 0) {
         out << "0;\n";
      } else {
         for (size_t i = 0; i < fAxis; ++i) {
            out << "a_" << i << " * " << strides[i].GetVal();
            if (i + 1 < fAxis) out << " + ";
         }
         out << ";\n";
      }

      out << fType << " rms_sum = 0;\n";
      for (size_t j = fAxis; j < fRank; ++j)
         out << "for (size_t n_" << j << " = 0; n_" << j << " < " << shape[j] << "; ++n_" << j << ") {\n";
      out << "size_t idx = row_base";
      for (size_t j = fAxis; j < fRank; ++j) out << " + n_" << j << " * " << strides[j].GetVal();
      out << ";\n";
      out << fType << " v = " << readX << ";\n";
      out << "rms_sum += v * v;\n";
      for (size_t j = fAxis; j < fRank; ++j) out << "}\n";

      out << fType << " inv_rms = static_cast<" << fType << ">(1) / std::sqrt(rms_sum / "
          << fType << "(" << fNormLength << ") + " << fAttrEpsilon << ");\n";

      for (size_t j = fAxis; j < fRank; ++j)
         out << "for (size_t n_" << j << " = 0; n_" << j << " < " << shape[j] << "; ++n_" << j << ") {\n";
      out << "size_t idx = row_base";
      for (size_t j = fAxis; j < fRank; ++j) out << " + n_" << j << " * " << strides[j].GetVal();
      out << ";\n";
      out << "size_t s_idx = ";
      {
         auto scaleStrides = UTILITY::ComputeStrideFromShape(fNormShape);
         bool first = true;
         for (size_t j = fAxis; j < fRank; ++j) {
            size_t ji = j - fAxis;
            if (!first) out << " + ";
            out << "n_" << j << " * " << scaleStrides[ji].GetVal();
            first = false;
         }
         if (first) out << "0";
      }
      out << ";\n";
      out << "tensor_" << fNY << "[idx] = tensor_" << fNScale << "[s_idx] * " << readX << " * inv_rms;\n";
      for (size_t j = fAxis; j < fRank; ++j) out << "}\n";
      for (size_t i = 0; i < fAxis; ++i) out << "}\n";

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE RMSNorm GPU kernel called without initialization");

      size_t normLenVal = 0;
      bool canParallel = false;
      try { normLenVal = std::stoul(fNormLength); canParallel = (normLenVal > 0 && normLenVal <= 1024); }
      catch (...) {}

      size_t blockSize = 1;
      if (canParallel) { while (blockSize < normLenVal) blockSize <<= 1; }

      std::vector<std::string> shape(fRank);
      for (size_t i = 0; i < fRank; ++i) shape[i] = fShapeX[i].GetVal();

      // Wraps a size expression (a literal, or a symbolic dim name/expression from
      // Dim::GetVal()) as an explicit std::size_t cast.
      auto sz = [](const std::string &e) { return "static_cast<std::size_t>(" + e + ")"; };

      auto strides      = UTILITY::ComputeStrideFromShape(fShapeX);
      auto axesStrides  = UTILITY::ComputeStrideFromShape(fAxesShape);
      auto normStrides  = UTILITY::ComputeStrideFromShape(fNormShape);

      std::string kname = "RMSNormKernel_" + opName;
      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      auto rdX = [&](const std::string &index) {
         return fHasStridedInput ? StridedKernelRead("X", "layoutX", index) : "X[" + index + "]";
      };
      std::string op;
      op  = "\n//------ RMSNORM_KERNEL_ALPAKA\n";
      op += "struct " + kname + " {\n";
      op += "template<typename TAcc, typename T>\n";
      op += "ALPAKA_FN_ACC void operator()(\n";
      op += "TAcc const& acc,\n";
      op += "T const* __restrict__ X,\n";
      op += "T const* __restrict__ scale,\n";
      op += "T* __restrict__ Y";
      for (auto &p : dynParamNames)
         op += ",\nstd::size_t const " + p;
      if (fHasStridedInput)
         op += ",\nsofie_strided_layout<" + std::to_string(fRank) + "> const layoutX";
      op += ",\nstd::size_t const axesLength) const {\n\n";

      std::string eps = std::to_string(fAttrEpsilon);
      std::string nl  = fNormLength;

      if (canParallel) {
         std::string bs = std::to_string(blockSize);

         op += "// Block-parallel RMSNorm: one block per row, " + bs + " threads, " + nl + " active.\n";
         op += "auto& shmem = alpaka::declareSharedVar<T[" + bs + "], __COUNTER__>(acc);\n";
         op += "auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
         op += "auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
         op += "if (row >= axesLength) return;\n\n";

         // Decompose row → axis coords
         if (fAxis > 0) {
            for (size_t i = 0; i < fAxis; ++i) {
               op += "std::size_t const axis_" + std::to_string(i)
                  + " = (row / " + sz(axesStrides[i].GetVal()) + ") % " + sz(shape[i]) + ";\n";
            }
            op += "\n";
         }

         // row_base
         op += "std::size_t const row_base =\n";
         if (fAxis == 0) {
            op += "0u;\n\n";
         } else {
            for (size_t i = 0; i < fAxis; ++i) {
               op += "axis_" + std::to_string(i) + " * " + sz(strides[i].GetVal());
               op += (i + 1 < fAxis) ? " +\n" : ";\n\n";
            }
         }

         op += "bool const in_range = (tid < " + sz(nl) + ");\n";
         op += "std::size_t norm_offset = 0u;\n";
         op += "std::size_t s_offset = 0u;\n";
         op += "if (in_range) {\n";
         if (fRank - fAxis == 1) {
            op += "norm_offset = tid * " + sz(strides[fAxis].GetVal()) + ";\n";
            op += "s_offset    = tid * " + sz(normStrides[0].GetVal())  + ";\n";
         } else {
            op += "std::size_t rem = tid;\n";
            for (size_t j = fAxis; j < fRank; ++j) {
               size_t ji = j - fAxis;
               op += "{ std::size_t nj = rem / " + sz(normStrides[ji].GetVal())
                  + "; rem %= " + sz(normStrides[ji].GetVal())
                  + "; norm_offset += nj * " + sz(strides[j].GetVal())
                  + "; s_offset += nj * " + sz(normStrides[ji].GetVal()) + "; }\n";
            }
         }
         op += "}\n\n";

         op += "std::size_t const norm_idx = row_base + norm_offset;\n";
         op += "T const val = in_range ? " + rdX("norm_idx") + " : static_cast<T>(0);\n\n";

         // Pass 1: sum of squares
         op += "// Pass 1: sum(X^2) via shared-memory tree reduction\n";
         op += "shmem[tid] = val * val;\n";
         op += "alpaka::syncBlockThreads(acc);\n";
         size_t half = blockSize / 2;
         while (half > 0) {
            op += "if (tid < " + std::to_string(half) + "u) shmem[tid] += shmem[tid + " + std::to_string(half) + "u];\n";
            op += "alpaka::syncBlockThreads(acc);\n";
            half >>= 1;
         }
         op += "T const inv_rms = static_cast<T>(1) / sqrt(acc,"
               " shmem[0] / static_cast<T>(" + nl + ") + static_cast<T>(" + eps + "));\n\n";

         op += "// Pass 2: Y = scale * X * inv_rms\n";
         op += "if (in_range) {\n";
         op += "Y[norm_idx] = scale[s_offset] * val * inv_rms;\n";
         op += "}\n";

      } else {
         op += "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += "if (global_thread_idx >= axesLength) return;\n";
         op += "auto const grid_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";
         op += "for (std::size_t row = global_thread_idx; row < axesLength; row += grid_extent) {\n\n";

         if (fAxis > 0) {
            for (size_t i = 0; i < fAxis; ++i) {
               op += "std::size_t const axis_" + std::to_string(i)
                  + " = (row / " + sz(axesStrides[i].GetVal()) + ") % " + sz(shape[i]) + ";\n";
            }
            op += "\n";
         }

         op += "std::size_t const row_base =\n";
         if (fAxis == 0) {
            op += "0u;\n\n";
         } else {
            for (size_t i = 0; i < fAxis; ++i) {
               op += "axis_" + std::to_string(i) + " * " + sz(strides[i].GetVal());
               op += (i + 1 < fAxis) ? " +\n" : ";\n\n";
            }
         }

         op += "T rms_sum = static_cast<T>(0);\n";
         for (size_t j = fAxis; j < fRank; ++j)
            op += "for (std::size_t n_" + std::to_string(j)
               + " = 0; n_" + std::to_string(j) + " < " + sz(shape[j]) + "; ++n_" + std::to_string(j) + ") {\n";
         op += "std::size_t const idx = row_base";
         for (size_t j = fAxis; j < fRank; ++j) op += " + n_" + std::to_string(j) + " * " + sz(strides[j].GetVal());
         op += ";\n";
         op += "T v = " + rdX("idx") + "; rms_sum += v * v;\n";
         for (size_t j = fAxis; j < fRank; ++j) op += "}\n";
         op += "T const inv_rms = static_cast<T>(1) / sqrt(acc,"
               " rms_sum / static_cast<T>(" + nl + ") + static_cast<T>(" + eps + "));\n\n";

         for (size_t j = fAxis; j < fRank; ++j)
            op += "for (std::size_t n_" + std::to_string(j)
               + " = 0; n_" + std::to_string(j) + " < " + sz(shape[j]) + "; ++n_" + std::to_string(j) + ") {\n";
         op += "std::size_t const idx = row_base";
         for (size_t j = fAxis; j < fRank; ++j) op += " + n_" + std::to_string(j) + " * " + sz(strides[j].GetVal());
         op += ";\n";
         op += "std::size_t s_idx = ";
         {
            bool first = true;
            for (size_t j = fAxis; j < fRank; ++j) {
               if (!first) op += " + ";
               op += "n_" + std::to_string(j) + " * " + sz(normStrides[j - fAxis].GetVal());
               first = false;
            }
            if (fRank == fAxis) op += "0u";
         }
         op += ";\n";
         op += "Y[idx] = scale[s_idx] * " + rdX("idx") + " * inv_rms;\n";
         for (size_t j = fAxis; j < fRank; ++j) op += "}\n";

         op += "}\n";
      }

      op += "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "RMSNormKernel_" + opName;
      return kname + " rmsNormKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE RMSNorm GPU dispatch called without initialization");

      size_t normLenVal = 0;
      bool canParallel = false;
      try { normLenVal = std::stoul(fNormLength); canParallel = (normLenVal > 0 && normLenVal <= 1024); }
      catch (...) {}
      size_t blockSize = 1;
      if (canParallel) { while (blockSize < normLenVal) blockSize <<= 1; }

      std::string args =
         "alpaka::getPtrNative(deviceBuf_" + fNX + "), "
         "alpaka::getPtrNative(deviceBuf_" + fNScale + "), "
         "alpaka::getPtrNative(deviceBuf_" + fNY + ")";
      for (auto &p : dynParamNames)
         args += ", static_cast<std::size_t>(" + p + ")";
      if (fHasStridedInput)
         args += ", layout_" + opName + "_X";
      args += ", static_cast<Idx>(" + fAxesLength + ")";

      std::stringstream out;
      out << "\n//------ RMSNORM_GPU_ALPAKA\n";
      if (fHasStridedInput)
         out << GenerateStridedBroadcastLayout(opName + "_X", fNX, fShapeX, fShapeX.size(), fShapeX);
      if (canParallel) {
         out << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << opName << "(\n";
         out << "Vec::all(Idx{" << fAxesLength << "}),\n";
         out << "Vec::all(Idx{" << blockSize << "u}),\n";
         out << "Vec::all(Idx{1u}));\n";
      } else {
         out << "auto const elementsPerGrid_" << opName << " = Vec::all(Idx{" << fAxesLength << "});\n";
         out << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      }
      out << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", rmsNormKernel_" << opName << ", " << args << ");\n";
      out << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_RMSNORM
