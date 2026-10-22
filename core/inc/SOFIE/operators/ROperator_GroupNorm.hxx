#ifndef SOFIE_ROPERATOR_GROUPNORM
#define SOFIE_ROPERATOR_GROUPNORM

#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/utils/SOFIE_common.hxx"
#include <sstream>
#include <string>

namespace SOFIE {

template <typename T>
class ROperator_GroupNorm : public ROperator_ALPAKA {
private:
   int fNumGroups;
   float fAttrEpsilon;

   std::string fNX;
   std::string fNScale;
   std::string fNBias;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

   size_t fRank;
   std::string fType;

   std::string fN;
   std::string fC;
   std::string fSpatial;
   std::string fGsize;

public:
   ROperator_GroupNorm() {}

   ROperator_GroupNorm(int numGroups, float epsilon,
                       const std::string &nameX,
                       const std::string &nameScale,
                       const std::string &nameBias,
                       const std::string &nameY)
      : fNumGroups(numGroups), fAttrEpsilon(epsilon),
        fNX(UTILITY::Clean_name(nameX)),
        fNScale(UTILITY::Clean_name(nameScale)),
        fNBias(UTILITY::Clean_name(nameBias)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::GROUPNORM;
      fInputTensorNames  = { fNX, fNScale };
      if (!fNBias.empty()) fInputTensorNames.emplace_back(fNBias);
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE GroupNorm: input tensor " + fNX + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNScale))
         throw std::runtime_error("SOFIE GroupNorm: scale tensor " + fNScale + " not found");
      if (!fNBias.empty() && !model.CheckIfTensorAlreadyExist(fNBias))
         throw std::runtime_error("SOFIE GroupNorm: bias tensor " + fNBias + " not found");

      fShapeX = model.GetDimTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShapeX.empty();
      if (model.IsStridedInputTensor(fNScale) || (!fNBias.empty() && model.IsStridedInputTensor(fNBias)))
         throw std::runtime_error("SOFIE GroupNorm - strided input is only supported for the data tensor X");
      fShapeY = fShapeX;
      fRank   = fShapeX.size();
      fType   = ConvertTypeToString(model.GetTensorType(fNX));

      if (fRank < 2)
         throw std::runtime_error("SOFIE GroupNorm: input must have rank >= 2 (N, C, ...)");

      fN = fShapeX[0].GetVal();
      fC = fShapeX[1].GetVal();

      if (fRank > 2) {
         std::vector<Dim> spatDims(fShapeX.begin() + 2, fShapeX.end());
         fSpatial = ConvertDimShapeToLength(spatDims);
      } else {
         fSpatial = "1";
      }

      fGsize = "(" + fC + " / " + std::to_string(fNumGroups) + ")";

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      model.AddNeededStdLib("cmath");
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE GroupNorm " + opName + " called to Generate without being initialized");

      std::string G  = std::to_string(fNumGroups);
      std::string gs = fGsize;
      std::string eps = std::to_string(fAttrEpsilon);

      std::stringstream out;
      out << "\n//---- GroupNorm operator " << opName << "\n";
      out << "{\n";
      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      auto readX = [&](const std::string &index) {
         return fHasStridedInput ? "tensor_" + fNX + "[xoff_" + opName + "(" + index + ")]"
                                 : "tensor_" + fNX + "[" + index + "]";
      };
      if (fHasStridedInput)
         out << GenerateStridedOffsetLambda(opName, fNX, fShapeX);
      out << "const size_t gn_N = " << fN << ";\n";
      out << "const size_t gn_C = " << fC << ";\n";
      out << "const size_t gn_spatial = " << fSpatial << ";\n";
      out << "const size_t gn_G = " << G << ";\n";
      out << "const size_t gn_gs = gn_C / gn_G;\n";
      out << "const " << fType << " gn_eps = " << eps << ";\n\n";

      out << "for (size_t n = 0; n < gn_N; ++n) {\n";
      out << "for (size_t g = 0; g < gn_G; ++g) {\n";
      out << "// mean over (channels in group) x spatial\n";
      out << fType << " gn_mean = 0;\n";
      out << "for (size_t ci = 0; ci < gn_gs; ++ci) {\n";
      out << "size_t c = g * gn_gs + ci;\n";
      out << "for (size_t s = 0; s < gn_spatial; ++s)\n";
      out << "gn_mean += " << readX("n * gn_C * gn_spatial + c * gn_spatial + s") << ";\n";
      out << "}\n";
      out << "gn_mean /= " << fType << "(gn_gs * gn_spatial);\n\n";

      out << "// variance\n";
      out << fType << " gn_var = 0;\n";
      out << "for (size_t ci = 0; ci < gn_gs; ++ci) {\n";
      out << "size_t c = g * gn_gs + ci;\n";
      out << "for (size_t s = 0; s < gn_spatial; ++s) {\n";
      out << fType << " d = " << readX("n * gn_C * gn_spatial + c * gn_spatial + s") << " - gn_mean;\n";
      out << "gn_var += d * d;\n";
      out << "}\n";
      out << "}\n";
      out << "gn_var /= " << fType << "(gn_gs * gn_spatial);\n";
      out << fType << " gn_inv_std = 1 / std::sqrt(gn_var + gn_eps);\n\n";

      out << "// normalize + scale + bias\n";
      out << "for (size_t ci = 0; ci < gn_gs; ++ci) {\n";
      out << "size_t c = g * gn_gs + ci;\n";
      out << "for (size_t s = 0; s < gn_spatial; ++s) {\n";
      out << "size_t idx = n * gn_C * gn_spatial + c * gn_spatial + s;\n";
      out << fType << " v = (" << readX("idx") << " - gn_mean) * gn_inv_std;\n";
      out << "v *= tensor_" << fNScale << "[c];\n";
      if (!fNBias.empty())
         out << "v += tensor_" << fNBias << "[c];\n";
      out << "tensor_" << fNY << "[idx] = v;\n";
      out << "}\n";
      out << "}\n";
      out << "}\n";
      out << "}\n";
      out << "}\n";

      return out.str();
   }

   // Threads cooperating on one (n,g) group. A group's gn_gs*gn_spatial elements are
   // contiguous in memory (c = g*gn_gs+ci, address = n*C*spatial + c*spatial + s =
   // (row base) + ci*spatial + s, and ci*spatial+s enumerates [0, gn_gs*spatial)
   // contiguously)
   static constexpr size_t kBlockSize = 256;

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE GroupNorm GPU kernel called without initialization");

      std::string bs  = std::to_string(kBlockSize);
      std::string eps = std::to_string(fAttrEpsilon);
      std::string kname = "GroupNormKernel_" + opName;

      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      auto rdX = [&](const std::string &index) {
         return fHasStridedInput ? StridedKernelRead("X", "layoutX", index) : "X[" + index + "]";
      };
      std::string op;
      op  = "\n//------ GROUPNORM_KERNEL_ALPAKA\n";
      op += "struct " + kname + " {\n";
      op += "template<typename TAcc, typename T>\n";
      op += "ALPAKA_FN_ACC void operator()(\n";
      op += "TAcc const& acc,\n";
      op += "T const* __restrict__ X,\n";
      op += "T const* __restrict__ scale,\n";
      if (!fNBias.empty())
         op += "T const* __restrict__ bias,\n";
      op += "T* __restrict__ Y,\n";
      op += "std::size_t const gn_C,\n";
      op += "std::size_t const gn_spatial,\n";
      op += "std::size_t const gn_G,\n";
      if (fHasStridedInput)
         op += "sofie_strided_layout<" + std::to_string(fRank) + "> const layoutX,\n";
      op += "std::size_t const numRows) const {\n\n";

      // one block per (n,g) row - a block-shared reduction buffer, reused across the
      // mean and variance passes
      op += "auto& sred = alpaka::declareSharedVar<T[" + bs + "], __COUNTER__>(acc);\n";
      op += "auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
      op += "auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      op += "if (row >= numRows) return;\n\n";

      op += "std::size_t const gn_gs   = gn_C / gn_G;\n";
      op += "std::size_t const rowLen  = gn_gs * gn_spatial;\n";
      op += "std::size_t const g       = row % gn_G;\n";
      op += "std::size_t const base    = row * rowLen;\n";
      op += "T const gn_eps = static_cast<T>(" + eps + ");\n\n";

      auto treeReduce = [&]() {
         std::string s;
         size_t half = kBlockSize / 2;
         while (half > 0) {
            s += "if (tid < " + std::to_string(half) + "u) sred[tid] += sred[tid + " + std::to_string(half) + "u];\n";
            s += "alpaka::syncBlockThreads(acc);\n";
            half >>= 1;
         }
         return s;
      };

      op += "// mean: per-thread partial sum, then block-shared tree reduction\n";
      op += "T sum = static_cast<T>(0);\n";
      op += "for (std::size_t l = tid; l < rowLen; l += " + bs + "u) sum += " + rdX("base + l") + ";\n";
      op += "sred[tid] = sum;\n";
      op += "alpaka::syncBlockThreads(acc);\n";
      op += treeReduce();
      op += "T const mean = sred[0] / static_cast<T>(rowLen);\n";
      op += "alpaka::syncBlockThreads(acc);\n\n"; // all reads of sred[0] done before sred is reused below

      op += "// variance: same pattern, from deviations around mean\n";
      op += "T devsq = static_cast<T>(0);\n";
      op += "for (std::size_t l = tid; l < rowLen; l += " + bs + "u) {\n";
      op += "T d = " + rdX("base + l") + " - mean;\n";
      op += "devsq += d * d;\n";
      op += "}\n";
      op += "sred[tid] = devsq;\n";
      op += "alpaka::syncBlockThreads(acc);\n";
      op += treeReduce();
      op += "T const var = sred[0] / static_cast<T>(rowLen);\n";
      op += "T const inv_std = static_cast<T>(1) / sqrt(acc, var + gn_eps);\n\n";

      op += "// normalize + scale + bias\n";
      op += "for (std::size_t l = tid; l < rowLen; l += " + bs + "u) {\n";
      op += "std::size_t idx = base + l;\n";
      op += "std::size_t c = g * gn_gs + l / gn_spatial;\n";
      op += "T v = (" + rdX("idx") + " - mean) * inv_std * scale[c];\n";
      if (!fNBias.empty())
         op += "v += bias[c];\n";
      op += "Y[idx] = v;\n";
      op += "}\n";

      op += "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "GroupNormKernel_" + opName;
      return kname + " groupNormKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE GroupNorm GPU dispatch called without initialization");

      std::string G = std::to_string(fNumGroups);
      std::string numRows = "static_cast<Idx>((" + fN + ") * " + G + ")";
      std::string args =
         "alpaka::getPtrNative(deviceBuf_" + fNX + "), "
         "alpaka::getPtrNative(deviceBuf_" + fNScale + "), ";
      if (!fNBias.empty())
         args += "alpaka::getPtrNative(deviceBuf_" + fNBias + "), ";
      args += "alpaka::getPtrNative(deviceBuf_" + fNY + "), "
              "static_cast<Idx>(" + fC + "), "
              "static_cast<Idx>(" + fSpatial + "), "
              "static_cast<Idx>(" + G + "), "
              + (fHasStridedInput ? "layout_" + opName + "_X, " : std::string())
              + numRows;

      std::stringstream out;
      out << "\n//------ GROUPNORM_GPU_ALPAKA\n";
      if (fHasStridedInput)
         out << GenerateStridedBroadcastLayout(opName + "_X", fNX, fShapeX, fShapeX.size(), fShapeX);
      out << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << opName << "(\n";
      out << "Vec::all(" << numRows << "),\n";
      out << "Vec::all(Idx{" << kBlockSize << "u}),\n";
      out << "Vec::all(Idx{1u}));\n";
      out << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", groupNormKernel_" << opName << ", " << args << ");\n";
      out << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_GROUPNORM
