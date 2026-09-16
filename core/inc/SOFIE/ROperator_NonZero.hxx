#ifndef SOFIE_ROPERATOR_NONZERO
#define SOFIE_ROPERATOR_NONZERO

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"
#include "onnx_proto3.pb.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace SOFIE {

template <typename T>
class ROperator_NonZero final : public ROperator
{
private:
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;
   size_t fRank = 0;
   bool fIsInputDynamic = false;  // true if any input dim is only known at runtime

   // Valid only when !fIsInputDynamic: baked into the kernel as compile-time constants so
   // the static case keeps its fully-optimized launch. When the input
   // is dynamic, N/P/chunkSize become runtime kernel parameters instead.
   size_t fInputLength = 0;
   size_t fNumChunks = 1;
   size_t fChunkSize = 0;

   std::string fCountName;
   std::string fCountScratchName;  // persistent 1-element on-device int64 scratch tensor

public:
   ROperator_NonZero() {}

   ROperator_NonZero(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)), fCountName(fNY + "_nonzero_count"),
        fCountScratchName(fCountName + "_scratch")
   {
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> /*input*/) override {
      return {ETensorType::INT64};
   }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE NonZero Op Input Tensor " + fNX + " is not found in model");

      fShapeX = model.GetDimTensorShape(fNX);
      fRank = fShapeX.size();
      fIsInputDynamic = std::any_of(fShapeX.begin(), fShapeX.end(), [](const Dim &d) { return d.isParam; });

      if (!fIsInputDynamic) {
         // Split the compaction across up to 256 chunks (one GPU thread per chunk, single
         // block) instead of doing it on a single thread.
         std::vector<size_t> concreteShape(fRank);
         for (size_t i = 0; i < fRank; i++)
            concreteShape[i] = fShapeX[i].dim;
         fInputLength = ConvertShapeToLength(concreteShape);
         fNumChunks = std::min<size_t>(256, std::max<size_t>(fInputLength, size_t(1)));
         fChunkSize = (fInputLength + fNumChunks - 1) / fNumChunks;
      }

      fShapeY = {Dim{fRank}, Dim{fCountName, size_t(-1)}};
      model.AddDynamicTensor(fNY, ETensorType::INT64, fShapeY);
      model.AddIntermediateTensor(fCountScratchName, ETensorType::INT64, {Dim{1}});
      model.RegisterInternalDynamicParam(fCountName);
   }

   std::string Generate(std::string /*opName*/) override {
      std::stringstream out;
      out << "\n//------ NonZero\n";
      std::string lengthExpr = fIsInputDynamic ? ConvertDimShapeToLength(fShapeX) : std::to_string(fInputLength);

      out << SP << "size_t " << fCountName << " = 0;\n";
      out << SP << "for (size_t i = 0; i < " << lengthExpr << "; i++) {\n";
      out << SP << SP << "if (tensor_" << fNX << "[i] != static_cast<T>(0)) " << fCountName << "++;\n";
      out << SP << "}\n";
      out << SP << "if (" << fRank << " * " << fCountName << " > fTensor_" << fNY << ".size()) {\n";
      out << SP << SP << "fTensor_" << fNY << ".resize(" << fRank << " * " << fCountName << ");\n";
      out << SP << SP << "tensor_" << fNY << " = fTensor_" << fNY << ".data();\n";
      out << SP << "}\n";
      out << SP << "size_t nz = 0;\n";
      out << SP << "for (size_t i = 0; i < " << lengthExpr << "; i++) {\n";
      out << SP << SP << "if (tensor_" << fNX << "[i] == static_cast<T>(0)) continue;\n";

      for (size_t d = 0; d < fRank; d++) {
         std::string strideExpr = "1";
         for (size_t k = d + 1; k < fRank; k++)
            strideExpr = (strideExpr == "1") ? fShapeX[k].GetVal() : ("(" + strideExpr + " * " + fShapeX[k].GetVal() + ")");

         out << SP << SP << "tensor_" << fNY << "[" << d << " * " << fCountName << " + nz] = (i / " << strideExpr << ") % " << fShapeX[d].GetVal() << ";\n";
      }

      out << SP << SP << "nz++;\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/, const std::vector<std::string> &dynParamNames) override {
      std::string op;
      op += "\n//------ NonZero kernel\n";
      op += SP + "struct NonZeroKernel_" + fNY + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, int64_t* output, int64_t* count";
      if (fIsInputDynamic) {
         op += ", std::size_t const nzTotalLen, std::size_t const nzNumChunks, std::size_t const nzChunkSize";
         for (auto &p : dynParamNames)
            op += ", std::size_t const " + p;
      }
      op += ") const {\n";

      if (!fIsInputDynamic) {
         op += SP + SP + SP + "constexpr std::size_t nzTotalLen = " + std::to_string(fInputLength) + "u;\n";
         op += SP + SP + SP + "constexpr std::size_t nzNumChunks = " + std::to_string(fNumChunks) + "u;\n";
         op += SP + SP + SP + "constexpr std::size_t nzChunkSize = " + std::to_string(fChunkSize) + "u;\n";
      }

      op += SP + SP + SP + "auto& nzChunkOffset = alpaka::declareSharedVar<std::size_t[256], __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto const nzTid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "std::size_t const nzBegin = nzTid * nzChunkSize;\n";
      op += SP + SP + SP + "std::size_t const nzEnd = (nzBegin + nzChunkSize < nzTotalLen) ? nzBegin + nzChunkSize : nzTotalLen;\n";
      op += SP + SP + SP + "std::size_t nzChunkCount = 0;\n";
      op += SP + SP + SP + "for (std::size_t j = nzBegin; j < nzEnd; ++j) nzChunkCount += input[j] != static_cast<T>(0);\n";
      op += SP + SP + SP + "nzChunkOffset[nzTid] = nzChunkCount;\n";
      op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
      op += SP + SP + SP + "if (nzTid == 0) {\n";
      op += SP + SP + SP + SP + "std::size_t off = 0;\n";
      op += SP + SP + SP + SP + "for (std::size_t k = 0; k < nzNumChunks; ++k) { std::size_t c = nzChunkOffset[k]; nzChunkOffset[k] = off; off += c; }\n";
      op += SP + SP + SP + SP + "*count = static_cast<int64_t>(off);\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
      op += SP + SP + SP + "std::size_t nz = nzChunkOffset[nzTid];\n";
      op += SP + SP + SP + "std::size_t const total = static_cast<std::size_t>(*count);\n";
      op += SP + SP + SP + "for (std::size_t j = nzBegin; j < nzEnd; ++j) {\n";
      op += SP + SP + SP + SP + "if (input[j] == static_cast<T>(0)) continue;\n";

      for (size_t d = 0; d < fRank; d++) {
         std::string strideExpr = "1";
         for (size_t k = d + 1; k < fRank; k++)
            strideExpr = (strideExpr == "1") ? fShapeX[k].GetVal() : ("(" + strideExpr + " * " + fShapeX[k].GetVal() + ")");

         op += SP + SP + SP + SP + "output[" + std::to_string(d) + "u * total + nz] = static_cast<int64_t>((j / " + strideExpr + ") % " + fShapeX[d].GetVal() + ");\n";
      }

      op += SP + SP + SP + SP + "++nz;\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "NonZeroKernel_" + fNY + " nonZeroKernel_" + fNY + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string /*opName*/, const std::vector<std::string> &dynParamNames) override {
      std::stringstream out;
      out << "\n//------ NonZero_GPU_ALPAKA\n";

      std::string countPtr = "alpaka::getPtrNative(deviceBuf_" + fCountScratchName + ")";

      if (fIsInputDynamic) {
         std::string nExpr = ConvertDimShapeToLength(fShapeX);
         out << SP << "std::size_t const nzN_" << fNY << " = static_cast<std::size_t>(" << nExpr << ");\n";
         out << SP << "std::size_t const nzP_" << fNY << " = std::min<std::size_t>(256u, std::max<std::size_t>(nzN_" << fNY << ", std::size_t(1)));\n";
         out << SP << "std::size_t const nzChunk_" << fNY << " = (nzN_" << fNY << " + nzP_" << fNY << " - 1) / nzP_" << fNY << ";\n";
         out << SP << "auto const workDivNonZero_" << fNY << " = sofie_workdiv(Vec::all(Idx{nzP_" << fNY << "}), Idx{nzP_" << fNY << "});\n";
         out << SP << "auto taskNonZero_" << fNY << " = alpaka::createTaskKernel<Acc>(workDivNonZero_" << fNY << ", nonZeroKernel_" << fNY
             << ", alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), " << countPtr
             << ", nzN_" << fNY << ", nzP_" << fNY << ", nzChunk_" << fNY;
         for (auto &p : dynParamNames)
            out << ", static_cast<std::size_t>(" << p << ")";
         out << ");\n";
      } else {
         out << SP << "auto const workDivNonZero_" << fNY << " = sofie_workdiv(Vec::all(Idx{" << fNumChunks << "}), Idx{" << fNumChunks << "});\n";
         out << SP << "auto taskNonZero_" << fNY << " = alpaka::createTaskKernel<Acc>(workDivNonZero_" << fNY << ", nonZeroKernel_" << fNY
             << ", alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), " << countPtr << ");\n";
      }
      out << SP << "alpaka::enqueue(queue, taskNonZero_" << fNY << ");\n";

      out << SP << "auto nonZeroCountHost_" << fNY << " = alpaka::allocBuf<int64_t, Idx>(hostAcc, Ext1D::all(Idx{1}));\n";
      out << SP << "alpaka::memcpy(queue, nonZeroCountHost_" << fNY << ", deviceBuf_" << fCountScratchName << ");\n";
      out << SP << "alpaka::wait(queue);\n";
      out << SP << "size_t " << fCountName << " = static_cast<size_t>(*alpaka::getPtrNative(nonZeroCountHost_" << fNY << "));\n";
      return out.str();
   }

   std::string GenerateInitCode_GPU_ALPAKA() override {
      std::string maxLen = fIsInputDynamic
         ? ("static_cast<Idx>(" + std::to_string(fRank) + "u * (" + ConvertDimShapeToLength(fShapeX) + "))")
         : (std::to_string(fRank * fInputLength) + "u");
      return SP + "deviceBuf_" + fNY + " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{" + maxLen + "}));\n";
   }
};


} // namespace SOFIE

#endif // SOFIE_ROPERATOR_NONZERO
