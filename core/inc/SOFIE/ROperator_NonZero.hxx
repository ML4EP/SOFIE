#ifndef SOFIE_ROPERATOR_NONZERO
#define SOFIE_ROPERATOR_NONZERO

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <algorithm>
#include <sstream>
namespace SOFIE{

template<class T>
class ROperator_NonZero final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::string fNonZeroParam;
   bool fDeclaresParam = false;
   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;
   size_t fRank = 0;
   bool fIsInputDynamic = false;
   size_t fInputLength = 0;
   size_t fNumChunks = 1;
   size_t fChunkSize = 0;
   std::string fCountName;
   std::string fCountScratchName;

public:
   ROperator_NonZero(){}
   ROperator_NonZero(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }



   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
         throw std::runtime_error("SOFIE NonZero Op Input Tensor " + fNX + " is not found in model");
      }


      if (model.IsConstantTensor(fNX)) {
         T * data = static_cast<T*>(model.GetInitializedTensorData(fNX).get());
         auto shapeX = model.GetTensorShape(fNX);
         std::vector<size_t> shapeY(2);
         shapeY[0] = shapeX.size();
         auto length = ConvertShapeToLength(shapeX);
         auto strides = UTILITY::ComputeStrideFromShape(shapeX);
         std::vector<std::vector<int64_t>> nonzero_indices;
         for (size_t i = 0; i < length; i++) {
            if (data[i] != 0) {
               size_t flat_index = i;
               std::vector<int64_t> indices(shapeX.size());
               for (size_t j = 0; j < shapeX.size(); ++j) {
                  indices[j] = flat_index / strides[j];
                  flat_index %= strides[j];
               }
               nonzero_indices.emplace_back(indices);
            }
         }
         shapeY[1] = nonzero_indices.size();
         std::vector<int64_t> dataY(shapeY[0]* shapeY[1]);
         size_t k = 0;
         for (size_t i = 0; i < shapeY[0]; i++) {
            for (size_t j = 0; j < shapeY[1]; j++) {
               dataY[k] = nonzero_indices[j][i];
               k++;
            }
         }
         if (dataY.empty()) {
            dataY.resize(1);
            shapeY.clear();
         }

         model.AddConstantTensor(fNY, shapeY, dataY);
         if (model.Verbose()) {
            std::cout << "NonZero : " << fNX << " -> " << fNY << " " << ConvertShapeToString(shapeY)
                     << " : " << ConvertValuesToString(dataY) << std::endl;
         }
         fIsOutputConstant = true;

      } else {

         fShapeX = model.GetDimTensorShape(fNX);

         fShapeY.resize(2);
         fShapeY[0] = fShapeX.size();

         fNonZeroParam = "v_NonZero_" + fNX;
         fShapeY[1] = Dim{fNonZeroParam, static_cast<size_t>(-1)};

         if (!model.IsComputedShapeParam(fNonZeroParam)) {
            fDeclaresParam = true;
            auto inputLength = ConvertDimShapeToLength(fShapeX);
            std::string codeDecl = SP + "size_t " + fNonZeroParam + " = " + inputLength + ";\n";
            codeDecl += SP + "fV_NonZero_" + fNX + " = " + fNonZeroParam + ";\n";
            model.AddExtraCodeForDimShapes(codeDecl);
            model.AddComputedShapeParam(fNonZeroParam);
         }

         model.AddIntermediateTensor(fNY, ETensorType::INT64, fShapeY);
         fRank = fShapeX.size();
         fCountName = fNonZeroParam;
         fCountScratchName = fNY + "_nonzero_count_scratch";
         fIsInputDynamic = std::any_of(fShapeX.begin(), fShapeX.end(), [](const Dim &d) { return d.isParam; });
         if (!fIsInputDynamic) {
            fInputLength = ConvertShapeToLength(ConvertShapeToInt(fShapeX));
            fNumChunks = std::min<size_t>(256, std::max<size_t>(fInputLength, size_t(1)));
            fChunkSize = (fInputLength + fNumChunks - 1) / fNumChunks;
         }
         model.AddIntermediateTensor(fCountScratchName, ETensorType::INT64, {Dim{1}});
         model.RegisterInternalDynamicParam(fCountName);
         if (model.Verbose()) {
            std::cout << "NonZero : " << fNX << " -> " << fNY << " " << ConvertDimShapeToString(fShapeY) << std::endl;
         }
      }
   }

   std::string GenerateSessionMembersCode(std::string /*opName*/) override {
      if (fIsOutputConstant || !fDeclaresParam)
         return "";
      std::stringstream out;
      out << SP << "size_t fV_NonZero_" << fNX << " = 0;\n";
      return out.str();
   }


   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) {
         return "";
      }
      opName = "op_" + opName;
      if (fShapeX.empty()) {
         throw std::runtime_error("SOFIE Operator NonZero called to Generate without being initialized first");
      }
      std::stringstream out;
      auto intShapeX = ConvertShapeToInt(fShapeX);
      size_t inputLength = 0;
      std::string s_inputLength = ConvertDimShapeToLength(fShapeX);
      if (!intShapeX.empty())
         inputLength = ConvertShapeToLength(intShapeX);

      size_t dims = fShapeX.size();
      out << "\n//------ NonZero  -> " << ConvertDimShapeToString(fShapeY) << "\n";

      std::string vnonzero = fDeclaresParam ? fNonZeroParam : ("nonzero_count_" + opName);

      out << SP << "size_t offset_" << opName << " = 0;\n";
      out << SP << "size_t " << vnonzero << " = 0;\n";
      for (size_t j = 0; j < dims; j++) {
         std::string index = "i_" + std::to_string(j);
         for (size_t k = 0; k <= j; k++) out << SP;
         out << "for (size_t " << index << " = 0; " << index << " < " << fShapeX[j] << "; " << index << "++) {\n";
      }
      for (size_t k = 0; k <= dims; k++) out << SP;
      out << "if (tensor_" << fNX << "[offset_" << opName << "++]) {\n";
      for (size_t j = 0; j < dims; j++) {
         for (size_t k = 0; k <= dims+1; k++) out << SP;
         out << "tensor_" << fNY << "[";
         if (j > 0) {
            if (inputLength > 0) {
               out << inputLength * j;
            } else {
               out << s_inputLength;
               if (j > 1) out << " * " << j;
            }
            out << " + ";
         }
         out << vnonzero << "] = i_" << j << ";\n";
      }
      for (size_t k = 0; k <= dims+1; k++) out << SP;
      out << vnonzero << "++;\n";
      for (size_t k = 0; k <= dims; k++) out << SP;
      out << "}\n";
      for (size_t j = dims; j > 0; j--) {
         for (size_t k = 0; k <j; k++) out << SP;
         out << "}\n";
      }
      out << SP << "if (" << vnonzero << " < " << s_inputLength << "){\n";
      for (size_t j = 1; j < dims; j++) {
         out << SP << SP << "std::copy(tensor_" << fNY;
         if (j>0) out << " + " << s_inputLength;
         if (j>1) out << " * " << j;
         out << ", tensor_" << fNY;
         if (j>0) out << " + " << s_inputLength;
         if (j>1) out << " * " << j;
         out << " + " << vnonzero << ", tensor_" <<  fNY;
         if (j>0) out << " + " << vnonzero;
         if (j>1) out << "* " << j;
         out << ");\n";
      }
      out << SP << "}\n";

      return out.str();
   }


   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/, const std::vector<std::string> &dynParamNames_) override {
      if (fIsOutputConstant) return "";
      // fCountName (this op's own nonzero-element count) is registered as an
      // internal dynamic param so DOWNSTREAM operators can receive it once
      // computed; strip it back out here since it would otherwise appear in
      // this operator's OWN kernel signature before it's actually known —
      // this operator is what computes it, not a consumer of it.
      std::vector<std::string> dynParamNames = dynParamNames_;
      dynParamNames.erase(std::remove(dynParamNames.begin(), dynParamNames.end(), fCountName), dynParamNames.end());

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
      if (fIsOutputConstant) return "";
      return SP + "NonZeroKernel_" + fNY + " nonZeroKernel_" + fNY + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string /*opName*/, const std::vector<std::string> &dynParamNames_) override {
      if (fIsOutputConstant) return "";
      // See Generate_GPU_Kernel_ALPAKA: fCountName must not be passed back into
      // this operator's own kernel launch before it has been computed.
      std::vector<std::string> dynParamNames = dynParamNames_;
      dynParamNames.erase(std::remove(dynParamNames.begin(), dynParamNames.end(), fCountName), dynParamNames.end());

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
      out << SP << (fDeclaresParam ? "size_t " : "") << fCountName << " = static_cast<size_t>(*alpaka::getPtrNative(nonZeroCountHost_" << fNY << "));\n";
      return out.str();
   }

   std::string GenerateInitCode_GPU_ALPAKA() override {
      if (fIsOutputConstant) return "";
      std::string maxLen = fIsInputDynamic
         ? ("static_cast<Idx>(" + std::to_string(fRank) + "u * (" + ConvertDimShapeToLength(fShapeX) + "))")
         : (std::to_string(fRank * fInputLength) + "u");
      return SP + "deviceBuf_" + fNY + " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{" + maxLen + "}));\n";
   }
};

}


#endif
