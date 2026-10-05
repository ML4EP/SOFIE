#ifndef SOFIE_ROPERATOR_GEMM
#define SOFIE_ROPERATOR_GEMM


#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIEHelpers.hxx"

#include <sstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <limits>
#include <cassert>


namespace SOFIE{


   template <typename T>
   class ROperator_Gemm final : public ROperator
   {

   private:
      bool fIsDynamic = false;
      bool fBroadcastBias = false;
      bool fCheckBiasShapeAtRuntime = false; // flag to identify the need to do a run time check of bias shape compatibility in case of dynamic shapes and uni-directional broadcasting

      // low rank factorization of weight matrix B: B (rows x cols) ~= Bin (rows x rank) * Bout (rank x cols)
      bool fLowRank = false;
      std::size_t fLowRankRank = 0;
      std::string fLowRankInName;
      std::string fLowRankOutName;
      bool fBiasBroadcastAssumed = false;

      float fAttrAlpha = 1.0;
      float fAttrBeta = 1.0;
      int_t fAttrTransA = 0;
      int_t fAttrTransB = 0;

      std::string fNA;
      std::string fNB;
      std::string fNC = "";
      std::string fNY;
      std::string fType;
      EActivationType fActivation;
      float fLeakyReluAlpha = 0.01f;   // used when fActivation == LEAKYRELU
      std::vector<Dim> fShapeA;
      std::vector<Dim> fShapeB;
      std::vector<size_t> fShapeC;
      std::vector<Dim> fDimShapeC;
      std::vector<Dim> fShapeY;
      RModel * fModel = nullptr;
      bool fStridedA = false;  ///< A is a graph input read through its strides (Options::kStridedInput)
      bool fStridedB = false;  ///< B is a graph input read through its strides

   public:

      ROperator_Gemm(){}
      ROperator_Gemm(float alpha, float beta, int_t transA, int_t transB, std::string nameA, std::string nameB, std::string nameY, EActivationType activation=EActivationType::UNDEFINED):
         fAttrAlpha(alpha), fAttrBeta(beta), fAttrTransA(transA), fAttrTransB(transB), fNA(UTILITY::Clean_name(nameA)),
         fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY))
      {
         fActivation = activation;
         fType = "float";
         static_assert(std::is_same_v<T, float>,
                  "SOFIE - Unsupported type parsing a Gemm operator");
         fInputTensorNames = { fNA, fNB };
         fOutputTensorNames = { fNY };
         fKind = OperatorKind::GEMM;
      }

      ROperator_Gemm(float alpha, float beta, int_t transA, int_t transB, std::string nameA, std::string nameB, std::string nameC, std::string nameY, EActivationType activation=EActivationType::UNDEFINED):
         fAttrAlpha(alpha), fAttrBeta(beta), fAttrTransA(transA), fAttrTransB(transB), fNA(UTILITY::Clean_name(nameA)),
         fNB(UTILITY::Clean_name(nameB)), fNC(UTILITY::Clean_name(nameC)), fNY(UTILITY::Clean_name(nameY)), fActivation(activation)
      {
         fActivation = activation;
         fType = "float";

         fInputTensorNames = {fNA, fNB, fNC};
         fOutputTensorNames = { fNY };
         fKind = OperatorKind::GEMM;
      }

      template <typename U>
      std::vector<U> DoShapeInference(const std::vector<std::vector<U>> & input){
         if (input.size() > 3) throw std::runtime_error("SOFIE Gemm Op Shape Inference only need 2 or 3 input tensor");
         // accept tensor with input dimensions > 2
         // example: A = (d1,d2,...,N1,N2)  B = (d1,d2,...,N2,N3)    --> Y = (d1,d2,..,N1,N3)
         for (auto& i: input){
            if (i.size() < 2){
               throw std::runtime_error("SOFIE Gemm Op Shape Inference only accept input tensor with >=2 dimensions");
            }
         }

         // when there are 3 inputs shape of Y is the one of C
         if (input.size() == 3){
            //shape of C is shape of Y
            return input[2];
         }
         // ioffset cannot be less than 2
         int ioffset = input[0].size()-2;  // in case of tensors with dim > 2

         std::vector<U> s_a(input[0].begin() + ioffset, input[0].begin() + ioffset + 2);
         std::vector<U> s_b(input[1].begin() + ioffset, input[1].begin() + ioffset + 2);
         // reverse in case of transpose
         if (fAttrTransA){
            std::reverse(s_a.begin(), s_a.end());
         }
         if (fAttrTransB){
            std::reverse(s_b.begin(), s_b.end());
         }
         std::vector<U> s_y;
         s_y.reserve(input[0].size());
         if (input[0].size() > 2 && input[1].size() == input[0].size()) {
            // in case of dim > 2 first dimensions are equal to the input ones not
            // equal to 1 (e.g. (1,2,3) * (2,3,4) -> (2,2,4))
            // here could probably use the Broadcasting function  UTILITY::MultidirectionalBroadcastShape
            for (size_t i = 0; i < input[0].size()-2; i++) {
               Dim valueA = input[0][i];
               Dim valueB = input[1][i];
               if (valueA.GetVal() != valueB.GetVal()) {
                  if (valueB.GetVal() == "1")
                     s_y.push_back(input[0][i]);
                  else if (valueA.GetVal() == "1")
                     s_y.push_back(input[1][i]);
                  else if (!valueA.isParam && !valueB.isParam)
                     throw std::runtime_error("SOFIE Gemm Op - invalid input shapes " + valueA.GetVal() + " and "
                        + valueB.GetVal());
                  else if (valueA.isParam && valueB.isParam){
                      // check which parameter is first in RModel list
                     auto & dimNames = fModel->GetDimShapeNames();
                     auto p1 = std::find(dimNames.begin(), dimNames.end(), valueA.param);
                     auto p2 = std::find(dimNames.begin(), dimNames.end(), valueB.param);
                     if (p1 < p2) s_y.push_back(input[0][i]);
                     else  s_y.push_back(input[1][i]);
                  }
                  else if (!valueA.isParam)
                     s_y.push_back(input[0][i]);
                  else if (!valueB.isParam)
                     s_y.push_back(input[1][i]);
                  else
                     throw std::runtime_error("SOFIE Gemm Op - invalid input shapes " + valueA.GetVal() + " and "
                        + valueB.GetVal());
               }
               else
                  s_y.push_back(input[0][i]);
            }
         }

         s_y.push_back(s_a[0]);
         s_y.push_back(s_b[1]);
         return s_y;
      }

      std::vector<Dim> DynamicShapeInference(const std::vector<std::vector<Dim>> & input){
         return DoShapeInference<Dim>(input);
      }



      void Initialize(RModel& model) override {
         //TODO: propagate A or B as specified by ONNX standard
         fModel = &model;

         if ((model.CheckIfTensorAlreadyExist(fNA) == false) || (model.CheckIfTensorAlreadyExist(fNB) == false) ){   //input must be a graph input, or already initialized intermediate tensor
            throw std::runtime_error("SOFIE Gemm Op Input Tensor " + fNA + " or " + fNB + " is not found in model");
         }
         if (fNC != ""){
            if (model.CheckIfTensorAlreadyExist(fNC) == false){   //input must be a graph input, or already initialized intermediate tensor
               throw std::runtime_error("SOFIE Gemm Op Input Tensor " + fNC + " is not found in model");
            }
         }
         if (model.IsDynamicTensor(fNA) || model.IsDimInputTensor(fNA) ) {
            fShapeA = model.GetDynamicTensorShape(fNA);
            fIsDynamic = true;
         } else {
            auto shapeA_int = model.GetTensorShape(fNA);
            fShapeA = ConvertShapeToDim(shapeA_int);
         }
         // case A is of dim1 we prepend a 1 but we need to remove later
         bool prependOne = false;
         if (fShapeA.size() == 1) {
            fShapeA.insert(fShapeA.begin(), Dim(1));
            prependOne = true;
         }

         if (model.IsDynamicTensor(fNB) || model.IsDimInputTensor(fNB)) {
            fShapeB = model.GetDynamicTensorShape(fNB);
            fIsDynamic = true;
         }
         else {
            auto shapeB_int = model.GetTensorShape(fNB);
            fShapeB = ConvertShapeToDim(shapeB_int);
         }
         // case B is dim1 we append a 1 but we need to remove later
         bool appendOne = false;
         if (fShapeB.size() == 1) {
            fShapeB.insert(fShapeB.end(), Dim(1));
            appendOne = true;
         }
         // assume if not shape is 2 that extra values are 1.
         // implement also MatMul case where we stack matrices (see numpy.matmul)
         if (fShapeA.size() != fShapeB.size()) {
            // if different dimensions we prepend 1 values
            if (fShapeA.size() < fShapeB.size()) {
               fShapeA.insert(fShapeA.begin(), fShapeB.size()-fShapeA.size(), Dim(1));
            } else if (fShapeB.size() < fShapeA.size()) {
               fShapeB.insert(fShapeB.begin(), fShapeA.size()-fShapeB.size(), Dim(1));
            }
         }

         // try low rank factorization of weight matrix B, if enabled on the model.
         // restrict to the plain 2D case (no MatMul batch stacking); both transB == 0
         // and transB == 1  are supported: the generated low-rank
         // Bin/Bout tensors are always stored logically untransposed (k x rank) and
         // (rank x n) regardless of how the original B was stored, so Generate methods
         //  never need to special-case transB for the chained
         // low-rank calls -- when transB == 1 we just transpose the weight data once
         // here, before factorizing, to get it into that same logical orientation.
         if (model.LowRankFactorize() && !fIsDynamic && !appendOne &&
             fShapeA.size() == 2 && fShapeB.size() == 2 &&
             model.IsInitializedTensor(fNB) && model.IsWeightTensor(fNB) &&
             model.GetTensorType(fNB) == ETensorType::FLOAT) {
            // logical (untransposed) orientation: k x n, i.e. rows = contraction dim
            std::size_t rows = fAttrTransB ? fShapeB[1].dim : fShapeB[0].dim; // k
            std::size_t cols = fAttrTransB ? fShapeB[0].dim : fShapeB[1].dim; // n
            std::size_t minDim = std::min(rows, cols);
            std::size_t rank = static_cast<std::size_t>(model.LowRankRatio() * minDim);
            if (rank < 1) rank = 1;
            if (rank < minDim) {
               auto sharedData = model.GetInitializedTensorData(fNB);
               const float *storedData = static_cast<const float *>(sharedData.get());
               std::vector<float> transposed;
               const float *wdata = storedData;
               if (fAttrTransB) {
                  // stored B is (n x k); transpose into logical (k x n) = (rows x cols)
                  transposed.resize(rows * cols);
                  for (std::size_t i = 0; i < cols; i++)    // i over n
                     for (std::size_t j = 0; j < rows; j++) // j over k
                        transposed[j * cols + i] = storedData[i * rows + j];
                  wdata = transposed.data();
               }
               std::vector<float> Ain, Bout;
               if (SOFIE::ComputeLowRankFactors(wdata, rows, cols, rank, Ain, Bout)) {
                  fLowRankRank = rank;
                  fLowRankInName = fNB + "_lrin";
                  fLowRankOutName = fNB + "_lrout";
                  model.AddInitializedTensor(fLowRankInName, ETensorType::FLOAT, std::vector<std::size_t>{rows, rank}, Ain.data());
                  model.AddInitializedTensor(fLowRankOutName, ETensorType::FLOAT, std::vector<std::size_t>{rank, cols}, Bout.data());
                  model.RemoveInitializedTensor(fNB);
                  fLowRank = true;
                  // update the operator's bookkeeping of input tensors: replace fNB with the two factors
                  for (auto &n : fInputTensorNames) {
                     if (n == fNB) { n = fLowRankInName; break; }
                  }
                  fInputTensorNames.push_back(fLowRankOutName);
                  if (model.Verbose())
                     std::cout << "Gemm " << fNB << ": low rank factorized (" << rows << "x" << cols
                                << ") -> rank " << rank << std::endl;
               }
            }
         }

         fShapeY = DynamicShapeInference({fShapeA, fShapeB});
         std::vector<size_t> shapeY = ConvertShapeToInt(fShapeY);

         // bias is normally not dynamic (not support it for time being)
         if (fNC != ""){
            if (model.IsDynamicTensor(fNC))
               fDimShapeC = model.GetDynamicTensorShape(fNC);
            else {
               fShapeC = model.GetTensorShape(fNC);
               fDimShapeC = ConvertShapeToDim(fShapeC);
            }
            // for dynamic outputs broadcasting is always needed
            bool broadcast_needed = false;
            if (fIsDynamic && shapeY.empty()) {
               broadcast_needed = true;
               fBiasBroadcastAssumed = true;
            } else
               // consider broadcasting also if they have different length
               broadcast_needed = (fShapeC != shapeY);


            if (broadcast_needed) {
               fBroadcastBias = true;
               // check if broadcasting is compatible and note that prepend 1 to shapeC
               auto r = UTILITY::MultidirectionalBroadcastShape(fShapeY, fDimShapeC);
               // return flag must not have bit equal to 2 since this is a unidirectional broadcast of C->Y
               //
               if ((r.first & 2) == 2) {
                  throw std::runtime_error("SOFIE Gemm Op - bias tensor of shape " + ConvertDimShapeToString(fDimShapeC) + " cannot be uni-directional broadcasted to " + ConvertDimShapeToString(fShapeY));
               } else if (r.first  == 4) {
                  // we need to do a run time check of bias shape if it is compatible
                  fCheckBiasShapeAtRuntime = true;
               }
               fShapeC = ConvertShapeToInt(fDimShapeC);
            }
         }

         // remove appended or prepended value of 1 in Y
         if (prependOne) {
            if (fIsDynamic)
               fShapeY.erase(fShapeY.begin());
            else
               shapeY.erase(shapeY.begin());
         }
         if (appendOne) {
            if (fIsDynamic)
               fShapeY.erase(fShapeY.end()-1);
            else
               shapeY.erase(shapeY.end()-1);
         }

         bool canFold = !fIsDynamic
            && model.IsInitializedTensor(fNA)
            && model.IsInitializedTensor(fNB)
            && (fNC.empty() || model.IsInitializedTensor(fNC))
            && fShapeA.size() <= 2
            && !fBroadcastBias
            && !fCheckBiasShapeAtRuntime;

         if (canFold) {
            auto shapeA_i = ConvertShapeToInt(fShapeA);
            auto shapeB_i = ConvertShapeToInt(fShapeB);
            size_t dimA = shapeA_i.size();
            size_t dimB = shapeB_i.size();
            size_t m = fAttrTransA ? shapeA_i[dimA - 1] : shapeA_i[dimA - 2];
            size_t k = fAttrTransA ? shapeA_i[dimA - 2] : shapeA_i[dimA - 1];
            size_t n = fAttrTransB ? shapeB_i[dimB - 2] : shapeB_i[dimB - 1];

            auto dataA = static_cast<T *>(model.GetInitializedTensorData(fNA).get());
            auto dataB = static_cast<T *>(model.GetInitializedTensorData(fNB).get());

            std::vector<T> dataY(m * n, T(0));
            for (size_t i = 0; i < m; i++) {
               for (size_t j = 0; j < n; j++) {
                  T sum{};
                  for (size_t p = 0; p < k; p++) {
                     T aVal = fAttrTransA ? dataA[p * m + i] : dataA[i * k + p];
                     T bVal = fAttrTransB ? dataB[j * k + p] : dataB[p * n + j];
                     sum += aVal * bVal;
                  }
                  dataY[i * n + j] = static_cast<T>(fAttrAlpha) * sum;
               }
            }
            if (!fNC.empty()) {
               auto dataC = static_cast<T *>(model.GetInitializedTensorData(fNC).get());
               for (size_t idx = 0; idx < dataY.size(); idx++)
                  dataY[idx] += static_cast<T>(fAttrBeta) * dataC[idx];
            }
            if (fActivation == EActivationType::RELU) {
               for (auto &v : dataY)
                  v = std::max(v, T(0));
            }

            model.AddConstantTensor<T>(fNY, shapeY, dataY.data());
            model.SetNotWritableInitializedTensor(fNA);
            model.SetNotWritableInitializedTensor(fNB);
            if (!fNC.empty())
               model.SetNotWritableInitializedTensor(fNC);
            fIsOutputConstant = true;

            if (model.Verbose()) {
               std::cout << "Gemm (or MatMul) " << fNA << " , " << fNB;
               if (!fNC.empty())
                  std::cout << " , " << fNC;
               std::cout << " ---> " << fNY << " (constant) " << ConvertShapeToString(shapeY) << std::endl;
            }
            return;
         }

         if (!fIsDynamic)
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), shapeY);
         else
            model.AddDynamicTensor(fNY, model.GetTensorType(fNA), fShapeY);

         if (model.Verbose()){
            std::cout << "Gemm (or MatMul) " << " ---> " << fNY << " shape ";
            if (fIsDynamic)
               std::cout << ConvertDimShapeToString(fShapeY) << std::endl;
            else
               std::cout << ConvertShapeToString(shapeY) << std::endl;
         }

         // the operands A and B can be graph inputs read through their strides (Options::kStridedInput).
         // BLAS describes a matrix by a leading dimension and a transpose flag, so the strides of an operand must have
         // a unit stride; the plain 2D case is supported.
         fStridedA = model.IsStridedInputTensor(fNA);
         fStridedB = model.IsStridedInputTensor(fNB);
         fHasStridedInput = fStridedA || fStridedB;
         if (!fNC.empty() && model.IsStridedInputTensor(fNC))
            throw std::runtime_error("SOFIE Gemm Op - strided input is not supported for the bias (C)");
         if (fHasStridedInput && (fLowRank || fShapeA.size() != 2 || fShapeB.size() != 2 ||
                                  model.GetDimTensorShape(fNA).size() != 2 || model.GetDimTensorShape(fNB).size() != 2))
            throw std::runtime_error("SOFIE Gemm Op - strided input is only supported for a plain 2D Gemm "
                                     "(no MatMul batching, no low rank factorization)");

         model.AddNeededStdLib("algorithm");


         if (fType == "float")
            model.AddNeededHelperFunction("Gemm_Call");
         if (fNC != "") {
            model.AddNeededHelperFunction("Copy");
            model.AddNeededHelperFunction("Fill");
         }
         if (fActivation == EActivationType::RELU)
            model.AddNeededHelperFunction("Relu");
      }

      std::string Generate(std::string opName) override {
         if (fIsOutputConstant)
            return "";

         opName = "op_" + opName;

         std::stringstream out;
         out << "\n//--------- Gemm " << opName << " " << ConvertDimShapeToString(fShapeA) << " * " << ConvertDimShapeToString(fShapeB)
             << " -> " << ConvertDimShapeToString(fShapeY) << "\n";
         // need to consider case A and B have dim > 2 (for MatMul)
         int64_t dimA = fShapeA.size();
         int64_t dimB = fShapeB.size();
         int64_t dimY = fShapeY.size();
         int64_t dimC = fDimShapeC.size();
         if (dimA != dimB || dimA != dimY || (fBroadcastBias && dimC != dimY)) {
             std::cout << " shape A " << ConvertDimShapeToString(fShapeA)
                       << " shape B " << ConvertDimShapeToString(fShapeB)
                       << " shape C " << ConvertDimShapeToString(fDimShapeC)
                       << " shape Y " << ConvertDimShapeToString(fShapeY) << std::endl;
             throw std::runtime_error("SOFIE Gemm(MatMul) has invalid shape for inputs or output");
         }
         auto m = (fAttrTransA ? fShapeA[dimA-1].GetVal() : fShapeA[dimA-2].GetVal());
         auto n = (fAttrTransB ? fShapeB[dimB-2].GetVal() : fShapeB[dimB-1].GetVal());
         auto k = (fAttrTransA ? fShapeA[dimA-2].GetVal() : fShapeA[dimA-1].GetVal());
         // size of A: if (transposeA) is m*k else k*m
         // size of B  n*k
         std::vector<Dim> sY = {fShapeY[dimY-2], fShapeY[dimY-1]};
         // extra dimensions in case of stacked MatMul
         std::vector<Dim> sExtraY;
         for (int64_t i = 0; i < dimY-2; i++) {
            sExtraY.push_back(fShapeY[i]);
         }
         auto lengthGemm = ConvertDimShapeToLength(sY); // size of the Gemm operation
         auto lengthExtra_Y = ConvertDimShapeToLength(sExtraY); // extra length in case input tensors are of dim>2 (MatMul)
         std::string lengthExtra_C;
         std::vector<Dim> sExtraC;
         std::vector<Dim> sC;
         bool haveExtraC = false;
         if (dimC > 2) {
            sC = {fDimShapeC[dimC-2], fDimShapeC[dimC-1]};
            for (int64_t i = 0; i < dimC-2; i++) {
               sExtraC.push_back(fDimShapeC[i]);
            }
            lengthExtra_C = ConvertDimShapeToLength(sExtraC);
            if (lengthExtra_C != "1") haveExtraC = true;
         } else if (dimC > 0) {
            for (int64_t i = 0; i < dimC; i++) {
               sC.push_back(fDimShapeC[i]);
            }
         }

         // case bias is present
         if (!fNC.empty()){
             // when the 2 last dims of bias and Y are not compatible we need to perform a run time broadcast
            if (sC != sY)
               fBroadcastBias = true;
            else if (fBiasBroadcastAssumed && sExtraC == sExtraY)
               fBroadcastBias = false;
            if (!fBroadcastBias) {
               // add a check in case broadcasting was not needed or done outside of session
               // C should have smaller dimension of Y
               if (!fIsDynamic) {
                  if ((std::stoi(lengthGemm) != std::stoi(ConvertDimShapeToLength(sC))) ||
                      ( haveExtraC &&  std::stoi(lengthExtra_Y) != std::stoi(lengthExtra_C)))
                     throw std::runtime_error("SOFIE Gemm Op " + opName + " Bias tensor " + fNC + " has not correct size "
                            + ConvertShapeToString(fShapeC) + " output length " + lengthGemm);
               } else {
                  // add a dynamic check (C should not be a dynamic tensor)
                  out << SP << "assert(" << lengthGemm << " == " <<  ConvertDimShapeToLength(sC) << ");\n";
                  if (haveExtraC) out << SP << "assert(" << lengthExtra_Y << " == " <<  lengthExtra_C << ");\n";
               }
            }
         } else {
            fBroadcastBias = false;
            //in this case fAttrBeta needs to be equal to zero otherwise second time we run we will use
            // the previous result
            if (fAttrBeta != 0) {
               // some model don't have bias but Beta is not zero - force it to zero
               fAttrBeta = 0;
               std::cout << "WARNING: SOFIE Gemm Op " + opName + " Bias tensor is not present but beta value in Gemm is not zero - force it to zero\n";
            }
         }

         // include MatMul case where we stack the Gemm operations
         // exclude case where we have only 1's in the additional dims
         bool doStackMul = dimY > 2 && ( fIsDynamic  || std::stoi(lengthExtra_Y) > 1);
         // compute input offset for stack multiplications
         std::string lengthExtra_A;
         std::string lengthExtra_B;
         std::string increment_A;
         std::string increment_B;

         if (doStackMul) {
            std::vector<Dim> sA(fShapeA.begin(), fShapeA.begin()+dimA-2);
            std::vector<Dim> sB(fShapeB.begin(), fShapeB.begin()+dimB-2);
            std::vector<Dim> mA = {fShapeA[dimA-2], fShapeA[dimA-1]};
            std::vector<Dim> mB = {fShapeB[dimB-2], fShapeB[dimB-1]};
            lengthExtra_A = ConvertDimShapeToLength(sA);
            lengthExtra_B = ConvertDimShapeToLength(sB);
            // if A ( b, m, k) and B (b, k, n) these are the strides of A and B ( m*k for A and n*k for B )
            increment_A = ConvertDimShapeToLength(mA);
            increment_B = ConvertDimShapeToLength(mB);
         }
         bool extraA = (doStackMul && lengthExtra_A != "1");
         bool extraB = (doStackMul && lengthExtra_B != "1");
         bool extraC = (doStackMul && haveExtraC && !fBroadcastBias);
         // run time check for bias broadcasting
         std::string biasShapeType = opName + "_biasShapeType";
         if (fBroadcastBias && fCheckBiasShapeAtRuntime) {
            // create a flag according to bias shape:
            // = 1 for (1,Y2)
            // = 2 for (Y1,1)
            // = 3 for a scalar
            out << SP << "int " << biasShapeType << " = 0;\n";
            // case vector of columns
            if (sC[0].GetVal() != "1" && sC[1].GetVal() != sY[1].GetVal())
               out << SP << "if (" << sC[0] << " == 1 && " << sC[1] << " == " << sY[1] << ")\n";
            else if (sC[0].GetVal() == "1")
               out << SP << "if (" << sC[1] << " == " << sY[1] << ")\n";
            else if (sC[1].GetVal() == sY[1].GetVal())
               out << SP << "if (" << sC[0] << " == 1)\n";

            out << SP << SP << biasShapeType << " = 1;\n";

            // case vector of rows
            if (sC[1].GetVal() != "1" && sC[0].GetVal() != sY[0].GetVal())
               out << SP << "else if (" << sC[1] << " == 1 && " << sC[0] << " == " << sY[0] << ")\n";
            else if (sC[1].GetVal() == "1")
                out << SP << "else if (" << sC[0] << " == " << sY[0] << ")\n";
            else if (sC[0].GetVal() == sY[0].GetVal())
               out << SP << "else if (" << sC[1] << " == 1)\n";

            out << SP << SP << biasShapeType << " = 2;\n";

            // case scalar
            if (sC[0].GetVal() != "1" && sC[1].GetVal() != "1")
               out << SP << "else if (" << sC[0] << " == 1 && " << sC[1] << " == 1 )\n";
            else if (sC[0].GetVal() == "1")
               out << SP << "else if (" << sC[1] << " == 1)\n";
            else if (sC[1].GetVal() == "1")
               out << SP << "else if (" << sC[0] << " == 1)\n";
            out << SP << SP << biasShapeType << " = 3;\n";
            out << SP << "else\n";
            out << SP << SP << "throw std::runtime_error(\"SOFIE Gemm Op - bias tensor "
                                 << ConvertDimShapeToString(fDimShapeC) << " cannot be broadcasted to "
                                 << ConvertDimShapeToString(fShapeY) << "\");\n";
         }
         auto SP2 = SP;
         if (doStackMul) {
            out << SP << "size_t " << opName << "_y_offset = 0;\n"; // needed if we stack the gemm operations
            if (extraA)
               out << SP << "size_t " << opName << "_A_offset = 0;\n";
            if (extraB)
               out << SP << "size_t " << opName << "_B_offset = 0;\n";
            if (extraC)
               out << SP << "size_t " << opName << "_C_offset = 0;\n";
            out << SP << "for (size_t i = 0; i < " << lengthExtra_Y << "; i++){\n";
            SP2 += SP;
         }
         // do the bias broadcasting at run time by
         // initializing output Y vector with bias values
         if (fBroadcastBias) {

            fAttrBeta = 1.;

            // loop on first output dimension
            out << SP2 << "for (size_t j = 0; j < " << sY[0] << "; j++) { \n";
            out << SP2 << SP << "size_t y_index = ";
            if (doStackMul) // add offset in case of stack multiplications (not sure if bias is present in these cases)
               out <<  opName << "_y_offset + ";
            if (sY[1].GetVal() != "1")
               out << sY[1] << " * j;\n";
            else
               out << "j;\n";

            std::string prefix = SP2 + SP;
            std::string target = "tensor_" + fNY;
            if (sC.size() != 2) {
               throw std::runtime_error("SOFIE Gemm Op - invalid rank for bias tensor " + ConvertDimShapeToString(fDimShapeC) + ConvertDimShapeToString(sC));
            } if (sC[0].GetVal() == "1" && sC[1].GetVal() == sY[1].GetVal()) {
               out << prefix << "Copy(" << target << " + y_index, tensor_" << fNC << ", " << sY[1] << ");\n";
            } else if (sC[1].GetVal() == "1" && sC[0].GetVal() == sY[0].GetVal()) {
               out << prefix << "Fill(" << target << " + y_index, tensor_" << fNC << "[j], " << sY[1] << ");\n";
            } else if (sC[0].GetVal() == "1" && sC[1].GetVal() == "1") {
               // scalar case
               out << prefix << "Fill(" << target << " + y_index, tensor_" << fNC << "[0], " << sY[1] << ");\n";
            } else if (fCheckBiasShapeAtRuntime) {
               // in the generic dynamic case we check at run time that bias is compatible
               // we check that bias[0] = 1 or equal to SY[0] and that bias[1] = 1 or equal to SY[1]
               // tbd: this run-time check coul;d be moved outside the loop for better run time efficiency
               out << SP2 << SP << "if (" << biasShapeType << " == 1)\n";   // case vector of columns
               out << SP << prefix << "Copy(" << target << " + y_index, tensor_" << fNC << ", " << sY[1] << ");\n";
               out << SP2 << SP << "else if (" << biasShapeType << " == 2)\n";  // case vector of rows
               out << SP << prefix << "Fill(" << target << " + y_index, tensor_" << fNC << "[j], " << sY[1] << ");\n";
               out << SP2 << SP << "else \n";  // scalar case
               out << SP << prefix << "Fill(" << target << " + y_index, tensor_" << fNC << "[0], " << sY[1] << ");\n";
            } else {
               throw std::runtime_error("SOFIE Gemm Op - invalid shape for bias tensor " + ConvertDimShapeToString(fDimShapeC));
            }

            out << SP2 << "}\n";
         }

         if (fType == "float" && fLowRank){
            // low rank factorized weight: B (k x n) ~= Bin (k x rank) * Bout (rank x n)
            // Y = alpha * op(A) * Bin * Bout (+ bias)  computed as two chained Gemm calls:
            //   tmp (m x rank) = alpha * op(A) * Bin
            //   Y   (m x n)    = 1 * tmp * Bout + beta * bias
            std::string tmpName = opName + "_lr_tmp";
            out << SP2 << "std::vector<float> " << tmpName << "(" << m << " * " << fLowRankRank << ");\n";
            out << SP2 << "Gemm_Call(" << tmpName << ".data(), false, "
                << (fAttrTransA ? "true, " : "false, ")
                << fLowRankRank << ", " << m << ", " << k << ", "
                << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrAlpha
                << ", tensor_" << fLowRankInName << ", tensor_" << fNA << ", 0.f, nullptr);\n";

            out << SP2 << "Gemm_Call(" << "tensor_" << fNY << ", false, false, "
                << n << ", " << m << ", " << fLowRankRank << ", "
                << std::setprecision(std::numeric_limits<float>::max_digits10) << 1.f
                << ", tensor_" << fLowRankOutName << ", " << tmpName << ".data(), "
                << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrBeta << ",";
            if (!fNC.empty() && !fBroadcastBias) {
               out << "tensor_" << fNC;
            } else {
               out << "nullptr";
            }
            out << ");\n";

         } else if (fType == "float" && fHasStridedInput) {
            // A and B are read through their strides {s0, s1}: element (i,j) of the stored matrix is at i*s0 + j*s1.
            // Row-major with s1 == 1: leading dimension s0. Column-major with s0 == 1: leading dimension s1 and the
            // transpose flag is flipped. Without a unit stride the matrix cannot be described to BLAS.
            auto resolve = [&](const std::string &tag, const std::string &name, bool strided,
                               const std::vector<Dim> &shape, bool trans, const std::string &defaultLd) {
               out << SP2 << "bool " << opName << "_trans" << tag << " = " << (trans ? "true" : "false") << ";\n";
               out << SP2 << "size_t " << opName << "_ld" << tag << " = " << defaultLd << ";\n";
               if (!strided)
                  return;
               const std::string st = "stride_" + opName + "_" + tag;
               const std::string rows = "static_cast<size_t>(" + shape[0].GetVal() + ")";
               const std::string cols = "static_cast<size_t>(" + shape[1].GetVal() + ")";
               out << GenerateInputStrideCode(opName + "_" + tag, name, shape);
               out << SP2 << "if (" << st << "[1] == 1) {\n";
               out << SP2 << SP << opName << "_ld" << tag << " = std::max<size_t>(" << st << "[0], " << cols << ");\n";
               out << SP2 << "} else if (" << st << "[0] == 1) {\n";
               out << SP2 << SP << opName << "_trans" << tag << " = !" << opName << "_trans" << tag << ";\n";
               out << SP2 << SP << opName << "_ld" << tag << " = std::max<size_t>(" << st << "[1], " << rows << ");\n";
               out << SP2 << "} else {\n";
               out << SP2 << SP << "throw std::runtime_error(\"SOFIE Gemm Op " << opName << " - input tensor " << name
                   << " has strides {\" + std::to_string(" << st << "[0]) + \", \" + std::to_string(" << st
                   << "[1]) + \"} without a unit stride, which BLAS cannot handle\");\n";
               out << SP2 << "}\n";
            };
            out << SP2 << "{\n";
            resolve("A", fNA, fStridedA, fShapeA, fAttrTransA, fAttrTransA ? m : k);
            resolve("B", fNB, fStridedB, fShapeB, fAttrTransB, fAttrTransB ? k : n);
            // the operands of Gemm_Call_ld are swapped: B first
            out << SP2 << "Gemm_Call_ld(" << "tensor_" << fNY << ", " << opName << "_transB, " << opName << "_transA, "
                << n << ", " << m << ", " << k << ", ";
            out << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrAlpha << ", tensor_" << fNB
                << ", " << opName << "_ldB, tensor_" << fNA << ", " << opName << "_ldA, "
                << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrBeta << ",";
            if (!fNC.empty() && !fBroadcastBias)
               out << "tensor_" << fNC;
            else
               out << "nullptr";
            out << ");\n";
            out << SP2 << "}\n";

         } else if (fType == "float"){

            out << SP2 << "Gemm_Call(" << "tensor_" << fNY;
             if (doStackMul) out << " + " << opName << "_y_offset";
            out <<   ", "
             << (fAttrTransB ? "true, " : "false, ")
             << (fAttrTransA ? "true, " : "false, ")
             << n << ", " << m << ", " << k << ", ";
            out << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrAlpha << ", tensor_" << fNB;
            if (extraB) out << " + " << opName << "_B_offset";
            out << ", tensor_" << fNA;
            if (extraA) out << " + " << opName << "_A_offset";
            out << ", " << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrBeta << ",";
            // in the case of bias and no broadcasting needed - I need to add bias as an extra tensor in Gemm call
            if (!fNC.empty() && !fBroadcastBias) {
               out << "tensor_" << fNC;
               if (extraC) {
                  out << " + " << opName << "_C_offset";
               }
            } else {
               out << "nullptr";
            }
            out << ");\n";

         }

         if (doStackMul) {
            out << SP << SP <<  opName << "_y_offset += " << lengthGemm << ";\n";
            if (lengthExtra_A != "1")
               out << SP << SP << opName << "_A_offset += " << increment_A << ";\n";
            if (lengthExtra_B != "1")
               out << SP << SP << opName << "_B_offset += " << increment_B << ";\n";
            if (extraC)
               // increment_C is lengthGEmm
               out << SP << SP << opName << "_C_offset += " << lengthGemm << ";\n";
            out << SP << "}\n"; // end of loop on the stacked multiplication
         }

         // fuse activation with GEMM output (in-place on fNY)
         if (fActivation == EActivationType::RELU) {
               out << SP << "//--- applying RELU to output\n";
               std::string tnsr = "tensor_" + fNY;
               std::string reluSize = ConvertDimShapeToLength(fShapeY);
               out << SP << "Relu(" << tnsr << ", " << tnsr << ", " << reluSize << ");\n";
         } else if (fActivation == EActivationType::LEAKYRELU) {
               out << SP << "//--- applying LEAKYRELU to output (in-place)\n";
               std::string tnsr = "tensor_" + fNY;
               std::string reluSize = ConvertDimShapeToLength(fShapeY);
               out << SP << "{\n";
               out << SP << SP << "constexpr float lrelu_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fLeakyReluAlpha << "f;\n";
               out << SP << SP << "for (size_t _i = 0; _i < " << reluSize << "; ++_i)\n";
               out << SP << SP << SP << tnsr << "[_i] = " << tnsr << "[_i] >= 0.f ? " << tnsr << "[_i] : lrelu_alpha * " << tnsr << "[_i];\n";
               out << SP << "}\n";
         }

         return out.str();
      }

      std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
         std::string op;
         op = "\n//------ GEMM_BIAS_ADD_KERNEL_ALPAKA\n";
         op += "struct GemmBiasAddKernel {\n";
         op += SP + "template<typename TAcc, typename T>\n";
         op += SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, T* __restrict__ Y, T const* __restrict__ C, std::size_t m, std::size_t n, std::size_t cRows, std::size_t cCols, bool hasExtraC, bool applyRelu, std::size_t totalElements) const {\n";
         op += SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + "if (idx >= totalElements) return;\n";
         op += SP + SP + "std::size_t sliceSize = m * n;\n";
         op += SP + SP + "std::size_t i = idx / sliceSize;\n";
         op += SP + SP + "std::size_t within = idx % sliceSize;\n";
         op += SP + SP + "std::size_t r = within / n;\n";
         op += SP + SP + "std::size_t c = within % n;\n";
         op += SP + SP + "std::size_t cSliceOffset = hasExtraC ? i * (cRows * cCols) : 0;\n";
         op += SP + SP + "std::size_t cr = (cRows == 1) ? 0 : r;\n";
         op += SP + SP + "std::size_t cc = (cCols == 1) ? 0 : c;\n";
         op += SP + SP + "T value = Y[idx] + C[cSliceOffset + cr * cCols + cc];\n";
         op += SP + SP + "Y[idx] = applyRelu ? (value >= T(0) ? value : T(0)) : value;\n";
         op += SP + "}\n";
         op += "};\n";
         return op;
      }

      std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
         return "GemmBiasAddKernel gemmBiasAddKernel;\n";
      }

      std::string Generate_GPU_ALPAKA(std::string opName) override {
         opName = "op_" + opName;

         if (fShapeA.empty() || fShapeB.empty() || fShapeY.empty() || (fNC != "" && fDimShapeC.empty())) {
            throw std::runtime_error("SOFIE Gemm Op called to Generate without being initialized first");
         }
         std::stringstream out;
         out << "\n//--------- Gemm_GPU_ALPAKA\n";
         out << SP << "char " << opName << "_transA = " << (fAttrTransA ? "\'t\'" : "\'n\'") << ";\n";
         out << SP << "char " << opName << "_transB = " << (fAttrTransB ? "\'t\'" : "\'n\'") << ";\n";
         // need to consider case A and B have dim > 2 (for MatMul)
         int64_t dimA = fShapeA.size();
         int64_t dimB = fShapeB.size();
         int64_t dimY = fShapeY.size();
         if (dimA != dimB || dimA != dimY) {
             throw std::runtime_error("SOFIE Gemm(MatMul) has invalid shape for inputs or output");
         }
         auto m = (fAttrTransA ? fShapeA[dimA-1].GetVal() : fShapeA[dimA-2].GetVal());
         auto n = (fAttrTransB ? fShapeB[dimB-2].GetVal() : fShapeB[dimB-1].GetVal());
         auto k = (fAttrTransA ? fShapeA[dimA-2].GetVal() : fShapeA[dimA-1].GetVal());
         std::vector<Dim> sY = {fShapeY[dimY-2], fShapeY[dimY-1]};
         // extra dimensions in case of stacked MatMul
         std::vector<Dim> sA;
         for (int64_t i = 0; i < dimY-2; i++) {
            sA.push_back(fShapeY[i]);
         }
         auto lengthGemm = ConvertDimShapeToLength(sY); // size of the Gemm operation
         auto lengthExtra = ConvertDimShapeToLength(sA); // extra length in case input tensors are of dim>2 (MatMul)

         std::vector<Dim> sC, sExtraC;
         bool haveExtraC = false;
         bool cShapeKnown = true;
         if (!fNC.empty()) {
            int64_t dimC = static_cast<int64_t>(fDimShapeC.size());
            if (dimC >= 2) {
               sC = {fDimShapeC[dimC-2], fDimShapeC[dimC-1]};
               for (int64_t i = 0; i < dimC-2; i++) sExtraC.push_back(fDimShapeC[i]);
            } else if (dimC == 1) {
               sC = {Dim(static_cast<size_t>(1)), fDimShapeC[0]};
            } else {
               sC = {Dim(static_cast<size_t>(1)), Dim(static_cast<size_t>(1))};
            }
            auto lengthExtraC = sExtraC.empty() ? std::string("1") : ConvertDimShapeToLength(sExtraC);
            haveExtraC = lengthExtraC != "1";
            cShapeKnown = !sC[0].isParam && !sC[1].isParam;
         }

         out << SP << "int " << opName << "_m = " << m << ";\n";
         out << SP << "int " << opName << "_n = " << n << ";\n";
         out << SP << "int " << opName << "_k = " << k << ";\n";
         out << SP << "float " << opName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrAlpha << ";\n";
         
         // restricting to a 0 beta since BIAS is configured separately through sofieBLAS interface
         out << SP << "float " << opName << "_beta = 0;\n";

         // case bias is present
         if (!fNC.empty()){
            if (!fBroadcastBias) {
               // add a check in case broadcasting was not needed or done outside of session
               // C should have same size as Y
               if (!fIsDynamic) {
                  if (std::stoi(lengthGemm) != static_cast<int>(ConvertShapeToLength(fShapeC)))
                     throw std::runtime_error("SOFIE Gemm Op " + opName + " Bias tensor has not correct size "
                            + ConvertDimShapeToString(fDimShapeC) + " output length " + lengthGemm);
               } else {
                  // add a dynamic check (C should equal output size)
                  out << SP << "assert(" << lengthGemm << " == " <<  ConvertDimShapeToLength(fDimShapeC) << ");\n";
               }
            }
         } else {
            fBroadcastBias = false;
            //in this case fAttrBeta needs to be equal to zero otherwise second time we run we will use
            // the previous result
            if (fAttrBeta != 0) {
               // some model don't have bias but Beta is not zero - force it to zero
               fAttrBeta = 0;
               std::cout << "WARNING: SOFIE Gemm Op " + opName + " Bias tensor is not present but beta value in Gemm is not zero - force it to zero\n";
            }
         }

         // include MatMul case where we stack the Gemm operations
         // exclude case where we have only 1's in the additional dims
         bool doStackMul = dimY > 2 && ( fIsDynamic  || std::stoi(lengthExtra) > 1);

         // B is a shared weight (broadcast over the stacked/batch dimension) when all its
         // leading dims (beyond the 2 matrix dims) are statically known to be 1
         bool bLeadingDimsAllOne = true;
         for (int64_t i = 0; i < dimB - 2; i++) {
            if (fShapeB[i].isParam || fShapeB[i].dim != 1) { bLeadingDimsAllOne = false; break; }
         }

         std::string strideAExpr = "(" + m + ") * (" + k + ")";
         std::string strideBExpr = bLeadingDimsAllOne ? "" : ("(" + n + ") * (" + k + ")");
         std::string strideYExpr = "(" + m + ") * (" + n + ")";
         std::string strideCExpr = (!fNC.empty() && !fBroadcastBias) ? lengthGemm : "";

         size_t strideA = 0, strideB = 0, strideY = 0, strideC = 0;
         bool batchCollapseB = false;
         bool useSBatched    = false;
         bool useBatchedBias = false;
         if (doStackMul && !fIsDynamic) {
            strideA = static_cast<size_t>(std::stoi(m)) * static_cast<size_t>(std::stoi(k));
            strideB = bLeadingDimsAllOne ? 0
                                         : static_cast<size_t>(std::stoi(n)) * static_cast<size_t>(std::stoi(k));
            strideY = static_cast<size_t>(std::stoi(m)) * static_cast<size_t>(std::stoi(n));
            strideC = !fNC.empty() ? static_cast<size_t>(std::stoi(lengthGemm)) : 0;

            batchCollapseB = (strideB == 0);
            useSBatched    = !batchCollapseB && fNC.empty();
            useBatchedBias = !batchCollapseB && !fNC.empty() && !fLowRank && cShapeKnown;
         }

         bool useSerialLoop = doStackMul && !batchCollapseB && !useSBatched && !useBatchedBias;
         if (useSerialLoop) {
            out << SP << "size_t " << opName << "_yoffset = 0;\n";
            out << SP << "for (int i = 0; i < " << lengthExtra << "; i++){\n";
         }

         std::string pA = "static_cast<const float*>(alpaka::getPtrNative(deviceBuf_" + fNA + "))";
         std::string pB = "static_cast<const float*>(alpaka::getPtrNative(deviceBuf_" + fNB + "))";
         std::string pY = "alpaka::getPtrNative(deviceBuf_" + fNY + ")";
         if (useSerialLoop) {
            pA += " + i * (" + strideAExpr + ")";
            if (!strideBExpr.empty()) pB += " + i * (" + strideBExpr + ")";
            // strideB shared (empty): B is a shared weight, pointer stays at base
            pY += " + i * (" + strideYExpr + ")";
         }

         if (fHasStridedInput) {
            // ----------------------------------------------------------------
            // plain 2D Gemm with A and/or B read through their strides {s0, s1} (Options::kStridedInput). The
            // sofieBLAS gemm/matmul entry points always use dense leading dimensions, so the cuBLAS strided batched
            // entry point (batchCount = 1), which takes explicit leading dimensions, is used. The bias (and the fused
            // ReLU) are then applied by the bias kernel. As for the CPU, a strided operand needs a unit stride:
            // row-major (s1 == 1) gives ld = s0, column-major (s0 == 1) gives ld = s1 and the transpose flag is flipped.
            // ----------------------------------------------------------------
            if (haveExtraC)
               throw std::runtime_error("SOFIE Gemm Op - strided input does not support a bias with batch dimensions");
            auto resolve = [&](const std::string &tag, const std::string &name, bool strided,
                               const std::vector<Dim> &shape, const std::string &defaultLd) {
               out << SP << "char " << opName << "_trans" << tag << "_rt = " << opName << "_trans" << tag << ";\n";
               out << SP << "int " << opName << "_ld" << tag << " = " << defaultLd << ";\n";
               if (!strided)
                  return;
               const std::string st = "stride_" + opName + "_" + tag;
               const std::string rows = "static_cast<size_t>(" + shape[0].GetVal() + ")";
               const std::string cols = "static_cast<size_t>(" + shape[1].GetVal() + ")";
               out << GenerateInputStrideCode(opName + "_" + tag, name, shape);
               out << SP << "if (" << st << "[1] == 1) {\n";
               out << SP << SP << opName << "_ld" << tag << " = static_cast<int>(std::max<size_t>(" << st << "[0], " << cols << "));\n";
               out << SP << "} else if (" << st << "[0] == 1) {\n";
               out << SP << SP << opName << "_trans" << tag << "_rt = (" << opName << "_trans" << tag << " == 'n') ? 't' : 'n';\n";
               out << SP << SP << opName << "_ld" << tag << " = static_cast<int>(std::max<size_t>(" << st << "[1], " << rows << "));\n";
               out << SP << "} else {\n";
               out << SP << SP << "throw std::runtime_error(\"SOFIE Gemm Op " << opName << " - input tensor " << name
                   << " has strides {\" + std::to_string(" << st << "[0]) + \", \" + std::to_string(" << st
                   << "[1]) + \"} without a unit stride, which BLAS cannot handle\");\n";
               out << SP << "}\n";
            };
            resolve("A", fNA, fStridedA, fShapeA, fAttrTransA ? opName + "_m" : opName + "_k");
            resolve("B", fNB, fStridedB, fShapeB, fAttrTransB ? opName + "_k" : opName + "_n");
            out << SP << "blas.gemmStridedBatched("
                << opName << "_transB_rt, " << opName << "_transA_rt, "
                << opName << "_n, " << opName << "_m, " << opName << "_k, " << opName << "_alpha, "
                << pB << ", " << opName << "_ldB, 0, "
                << pA << ", " << opName << "_ldA, 0, "
                << opName << "_beta, " << pY << ", " << opName << "_n, 0, 1);\n";
            if (!fNC.empty()) {
               out << SP << "auto const elementsPerGrid_" << opName << "_bias = Vec::all(Idx{static_cast<Idx>(" << lengthGemm << ")});\n";
               out << SP << "auto const workDiv_" << opName << "_bias = sofie_workdiv(elementsPerGrid_" << opName << "_bias);\n";
               out << SP << "auto task_" << opName << "_bias = alpaka::createTaskKernel<Acc>(workDiv_" << opName << "_bias, gemmBiasAddKernel, "
                   << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
                   << "alpaka::getPtrNative(deviceBuf_" << fNC << "), "
                   << "static_cast<std::size_t>(" << m << "), static_cast<std::size_t>(" << n << "), "
                   << "static_cast<std::size_t>(" << sC[0].GetVal() << "), static_cast<std::size_t>(" << sC[1].GetVal() << "), "
                   << "false, "
                   << (fActivation == EActivationType::RELU ? "true" : "false") << ", "
                   << "static_cast<Idx>(" << lengthGemm << "));\n";
               out << SP << "alpaka::enqueue(queue, task_" << opName << "_bias);\n";
            }
         } else if (fLowRank) {
            // ----------------------------------------------------------------
            // low rank factorized GEMM: B (k x n) ~= Bin (k x rank) * Bout (rank x n).
            // Restricted (in Initialize()) to the plain 2D case (no MatMul batch
            // stacking, untransposed B), so doStackMul/batchCollapseB/useSBatched
            // are always false here and the two chained calls below use the
            // plain (non-batched) blas entry points, exactly like the m/n/k
            // (no-stacking) branches above.
            //   tmp (m x rank) = alpha * op(A) * Bin
            //   Y   (m x n)    = 1 * tmp * Bout (+ bias, via epilogue as usual)
            // ----------------------------------------------------------------
            size_t tmpSize = static_cast<size_t>(std::stoi(m)) * fLowRankRank;
            out << SP << "auto buf_" << opName << "_lrtmp = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{"
                << tmpSize << "}));\n";
            out << SP << "auto tensor_" << opName << "_lrtmp = alpaka::getPtrNative(buf_" << opName << "_lrtmp);\n";

            // step 1: tmp = alpha * op(A) * Bin. Bin is always stored logically
            // untransposed (see Initialize(): when the original B was transB==1,
            // its data was transposed once before factorizing), regardless of the
            // original op's transB attribute, so transB is hardcoded 'n' here.
            out << SP << "blas.matmul("
                << "'n', " << opName << "_transA, "
                << fLowRankRank << ", " << opName << "_m, " << opName << "_k, "
                << opName << "_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_" << fLowRankInName << ")), " << pA
                << ", 0.f, tensor_" << opName << "_lrtmp);\n";

            // step 2: Y = 1 * tmp * Bout (+ bias). tmp and Bout are both untransposed
            // by construction, so transA/transB are hardcoded to 'n' here.
            if (!fNC.empty()) {
               std::string pC = "alpaka::getPtrNative(deviceBuf_" + fNC + ")";
               const char *callFn = (fActivation == EActivationType::RELU) ? "blas.gemmrelu(" : "blas.gemm(";
               out << SP << callFn << "'n', 'n', "
                   << opName << "_n, " << opName << "_m, " << fLowRankRank << ", "
                   << "1.f, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_" << fLowRankOutName << ")), static_cast<const float*>(tensor_" << opName << "_lrtmp), "
                   << opName << "_beta, " << pC << ", " << pY << ");\n";
            } else {
               out << SP << "blas.matmul('n', 'n', "
                   << opName << "_n, " << opName << "_m, " << fLowRankRank << ", "
                   << "1.f, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_" << fLowRankOutName << ")), static_cast<const float*>(tensor_" << opName << "_lrtmp), "
                   << opName << "_beta, " << pY << ");\n";
            }
         } else if (useSBatched) {
            // ----------------------------------------------------------------
            // gemmStridedBatched: both A and B vary per batch (e.g. per attention
            // head), and there is no bias.  Uses cublasSgemmStridedBatched via
            // the legacy cuBLAS handle so all N GEMMs are issued in one driver call.
            //
            // sofieBLAS convention (column-major transpose trick):
            //   transa_sofie = transB_onnx,  transb_sofie = transA_onnx
            //   m_sofie      = n_onnx,        n_sofie      = m_onnx
            //   A_sofie      = fNB,           B_sofie      = fNA
            //   lda = transA_sofie ? k : m_sofie
            //   ldb = transB_sofie ? n_sofie : k
            //   ldc = m_sofie  (leading dim of C)
            // ----------------------------------------------------------------
            size_t m_sofie = static_cast<size_t>(std::stoi(n)); // ONNX n
            size_t n_sofie = static_cast<size_t>(std::stoi(m)); // ONNX m
            size_t k_val = static_cast<size_t>(std::stoi(k));

            // cuBLAS receives the ONNX B operand first, so its transpose flag is fAttrTransB.
            // It receives the ONNX A operand second, so its transpose flag is fAttrTransA.
            size_t lda = fAttrTransB ? k_val : m_sofie;
            size_t ldb = fAttrTransA ? n_sofie : k_val;
            size_t ldc = m_sofie;
            size_t sA         = m_sofie * k_val;     // stride per batch for fNB
            size_t sB         = k_val  * n_sofie;    // stride per batch for fNA
            size_t sC         = m_sofie * n_sofie;   // stride per batch for fNY
            size_t batchCount = static_cast<size_t>(std::stoi(lengthExtra));
            out << SP << "blas.gemmStridedBatched("
                << opName << "_transB, " << opName << "_transA, "
                << m_sofie << ", " << n_sofie << ", " << k_val << ", "
                << opName << "_alpha, "
                << "alpaka::getPtrNative(deviceBuf_" << fNB << "), "
                << lda << ", " << sA << ", "
                << "alpaka::getPtrNative(deviceBuf_" << fNA << "), "
                << ldb << ", " << sB << ", "
                << opName << "_beta, "
                << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
                << ldc << ", " << sC << ", "
                << batchCount << ");\n";
         } else if (useBatchedBias) {
            size_t m_sofie = static_cast<size_t>(std::stoi(n));
            size_t n_sofie = static_cast<size_t>(std::stoi(m));
            size_t k_val = static_cast<size_t>(std::stoi(k));

            size_t lda = fAttrTransB ? k_val : m_sofie;
            size_t ldb = fAttrTransA ? n_sofie : k_val;
            size_t ldc = m_sofie;
            size_t sAb        = m_sofie * k_val;
            size_t sBb        = k_val  * n_sofie;
            size_t sCb        = m_sofie * n_sofie;
            size_t batchCount = static_cast<size_t>(std::stoi(lengthExtra));
            out << SP << "blas.gemmStridedBatched("
                << opName << "_transB, " << opName << "_transA, "
                << m_sofie << ", " << n_sofie << ", " << k_val << ", "
                << opName << "_alpha, "
                << "alpaka::getPtrNative(deviceBuf_" << fNB << "), "
                << lda << ", " << sAb << ", "
                << "alpaka::getPtrNative(deviceBuf_" << fNA << "), "
                << ldb << ", " << sBb << ", "
                << opName << "_beta, "
                << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
                << ldc << ", " << sCb << ", "
                << batchCount << ");\n";

            out << SP << "auto const elementsPerGrid_" << opName << "_bias = Vec::all(Idx{static_cast<Idx>((" << lengthExtra << ") * (" << lengthGemm << "))});\n";
            out << SP << "auto const workDiv_" << opName << "_bias = sofie_workdiv(elementsPerGrid_" << opName << "_bias);\n";
            out << SP << "auto task_" << opName << "_bias = alpaka::createTaskKernel<Acc>(workDiv_" << opName << "_bias, gemmBiasAddKernel, "
                << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
                << "alpaka::getPtrNative(deviceBuf_" << fNC << "), "
                << "static_cast<std::size_t>(" << m << "), static_cast<std::size_t>(" << n << "), "
                << "static_cast<std::size_t>(" << sC[0].GetVal() << "), static_cast<std::size_t>(" << sC[1].GetVal() << "), "
                << (haveExtraC ? "true" : "false") << ", "
                << (fActivation == EActivationType::RELU ? "true" : "false") << ", "
                << "static_cast<Idx>((" << lengthExtra << ") * (" << lengthGemm << ")));\n";
            out << SP << "alpaka::enqueue(queue, task_" << opName << "_bias);\n";
         } else if (!fNC.empty()) {
            std::string call_m = batchCollapseB
               ? std::to_string(static_cast<size_t>(std::stoi(m)) * static_cast<size_t>(std::stoi(lengthExtra)))
               : (opName + "_m");

            std::string pC = "alpaka::getPtrNative(deviceBuf_" + fNC + ")";
            if (useSerialLoop && !strideCExpr.empty())
               pC += " + i * (" + strideCExpr + ")";
            if (fActivation == EActivationType::RELU) {
               out << SP << "blas.gemmrelu("
                   << opName << "_transB, " << opName << "_transA, "
                   << opName << "_n, "      << call_m << ", "
                   << opName << "_k, "      << opName << "_alpha, "
                   << pB << ", " << pA << ", "
                   << opName << "_beta, " << pC << ", " << pY << ");\n";
            } else {
               out << SP << "blas.gemm("
                   << opName << "_transB, " << opName << "_transA, "
                   << opName << "_n, "      << call_m << ", "
                   << opName << "_k, "      << opName << "_alpha, "
                   << pB << ", " << pA << ", "
                   << opName << "_beta, " << pC << ", " << pY << ");\n";
            }
         } else {
            // ----------------------------------------------------------------
            // Pure MatMul (no bias):  Y = alpha * op(A) * op(B)
            // This covers Scaled Dot-Product Attention and other no-bias matrix multiplication
            // ----------------------------------------------------------------
            std::string call_m = batchCollapseB
               ? std::to_string(static_cast<size_t>(std::stoi(m)) * static_cast<size_t>(std::stoi(lengthExtra)))
               : (opName + "_m");

            out << SP << "blas.matmul("
                << opName << "_transB, " << opName << "_transA, "
                << opName << "_n, "      << call_m << ", "
                << opName << "_k, "      << opName << "_alpha, "
                << pB << ", " << pA << ", "
                << opName << "_beta, "  << pY << ");\n";
         }

         if (useSerialLoop) {
            out << SP << "}\n"; // end of loop on the stacked multiplication
         }

         // GEMM+LeakyReLU fusion (GPU): cuBLASLt has no native LeakyReLU epilogue,
         // so we emit a cheap in-place ALPAKA kernel immediately after the GEMM.
         // This avoids allocating a separate intermediate output buffer, but still
         // emits one ALPAKA activation kernel after the GEMM call.
         if (fActivation == EActivationType::LEAKYRELU) {
            std::string numElem = ConvertDimShapeToLength(fShapeY);
            out << SP << "//--- GEMM+LeakyReLU in-place fusion\n";
            out << SP << "{\n";
            out << SP << SP << "constexpr float " << opName << "_lrelu_alpha = "
                << std::setprecision(std::numeric_limits<float>::max_digits10)
                << fLeakyReluAlpha << "f;\n";
            out << SP << SP << "auto const elementsPerThread_lrelu_" << opName
                << " = Vec::all(static_cast<Idx>(1));\n";
            out << SP << SP << "auto const elementsPerGrid_lrelu_" << opName
                << " = Vec::all(Idx{" << numElem << "});\n";
            out << SP << SP << "auto const workDiv_lrelu_" << opName
                << " = sofie_workdiv(elementsPerGrid_lrelu_" << opName << ");\n";
            // In-place: input and output pointer are the same device buffer.
            out << SP << SP << "auto task_lrelu_" << opName
                << " = alpaka::createTaskKernel<Acc>(workDiv_lrelu_" << opName
                << ", leakyReluKernel"
                << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
                << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
                << ", static_cast<Idx>(" << numElem << ")"
                << ", static_cast<float>(" << opName << "_lrelu_alpha));\n";
            out << SP << SP << "alpaka::enqueue(queue, task_lrelu_" << opName << ");\n";
            out << SP << "}\n";
         }

         return out.str();
      }

      bool SupportsStridedInput() const override { return true; }

      std::vector<std::string> GetBlasRoutines() override { return { std::string("Gemm"), std::string("Gemv") }; }
      std::string GetFusableOutputTensorName() override {
         return fNY;
      }

      void UpdateFusableTensorName(std::string fusable_tensor_name, const std::function<void(const std::string&)>& removal_func){
         removal_func(fNY);
         fNY = fusable_tensor_name;
         fOutputTensorNames[0] = fNY;
      }

      // --- Activation fusion accessors (used by FusionPlanner) ---
      EActivationType GetActivationType() const { return fActivation; }
      bool HasBias() const { return !fNC.empty(); }

      /// Set fused activation.  alpha is only meaningful for LEAKYRELU.
      void SetActivation(EActivationType act, float alpha = 0.f) {
         fActivation      = act;
         fLeakyReluAlpha  = alpha;
      }

      std::string GetBlasConfig(){
         int64_t dimA = fShapeA.size();
         int64_t dimB = fShapeB.size();
         int64_t dimY = fShapeY.size();
         auto m = (fAttrTransA ? fShapeA[dimA-1].GetVal() : fShapeA[dimA-2].GetVal());
         auto n = (fAttrTransB ? fShapeB[dimB-2].GetVal() : fShapeB[dimB-1].GetVal());
         auto k = (fAttrTransA ? fShapeA[dimA-2].GetVal() : fShapeA[dimA-1].GetVal());
         auto lda = (fAttrTransA ? m : k);
         auto ldb = (fAttrTransB ? k : n);
         auto ldc = n;
         std::string transFlags = std::string(fAttrTransB ? "'t'" : "'n'") + ", " + (fAttrTransA ? "'t'" : "'n'");
         std::string epilogue = fNC.empty() ? "Epilogue::Default"
                                            : (fActivation == EActivationType::RELU ? "Epilogue::ReluBias" : "Epilogue::Bias");
         if (dimY > 2 && !fIsDynamic) {
            std::vector<Dim> sExtra;
            for (int64_t i = 0; i < dimY - 2; i++) sExtra.push_back(fShapeY[i]);
            auto lengthExtra = ConvertDimShapeToLength(sExtra);
            if (std::stoi(lengthExtra) > 1) {
               bool bLeadingDimsAllOne = true;
               for (int64_t i = 0; i < dimB - 2; i++) {
                  if (fShapeB[i].dim != 1) { bLeadingDimsAllOne = false; break; }
               }
               if (bLeadingDimsAllOne) {
                  // batch-collapse: register layout for the full-batch GEMM
                  auto m_batched = std::to_string(std::stoi(m) * std::stoi(lengthExtra));
                  return n+", "+m_batched+", "+k+", "+ldb+", "+lda+", "+ldc+", "+transFlags+", "+epilogue;
               } else if (fNC.empty()) {
                  return "";
               }
            }
         }

         return n+", "+m+", "+k+", "+ldb+", "+lda+", "+ldc+", "+transFlags+", "+epilogue;
      }

      std::vector<std::string> GetBlasConfigs() override {
         if (!fLowRank)
            return ROperator::GetBlasConfigs();

         int64_t dimA = fShapeA.size();
         auto m = (fAttrTransA ? fShapeA[dimA-1].GetVal() : fShapeA[dimA-2].GetVal());
         auto k = (fAttrTransA ? fShapeA[dimA-2].GetVal() : fShapeA[dimA-1].GetVal());
         int64_t dimB = fShapeB.size();
         auto n = (fAttrTransB ? fShapeB[dimB-2].GetVal() : fShapeB[dimB-1].GetVal());
         std::string rankStr = std::to_string(fLowRankRank);
         std::string transFlags1 = std::string("'n', ") + (fAttrTransA ? "'t'" : "'n'");
         auto lda1 = (fAttrTransA ? m : k);

         std::string epilogue = fNC.empty() ? "Epilogue::Default"
                                            : (fActivation == EActivationType::RELU ? "Epilogue::ReluBias" : "Epilogue::Bias");

         // step 1: tmp (m x rank) = op(A) * Bin
         std::string cfg1 = rankStr+", "+m+", "+k+", "+rankStr+", "+lda1+", "+rankStr+", "+transFlags1+", Epilogue::Default";

         // step 2: Y (m x n) = tmp * Bout (+ bias)
         std::string cfg2 = n+", "+m+", "+rankStr+", "+n+", "+rankStr+", "+n+", 'n', 'n', "+epilogue;

         return {cfg1, cfg2};
      }
   };


}//SOFIE

#endif //SOFIE_ROPERATOR_GEMM
