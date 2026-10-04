#ifndef SOFIE_ROPERATOR_SDPA
#define SOFIE_ROPERATOR_SDPA

#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include <sstream>
#include <string>
#include <cmath>

namespace SOFIE {

template <typename T>
class ROperator_SDPA : public ROperator {
private:
   std::string fNQ, fNK, fNV, fNMask, fNY;
   float fScale = 0.0f;  // 0 → use 1/sqrt(D)
   size_t fNumHeads = 0; // only used when Q/K/V are rank-3 (folded-head layout)

   std::vector<Dim> fShapeQ;
   std::string fB, fH, fS, fD, fDv;
   std::string fType;
   bool fHasMask = false;

   bool fFolded = false;
   std::string fQKBatchStride, fQKHeadStride, fQKSeqStride;
   std::string fVYBatchStride, fVYHeadStride, fVYSeqStride;

   static constexpr size_t kSdpaBlockSize = 256;

public:
   ROperator_SDPA() {}

   ROperator_SDPA(const std::string &nameQ,
                  const std::string &nameK,
                  const std::string &nameV,
                  const std::string &nameY,
                  const std::string &nameMask = "",
                  float scale = 0.0f,
                  size_t numHeads = 0)
      : fNQ(UTILITY::Clean_name(nameQ)),
        fNK(UTILITY::Clean_name(nameK)),
        fNV(UTILITY::Clean_name(nameV)),
        fNMask(UTILITY::Clean_name(nameMask)),
        fNY(UTILITY::Clean_name(nameY)),
        fScale(scale),
        fNumHeads(numHeads)
   {
      fKind = OperatorKind::SDPA;
      fInputTensorNames  = { fNQ, fNK, fNV };
      if (!fNMask.empty()) fInputTensorNames.push_back(fNMask);
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNQ))
         throw std::runtime_error("SOFIE SDPA: query tensor " + fNQ + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNK))
         throw std::runtime_error("SOFIE SDPA: key tensor " + fNK + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNV))
         throw std::runtime_error("SOFIE SDPA: value tensor " + fNV + " not found");

      fShapeQ = model.GetDimTensorShape(fNQ);
      auto shapeV = model.GetDimTensorShape(fNV);
      fType = ConvertTypeToString(model.GetTensorType(fNQ));

      if (fShapeQ.size() == 4) {
         fFolded = false;
         fB  = fShapeQ[0].GetVal();
         fH  = fShapeQ[1].GetVal();
         fS  = fShapeQ[2].GetVal();
         fD  = fShapeQ[3].GetVal();
         fDv = shapeV[3].GetVal();

         model.AddIntermediateTensor(fNY, model.GetTensorType(fNQ),
                                     { fShapeQ[0], fShapeQ[1], fShapeQ[2], shapeV[3] });
      } else if (fShapeQ.size() == 3) {
         if (fNumHeads == 0)
            throw std::runtime_error("SOFIE SDPA: rank-3 [B, S, H*D] query requires num_heads "
                                     "to be specified (it cannot be inferred from the tensor shape)");

         fB = fShapeQ[0].GetVal();
         fS = fShapeQ[1].GetVal();
         fH = std::to_string(fNumHeads);

         auto headDim = [this](const Dim &total) {
            if (!total.isParam)
               return std::to_string(total.dim / fNumHeads);
            return "(" + total.GetVal() + ") / " + std::to_string(fNumHeads);
         };
         fD  = headDim(fShapeQ[2]);
         fDv = headDim(shapeV[2]);
         fFolded = true;

         // output [B, S, H*Dv] — same folded convention as the inputs
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNQ),
                                     { fShapeQ[0], fShapeQ[1], shapeV[2] });
      } else {
         throw std::runtime_error("SOFIE SDPA: query must be rank-4 [B, H, S, D] or "
                                  "rank-3 [B, S, H*D] with num_heads specified");
      }

      if (fFolded) {
         fQKBatchStride = "(" + fS + ") * (" + fH + ") * (" + fD + ")";
         fQKHeadStride  = fD;
         fQKSeqStride   = "(" + fH + ") * (" + fD + ")";
         fVYBatchStride = "(" + fS + ") * (" + fH + ") * (" + fDv + ")";
         fVYHeadStride  = fDv;
         fVYSeqStride   = "(" + fH + ") * (" + fDv + ")";
      } else {
         fQKBatchStride = "(" + fH + ") * (" + fS + ") * (" + fD + ")";
         fQKHeadStride  = "(" + fS + ") * (" + fD + ")";
         fQKSeqStride   = fD;
         fVYBatchStride = "(" + fH + ") * (" + fS + ") * (" + fDv + ")";
         fVYHeadStride  = "(" + fS + ") * (" + fDv + ")";
         fVYSeqStride   = fDv;
      }

      fHasMask = !fNMask.empty() && model.CheckIfTensorAlreadyExist(fNMask);

      model.AddNeededStdLib("cmath");
      model.AddNeededStdLib("limits");
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      std::stringstream out;
      // If scale not specified use 1/sqrt(D) — emit as expression
      std::string scaleExpr = (fScale != 0.0f)
         ? (fType + "(" + std::to_string(fScale) + ")")
         : ("static_cast<" + fType + ">(1) / std::sqrt(static_cast<" + fType + ">(" + fD + "))");

      out << "\n//---- SDPA " << opName << "\n";
      out << SP << "for (size_t b = 0; b < " << fB << "; ++b)\n";
      out << SP << "for (size_t h = 0; h < " << fH << "; ++h)\n";
      out << SP << "for (size_t s = 0; s < " << fS << "; ++s) {\n";
      out << SP << SP << "size_t qBase = b*(" << fQKBatchStride << ") + h*(" << fQKHeadStride
          << ") + s*(" << fQKSeqStride << ");\n";
      // Compute scores
      out << SP << SP << "std::vector<" << fType << "> scores(" << fS << ");\n";
      out << SP << SP << "for (size_t j = 0; j < " << fS << "; ++j) {\n";
      out << SP << SP << SP << "size_t kBase = b*(" << fQKBatchStride << ") + h*(" << fQKHeadStride
          << ") + j*(" << fQKSeqStride << ");\n";
      out << SP << SP << SP << fType << " dot = 0;\n";
      out << SP << SP << SP << "for (size_t d = 0; d < " << fD << "; ++d)\n";
      out << SP << SP << SP << SP << "dot += tensor_" << fNQ << "[qBase+d] * tensor_" << fNK << "[kBase+d];\n";
      out << SP << SP << SP << "scores[j] = dot * " << scaleExpr << ";\n";
      if (fHasMask)
         out << SP << SP << SP << "scores[j] += tensor_" << fNMask
             << "[b*" << fH << "*" << fS << "*" << fS << " + h*" << fS << "*" << fS
             << " + s*" << fS << " + j];\n";
      out << SP << SP << "}\n";
      // Softmax
      out << SP << SP << fType << " maxScore = *std::max_element(scores.begin(), scores.end());\n";
      out << SP << SP << fType << " sumExp = 0;\n";
      out << SP << SP << "for (size_t j = 0; j < " << fS << "; ++j) { scores[j] = std::exp(scores[j]-maxScore); sumExp += scores[j]; }\n";
      out << SP << SP << "for (size_t j = 0; j < " << fS << "; ++j) scores[j] /= sumExp;\n";
      // Weighted sum over V
      out << SP << SP << "for (size_t d = 0; d < " << fDv << "; ++d) {\n";
      out << SP << SP << SP << fType << " acc = 0;\n";
      out << SP << SP << SP << "for (size_t j = 0; j < " << fS << "; ++j)\n";
      out << SP << SP << SP << SP << "acc += scores[j] * tensor_" << fNV
          << "[b*(" << fVYBatchStride << ") + h*(" << fVYHeadStride << ") + j*(" << fVYSeqStride << ") + d];\n";
      out << SP << SP << SP << "tensor_" << fNY
          << "[b*(" << fVYBatchStride << ") + h*(" << fVYHeadStride << ") + s*(" << fVYSeqStride << ") + d] = acc;\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";
      return out.str();
   }

   // Tile size (in K/V rows) for the shared-memory-tiled kernel below, chosen
   // at codegen time to keep the K+V tile comfortably within a block's shared
   // memory budget regardless of head_dim, while still amortizing well.
   static size_t TileRowsFor(int dTotal, size_t elemSize, size_t blockSize) {
      const size_t budgetBytes = 12288; // ~12KB combined K+V tile
      size_t rows = budgetBytes / (static_cast<size_t>(dTotal) * elemSize);
      if (rows > blockSize) rows = blockSize;
      if (rows < 8) rows = 8;
      return rows;
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "SDPAKernel_" + opName;

      // The tiled kernel below needs D/Dv (and hence the shared-memory tile
      // and per-thread accumulator sizes) fixed at compile time. Dynamic
      // head dims are rare (num_heads is normally a static graph attribute);
      // fall back to the original one-thread-per-row kernel in that case.
      bool canTile = IsInteger(fD) && IsInteger(fDv);

      if (canTile) {
         const size_t blockSize = kSdpaBlockSize;
         int dInt  = std::stoi(fD);
         int dvInt = std::stoi(fDv);
         size_t elemSize = (fType == "double") ? 8 : 4;
         size_t tileRows = TileRowsFor(dInt + dvInt, elemSize, blockSize);

         std::string out;
         out  = "\n//------ SDPA_KERNEL_ALPAKA (shared-memory K/V tiled)\n";
         out += SP + "struct " + kname + " {\n";
         out += SP + SP + "template<typename TAcc, typename T>\n";
         out += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
         out += SP + SP + SP + "TAcc const& acc,\n";
         out += SP + SP + SP + "T const* __restrict__ Q,\n";
         out += SP + SP + SP + "T const* __restrict__ K,\n";
         out += SP + SP + SP + "T const* __restrict__ V,\n";
         out += (fHasMask ? (SP + SP + SP + "T const* __restrict__ mask,\n") : "");
         out += SP + SP + SP + "T* __restrict__ Y,\n";
         out += SP + SP + SP + "std::size_t const B,\n";
         out += SP + SP + SP + "std::size_t const H,\n";
         out += SP + SP + SP + "std::size_t const S,\n";
         out += SP + SP + SP + "std::size_t const /*D*/,\n";
         out += SP + SP + SP + "std::size_t const /*Dv*/,\n";
         out += SP + SP + SP + "T const scale) const {\n\n";

         out += SP + SP + SP + "constexpr std::size_t kD = " + std::to_string(dInt) + ";\n";
         out += SP + SP + SP + "constexpr std::size_t kDv = " + std::to_string(dvInt) + ";\n";
         out += SP + SP + SP + "constexpr std::size_t kTileRows = " + std::to_string(tileRows) + ";\n";
         out += SP + SP + SP + "constexpr std::size_t kBlockSize = " + std::to_string(blockSize) + ";\n\n";

         out += SP + SP + SP + "auto& kTile = alpaka::declareSharedVar<T[kTileRows * kD], __COUNTER__>(acc);\n";
         out += SP + SP + SP + "auto& vTile = alpaka::declareSharedVar<T[kTileRows * kDv], __COUNTER__>(acc);\n\n";

         out += SP + SP + SP + "auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
         out += SP + SP + SP + "auto const blockIdxFlat = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
         out += SP + SP + SP + "std::size_t const qTilesPerBH = (S + kBlockSize - 1) / kBlockSize;\n";
         out += SP + SP + SP + "std::size_t const bh = blockIdxFlat / qTilesPerBH;\n";
         out += SP + SP + SP + "std::size_t const qTileIdx = blockIdxFlat % qTilesPerBH;\n";
         out += SP + SP + SP + "std::size_t const b = bh / H;\n";
         out += SP + SP + SP + "std::size_t const h = bh % H;\n";
         out += SP + SP + SP + "std::size_t const s = qTileIdx * kBlockSize + tid;\n";
         out += SP + SP + SP + "bool const active = s < S;\n\n";

         const std::string qkBatch = fFolded ? "S*H*kD"  : "H*S*kD";
         const std::string qkHead  = fFolded ? "kD"       : "S*kD";
         const std::string qkSeq   = fFolded ? "H*kD"      : "kD";
         const std::string vyBatch = fFolded ? "S*H*kDv" : "H*S*kDv";
         const std::string vyHead  = fFolded ? "kDv"       : "S*kDv";
         const std::string vySeq   = fFolded ? "H*kDv"     : "kDv";
         const std::string qBaseExpr = "b*(" + qkBatch + ") + h*(" + qkHead + ") + s*(" + qkSeq + ")";
         const std::string kBaseExpr = "b*(" + qkBatch + ") + h*(" + qkHead + ") + j*(" + qkSeq + ")";
         const std::string vBaseExpr = "b*(" + vyBatch + ") + h*(" + vyHead + ") + j*(" + vySeq + ")";
         const std::string ySBaseExpr = "b*(" + vyBatch + ") + h*(" + vyHead + ") + s*(" + vySeq + ")";

         out += SP + SP + SP + "T q_local[kD];\n";
         out += SP + SP + SP + "T acc_v[kDv] = {};\n";
         out += SP + SP + SP + "T max_score = -std::numeric_limits<T>::max();\n";
         out += SP + SP + SP + "T sum_exp = static_cast<T>(0);\n";
         out += SP + SP + SP + "if (active) {\n";
         out += SP + SP + SP + SP + "std::size_t const qBase = " + qBaseExpr + ";\n";
         out += SP + SP + SP + SP + "for (std::size_t d = 0; d < kD; ++d) q_local[d] = Q[qBase + d];\n";
         out += SP + SP + SP + "}\n\n";

         out += SP + SP + SP + "// Sweep the whole K/V sequence in shared-memory tiles: the whole\n";
         out += SP + SP + SP + "// block cooperatively loads each tile once and every thread (one\n";
         out += SP + SP + SP + "// per query row) reuses it, instead of each of the S query rows\n";
         out += SP + SP + SP + "// independently re-reading all of K/V from global memory.\n";
         out += SP + SP + SP + "for (std::size_t tileStart = 0; tileStart < S; tileStart += kTileRows) {\n";
         out += SP + SP + SP + SP + "std::size_t const tileLen = (tileStart + kTileRows <= S) ? kTileRows : (S - tileStart);\n\n";
         out += SP + SP + SP + SP + "for (std::size_t idx = tid; idx < tileLen * kD; idx += kBlockSize) {\n";
         out += SP + SP + SP + SP + SP + "std::size_t const row = idx / kD, d = idx % kD;\n";
         out += SP + SP + SP + SP + SP + "std::size_t const j = tileStart + row;\n";
         out += SP + SP + SP + SP + SP + "kTile[idx] = K[" + kBaseExpr + " + d];\n";
         out += SP + SP + SP + SP + "}\n";
         out += SP + SP + SP + SP + "for (std::size_t idx = tid; idx < tileLen * kDv; idx += kBlockSize) {\n";
         out += SP + SP + SP + SP + SP + "std::size_t const row = idx / kDv, d = idx % kDv;\n";
         out += SP + SP + SP + SP + SP + "std::size_t const j = tileStart + row;\n";
         out += SP + SP + SP + SP + SP + "vTile[idx] = V[" + vBaseExpr + " + d];\n";
         out += SP + SP + SP + SP + "}\n";
         out += SP + SP + SP + SP + "alpaka::syncBlockThreads(acc);\n\n";

         out += SP + SP + SP + SP + "if (active) {\n";
         out += SP + SP + SP + SP + SP + "for (std::size_t row = 0; row < tileLen; ++row) {\n";
         out += SP + SP + SP + SP + SP + SP + "std::size_t const j = tileStart + row;\n";
         out += SP + SP + SP + SP + SP + SP + "T dot = static_cast<T>(0);\n";
         out += SP + SP + SP + SP + SP + SP + "for (std::size_t d = 0; d < kD; ++d) dot += q_local[d] * kTile[row * kD + d];\n";
         out += SP + SP + SP + SP + SP + SP + "T sc = dot * scale;\n";
         if (fHasMask)
            out += SP + SP + SP + SP + SP + SP + "sc += mask[b*H*S*S + h*S*S + s*S + j];\n";
         out += SP + SP + SP + SP + SP + SP + "T const new_max = (sc > max_score) ? sc : max_score;\n";
         out += SP + SP + SP + SP + SP + SP + "T const correction = alpaka::math::exp(acc, max_score - new_max);\n";
         out += SP + SP + SP + SP + SP + SP + "T const p = alpaka::math::exp(acc, sc - new_max);\n";
         out += SP + SP + SP + SP + SP + SP + "sum_exp = sum_exp * correction + p;\n";
         out += SP + SP + SP + SP + SP + SP + "for (std::size_t d = 0; d < kDv; ++d)\n";
         out += SP + SP + SP + SP + SP + SP + SP + "acc_v[d] = acc_v[d] * correction + p * vTile[row * kDv + d];\n";
         out += SP + SP + SP + SP + SP + SP + "max_score = new_max;\n";
         out += SP + SP + SP + SP + SP + "}\n";
         out += SP + SP + SP + SP + "}\n";
         out += SP + SP + SP + SP + "alpaka::syncBlockThreads(acc);\n"; // before next tile overwrites shared mem
         out += SP + SP + SP + "}\n\n";

         out += SP + SP + SP + "if (active) {\n";
         out += SP + SP + SP + SP + "std::size_t const yBase = " + ySBaseExpr + ";\n";
         out += SP + SP + SP + SP + "for (std::size_t d = 0; d < kDv; ++d)\n";
         out += SP + SP + SP + SP + SP + "Y[yBase + d] = acc_v[d] / sum_exp;\n";
         out += SP + SP + SP + "}\n";
         out += SP + SP + "}\n" + SP + "};\n";
         return out;
      }

      std::string out;
      out  = "\n//------ SDPA_KERNEL_ALPAKA\n";
      out += SP + "struct " + kname + " {\n";
      out += SP + SP + "template<typename TAcc, typename T>\n";
      out += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      out += SP + SP + SP + "TAcc const& acc,\n";
      out += SP + SP + SP + "T const* __restrict__ Q,\n";
      out += SP + SP + SP + "T const* __restrict__ K,\n";
      out += SP + SP + SP + "T const* __restrict__ V,\n";
      out += (fHasMask ? (SP + SP + SP + "T const* __restrict__ mask,\n") : "");
      out += SP + SP + SP + "T* __restrict__ Y,\n";
      out += SP + SP + SP + "std::size_t const B,\n";
      out += SP + SP + SP + "std::size_t const H,\n";
      out += SP + SP + SP + "std::size_t const S,\n";
      out += SP + SP + SP + "std::size_t const D,\n";
      out += SP + SP + SP + "std::size_t const Dv,\n";
      out += SP + SP + SP + "T const scale) const {\n\n";

      out += SP + SP + SP + "auto const global_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      out += SP + SP + SP + "if (global_idx >= B * H * S) return;\n";
      out += SP + SP + SP + "std::size_t const b = global_idx / (H * S);\n";
      out += SP + SP + SP + "std::size_t const h = (global_idx / S) % H;\n";
      out += SP + SP + SP + "std::size_t const s = global_idx % S;\n\n";

      const std::string qkBatch = fFolded ? "S*H*D"  : "H*S*D";
      const std::string qkHead  = fFolded ? "D"       : "S*D";
      const std::string qkSeq   = fFolded ? "H*D"      : "D";
      const std::string vyBatch = fFolded ? "S*H*Dv" : "H*S*Dv";
      const std::string vyHead  = fFolded ? "Dv"       : "S*Dv";
      const std::string vySeq   = fFolded ? "H*Dv"     : "Dv";
      const std::string qBaseExpr = "b*(" + qkBatch + ") + h*(" + qkHead + ") + s*(" + qkSeq + ")";
      const std::string kBaseExpr = "b*(" + qkBatch + ") + h*(" + qkHead + ") + j*(" + qkSeq + ")";
      const std::string vBaseExpr = "b*(" + vyBatch + ") + h*(" + vyHead + ") + j*(" + vySeq + ")";
      const std::string ySBaseExpr = "b*(" + vyBatch + ") + h*(" + vyHead + ") + s*(" + vySeq + ")";

      out += SP + SP + SP + "std::size_t const qBase = " + qBaseExpr + ";\n\n";

      bool canFuse = IsInteger(fDv);
      if (canFuse) {
         std::string dv = fDv;
         out += SP + SP + SP + "// Fused online-softmax pass: max, sum and weighted-V accumulator\n";
         out += SP + SP + SP + "// updated together, each rescaled when a new row max is found.\n";
         out += SP + SP + SP + "T acc_v[" + dv + "] = {};\n";
         out += SP + SP + SP + "T max_score = -std::numeric_limits<T>::max();\n";
         out += SP + SP + SP + "T sum_exp = static_cast<T>(0);\n\n";

         out += SP + SP + SP + "for (std::size_t j = 0; j < S; ++j) {\n";
         out += SP + SP + SP + SP + "std::size_t const kBase = " + kBaseExpr + ";\n";
         out += SP + SP + SP + SP + "T dot = static_cast<T>(0);\n";
         out += SP + SP + SP + SP + "for (std::size_t d = 0; d < D; ++d) dot += Q[qBase+d] * K[kBase+d];\n";
         out += SP + SP + SP + SP + "T sc = dot * scale;\n";
         if (fHasMask)
            out += SP + SP + SP + SP + "sc += mask[b*H*S*S + h*S*S + s*S + j];\n";
         out += SP + SP + SP + SP + "T const new_max = (sc > max_score) ? sc : max_score;\n";
         out += SP + SP + SP + SP + "T const correction = alpaka::math::exp(acc, max_score - new_max);\n";
         out += SP + SP + SP + SP + "T const p = alpaka::math::exp(acc, sc - new_max);\n";
         out += SP + SP + SP + SP + "sum_exp = sum_exp * correction + p;\n";
         out += SP + SP + SP + SP + "std::size_t const vBase = " + vBaseExpr + ";\n";
         out += SP + SP + SP + SP + "for (std::size_t d = 0; d < Dv; ++d)\n";
         out += SP + SP + SP + SP + SP + "acc_v[d] = acc_v[d] * correction + p * V[vBase+d];\n";
         out += SP + SP + SP + SP + "max_score = new_max;\n";
         out += SP + SP + SP + "}\n\n";
         out += SP + SP + SP + "for (std::size_t d = 0; d < Dv; ++d)\n";
         out += SP + SP + SP + SP + "Y[" + ySBaseExpr + " + d] = acc_v[d] / sum_exp;\n";
      } else {
      out += SP + SP + SP + "// scores computed in thread-local storage\n";
      out += SP + SP + SP + "T max_score = -std::numeric_limits<T>::max();\n";
      out += SP + SP + SP + "T sum_exp = static_cast<T>(0);\n\n";

      out += SP + SP + SP + "for (std::size_t j = 0; j < S; ++j) {\n";
      out += SP + SP + SP + SP + "std::size_t const kBase = " + kBaseExpr + ";\n";
      out += SP + SP + SP + SP + "T dot = static_cast<T>(0);\n";
      out += SP + SP + SP + SP + "for (std::size_t d = 0; d < D; ++d) dot += Q[qBase+d] * K[kBase+d];\n";
      out += SP + SP + SP + SP + "T sc = dot * scale;\n";
      if (fHasMask)
         out += SP + SP + SP + SP + "sc += mask[b*H*S*S + h*S*S + s*S + j];\n";
      out += SP + SP + SP + SP + "if (sc > max_score) max_score = sc;\n";
      out += SP + SP + SP + "}\n\n";

      out += SP + SP + SP + "// Pass 2: compute attn weights and weighted sum simultaneously\n";
      out += SP + SP + SP + "for (std::size_t d = 0; d < Dv; ++d) Y[" + ySBaseExpr + " + d] = static_cast<T>(0);\n\n";
      out += SP + SP + SP + "for (std::size_t j = 0; j < S; ++j) {\n";
      out += SP + SP + SP + SP + "std::size_t const kBase = " + kBaseExpr + ";\n";
      out += SP + SP + SP + SP + "T dot = static_cast<T>(0);\n";
      out += SP + SP + SP + SP + "for (std::size_t d = 0; d < D; ++d) dot += Q[qBase+d] * K[kBase+d];\n";
      out += SP + SP + SP + SP + "T sc = dot * scale;\n";
      if (fHasMask)
         out += SP + SP + SP + SP + "sc += mask[b*H*S*S + h*S*S + s*S + j];\n";
      out += SP + SP + SP + SP + "T const a_j = alpaka::math::exp(acc, sc - max_score);\n";
      out += SP + SP + SP + SP + "sum_exp += a_j;\n";
      out += SP + SP + SP + SP + "std::size_t const vBase = " + vBaseExpr + ";\n";
      out += SP + SP + SP + SP + "for (std::size_t d = 0; d < Dv; ++d)\n";
      out += SP + SP + SP + SP + SP + "Y[" + ySBaseExpr + " + d] += a_j * V[vBase+d];\n";
      out += SP + SP + SP + "}\n\n";
      out += SP + SP + SP + "// Normalize\n";
      out += SP + SP + SP + "for (std::size_t d = 0; d < Dv; ++d)\n";
      out += SP + SP + SP + SP + "Y[" + ySBaseExpr + " + d] /= sum_exp;\n";
      }
      out += SP + SP + "}\n" + SP + "};\n";
      return out;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      return SP + "SDPAKernel_" + opName + " sdpaKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string scaleExpr = (fScale != 0.0f)
         ? ("static_cast<" + fType + ">(" + std::to_string(fScale) + "f)")
         : ("static_cast<" + fType + ">(1) / sqrt(/* host-side: */ static_cast<" + fType + ">(" + fD + "))");

      // Scale is computed at dispatch time as a host-side constant passed to kernel
      std::string scaleVal = (fScale != 0.0f)
         ? ("static_cast<" + fType + ">(" + std::to_string(fScale) + "f)")
         : ("static_cast<" + fType + ">(1.0f) / std::sqrt(static_cast<" + fType + ">(" + fD + "))");

      bool canTile = IsInteger(fD) && IsInteger(fDv);

      std::stringstream out;
      out << "\n//------ SDPA_GPU_ALPAKA\n";
      out << SP << "{\n";
      out << SP << SP << fType << " const sdpaScale_" << opName << " = " << scaleVal << ";\n";
      if (canTile) {
         // One block per (batch, head, query-tile) so the K/V shared-memory
         // tile loaded by the kernel is reused across all kSdpaBlockSize
         // query rows in that block, instead of being re-read from global
         // memory independently per row.
         out << SP << SP << "std::size_t const sdpaQTiles_" << opName
             << " = (static_cast<std::size_t>(" << fS << ") + " << kSdpaBlockSize << " - 1) / " << kSdpaBlockSize << ";\n";
         out << SP << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << opName << "(\n";
         out << SP << SP << SP << "Vec::all(Idx{static_cast<Idx>(static_cast<std::size_t>(" << fB << ") * static_cast<std::size_t>(" << fH
             << ") * sdpaQTiles_" << opName << ")}),\n";
         out << SP << SP << SP << "Vec::all(Idx{" << kSdpaBlockSize << "u}),\n";
         out << SP << SP << SP << "Vec::all(Idx{1u}));\n";
      } else {
         out << SP << SP << "auto const elementsPerGrid_" << opName
             << " = Vec::all(Idx{" << fB << " * " << fH << " * " << fS << "});\n";
         out << SP << SP << "auto const workDiv_" << opName
             << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      }
      out << SP << SP << "auto task_" << opName
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", sdpaKernel_" << opName << ", "
          << "alpaka::getPtrNative(deviceBuf_" << fNQ << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNK << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNV << "), ";
      if (fHasMask)
         out << "alpaka::getPtrNative(deviceBuf_" << fNMask << "), ";
      out << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
          << "static_cast<Idx>(" << fB << "), "
          << "static_cast<Idx>(" << fH << "), "
          << "static_cast<Idx>(" << fS << "), "
          << "static_cast<Idx>(" << fD << "), "
          << "static_cast<Idx>(" << fDv << "), "
          << "sdpaScale_" << opName << ");\n";
      out << SP << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_SDPA
