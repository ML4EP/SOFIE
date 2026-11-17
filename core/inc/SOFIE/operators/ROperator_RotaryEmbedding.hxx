#ifndef SOFIE_ROPERATOR_ROTARYEMBEDDING
#define SOFIE_ROPERATOR_ROTARYEMBEDDING

#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/utils/SOFIE_common.hxx"

#include <sstream>
#include <string>

namespace SOFIE {

/// ONNX RotaryEmbedding (opset 23), the rotary positional embedding (RoPE) of https://arxiv.org/abs/2104.09864.
///
/// X is either 4d [batch, num_heads, seq, head_size] or 3d [batch, seq, num_heads * head_size] (then the
/// num_heads attribute is required). The first rotary_embedding_dim (R, default head_size) elements of every
/// head are rotated pairwise, the remaining ones are copied:
///    y1 = cos * x1 - sin * x2,   y2 = sin * x1 + cos * x2
/// where (x1, x2) are the two halves of the rotated part, or its even and odd elements if interleaved = 1.
/// cos_cache and sin_cache are [batch, seq, R/2], or [max_position, R/2] indexed with position_ids [batch, seq].
template <typename T>
class ROperator_RotaryEmbedding final : public ROperator_ALPAKA {
private:
   bool fInterleaved;
   size_t fAttrRotaryDim;
   size_t fAttrNumHeads;

   std::string fNX;
   std::string fNCos;
   std::string fNSin;
   std::string fNPositionIds; // optional
   std::string fNY;

   std::vector<Dim> fShapeX;
   bool fIs3d = false;
   Dim fBatch;
   Dim fSeq;
   size_t fNumHeads = 0;
   size_t fHeadSize = 0;
   size_t fRotaryDim = 0;
   std::string fType;

   // index expressions shared by the CPU code and the kernel, in terms of the
   // loop variables b, s, h (batch, token, head)
   std::string CacheOffset() const
   {
      const std::string half = std::to_string(fRotaryDim / 2);
      if (fNPositionIds.empty())
         return "(b * " + fSeq.GetVal() + " + s) * " + half;
      return "static_cast<std::size_t>(POS[b * " + fSeq.GetVal() + " + s]) * " + half;
   }
   std::string HeadOffset() const
   {
      const std::string H = std::to_string(fNumHeads), D = std::to_string(fHeadSize);
      if (fIs3d)
         return "((b * " + fSeq.GetVal() + " + s) * " + H + " + h) * " + D;
      return "((b * " + H + " + h) * " + fSeq.GetVal() + " + s) * " + D;
   }
   std::string FirstOfPair(const std::string &i) const { return fInterleaved ? "2 * " + i : i; }
   std::string SecondOfPair(const std::string &i) const
   {
      return fInterleaved ? "2 * " + i + " + 1" : i + " + " + std::to_string(fRotaryDim / 2);
   }

public:
   ROperator_RotaryEmbedding() {}

   ROperator_RotaryEmbedding(bool interleaved, size_t rotaryDim, size_t numHeads, const std::string &nameX,
                             const std::string &nameCos, const std::string &nameSin,
                             const std::string &namePositionIds, const std::string &nameY)
      : fInterleaved(interleaved),
        fAttrRotaryDim(rotaryDim),
        fAttrNumHeads(numHeads),
        fNX(UTILITY::Clean_name(nameX)),
        fNCos(UTILITY::Clean_name(nameCos)),
        fNSin(UTILITY::Clean_name(nameSin)),
        fNPositionIds(namePositionIds.empty() ? "" : UTILITY::Clean_name(namePositionIds)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::ROTARY_EMBEDDING;
      fInputTensorNames = {fNX, fNCos, fNSin};
      if (!fNPositionIds.empty())
         fInputTensorNames.emplace_back(fNPositionIds);
      fOutputTensorNames = {fNY};
   }

   void Initialize(RModel &model) override
   {
      for (const std::string &name : {fNX, fNCos, fNSin, fNPositionIds}) {
         if (!name.empty() && !model.CheckIfTensorAlreadyExist(name))
            throw std::runtime_error("SOFIE RotaryEmbedding: tensor " + name + " not found");
         if (!name.empty() && model.IsStridedInputTensor(name))
            throw std::runtime_error("SOFIE RotaryEmbedding: strided input is not supported");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      // the last dimension (head_size or hidden_size) must be known to split the heads and the rotated part
      if (fShapeX.size() == 4) {
         fIs3d = false;
         fBatch = fShapeX[0];
         fSeq = fShapeX[2];
         if (fShapeX[1].isParam || fShapeX[3].isParam)
            throw std::runtime_error("SOFIE RotaryEmbedding: the number of heads and the head size of " + fNX +
                                     " must be known, got shape " + ConvertDimShapeToString(fShapeX));
         fNumHeads = fShapeX[1].dim;
         fHeadSize = fShapeX[3].dim;
         if (fAttrNumHeads != 0 && fAttrNumHeads != fNumHeads)
            throw std::runtime_error("SOFIE RotaryEmbedding: num_heads attribute " + std::to_string(fAttrNumHeads) +
                                     " does not match the input shape " + ConvertDimShapeToString(fShapeX));
      } else if (fShapeX.size() == 3) {
         fIs3d = true;
         fBatch = fShapeX[0];
         fSeq = fShapeX[1];
         if (fAttrNumHeads == 0)
            throw std::runtime_error("SOFIE RotaryEmbedding: num_heads attribute is required for a 3d input");
         if (fShapeX[2].isParam || fShapeX[2].dim % fAttrNumHeads != 0)
            throw std::runtime_error("SOFIE RotaryEmbedding: hidden size of " + fNX + " with shape " +
                                     ConvertDimShapeToString(fShapeX) + " must be a known multiple of num_heads " +
                                     std::to_string(fAttrNumHeads));
         fNumHeads = fAttrNumHeads;
         fHeadSize = fShapeX[2].dim / fAttrNumHeads;
      } else {
         throw std::runtime_error("SOFIE RotaryEmbedding: input " + fNX + " must be 3d or 4d, got shape " +
                                  ConvertDimShapeToString(fShapeX));
      }

      fRotaryDim = (fAttrRotaryDim == 0) ? fHeadSize : fAttrRotaryDim;
      if (fRotaryDim > fHeadSize || fRotaryDim % 2 != 0)
         throw std::runtime_error("SOFIE RotaryEmbedding: rotary_embedding_dim " + std::to_string(fRotaryDim) +
                                  " must be even and not larger than the head size " + std::to_string(fHeadSize));

      // the caches are [batch, seq, R/2], or [max_position, R/2] when indexed with position_ids
      for (const std::string &name : {fNCos, fNSin}) {
         auto shape = model.GetDimTensorShape(name);
         const size_t expectedRank = fNPositionIds.empty() ? 3 : 2;
         if (shape.size() != expectedRank || shape.back().isParam || shape.back().dim != fRotaryDim / 2)
            throw std::runtime_error("SOFIE RotaryEmbedding: cache " + name + " has shape " +
                                     ConvertDimShapeToString(shape) + ", expected rank " +
                                     std::to_string(expectedRank) + " with last dimension rotary_embedding_dim / 2 = " +
                                     std::to_string(fRotaryDim / 2));
         if (fNPositionIds.empty()) {
            const bool batchOk = shape[0].isParam || fBatch.isParam || shape[0].dim == fBatch.dim;
            const bool seqOk = shape[1].isParam || fSeq.isParam || shape[1].dim == fSeq.dim;
            if (!batchOk || !seqOk)
               throw std::runtime_error("SOFIE RotaryEmbedding: cache " + name + " has shape " +
                                        ConvertDimShapeToString(shape) +
                                        " which does not match the batch size and sequence length of input " +
                                        ConvertDimShapeToString(fShapeX));
         }
      }
      if (!fNPositionIds.empty()) {
         if (model.GetTensorType(fNPositionIds) != ETensorType::INT64)
            throw std::runtime_error("SOFIE RotaryEmbedding: position_ids " + fNPositionIds + " must be int64");
         if (model.GetDimTensorShape(fNPositionIds).size() != 2)
            throw std::runtime_error("SOFIE RotaryEmbedding: position_ids " + fNPositionIds +
                                     " must have shape [batch, sequence_length]");
      }

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeX);
   }

   std::string Generate(std::string opName) override
   {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE RotaryEmbedding " + opName + " called to Generate without being initialized");
      const size_t half = fRotaryDim / 2;
      auto tensor = [](const std::string &n) { return "tensor_" + n; };

      std::stringstream out;
      out << "\n//---- RotaryEmbedding operator " << opName << "\n";
      out << "{\n";
      if (!fNPositionIds.empty())
         out << "const int64_t* POS = " << tensor(fNPositionIds) << ";\n";
      out << "for (std::size_t b = 0; b < " << fBatch.GetVal() << "; ++b) {\n";
      out << "for (std::size_t s = 0; s < " << fSeq.GetVal() << "; ++s) {\n";
      out << "const std::size_t cacheOffset = " << CacheOffset() << ";\n";
      out << "for (std::size_t h = 0; h < " << fNumHeads << "; ++h) {\n";
      out << "const std::size_t offset = " << HeadOffset() << ";\n";
      out << "for (std::size_t i = 0; i < " << half << "; ++i) {\n";
      out << "const " << fType << " c = " << tensor(fNCos) << "[cacheOffset + i];\n";
      out << "const " << fType << " sn = " << tensor(fNSin) << "[cacheOffset + i];\n";
      out << "const " << fType << " x1 = " << tensor(fNX) << "[offset + " << FirstOfPair("i") << "];\n";
      out << "const " << fType << " x2 = " << tensor(fNX) << "[offset + " << SecondOfPair("i") << "];\n";
      out << tensor(fNY) << "[offset + " << FirstOfPair("i") << "] = c * x1 - sn * x2;\n";
      out << tensor(fNY) << "[offset + " << SecondOfPair("i") << "] = sn * x1 + c * x2;\n";
      out << "}\n";
      if (fRotaryDim < fHeadSize) {
         out << "for (std::size_t i = " << fRotaryDim << "; i < " << fHeadSize << "; ++i)\n";
         out << tensor(fNY) << "[offset + i] = " << tensor(fNX) << "[offset + i];\n";
      }
      out << "}\n}\n}\n}\n";
      return out.str();
   }

   /* One thread per (batch, token, head, j) with j in [0, R/2 + D - R): j < R/2 rotates the pair j of the
      head, the others copy one element of the head beyond the rotated part. Every thread writes distinct
      outputs, so there is no shared memory and no synchronisation. */
   std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override
   {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE RotaryEmbedding GPU kernel called without initialization");
      const size_t half = fRotaryDim / 2;
      const size_t perHead = half + (fHeadSize - fRotaryDim);

      std::string op = "\n//------ ROTARY_EMBEDDING_KERNEL_ALPAKA\n";
      op += "struct RotaryEmbeddingKernel_" + opName + " {\n";
      op += "template<typename TAcc, typename T>\n";
      op += "ALPAKA_FN_ACC void operator()(\n";
      op += "TAcc const& acc,\n";
      op += "T const* __restrict__ X,\n";
      op += "T const* __restrict__ COS,\n";
      op += "T const* __restrict__ SIN,\n";
      if (!fNPositionIds.empty())
         op += "int64_t const* __restrict__ POS,\n";
      op += "T* __restrict__ Y";
      for (auto &p : dynParamNames)
         op += ",\nstd::size_t const " + p;
      op += ",\nstd::size_t const numWork) const {\n";
      op += "auto const t = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += "if (t >= numWork) return;\n";
      op += "std::size_t const j = t % " + std::to_string(perHead) + ";\n";
      op += "std::size_t const h = (t / " + std::to_string(perHead) + ") % " + std::to_string(fNumHeads) + ";\n";
      op += "std::size_t const token = t / " + std::to_string(perHead * fNumHeads) + ";\n";
      op += "std::size_t const s = token % static_cast<std::size_t>(" + fSeq.GetVal() + ");\n";
      op += "std::size_t const b = token / static_cast<std::size_t>(" + fSeq.GetVal() + ");\n";
      op += "std::size_t const offset = " + HeadOffset() + ";\n";
      op += "if (j < " + std::to_string(half) + ") {\n";
      op += "std::size_t const cacheOffset = " + CacheOffset() + ";\n";
      op += "T const c = COS[cacheOffset + j];\n";
      op += "T const sn = SIN[cacheOffset + j];\n";
      op += "T const x1 = X[offset + " + FirstOfPair("j") + "];\n";
      op += "T const x2 = X[offset + " + SecondOfPair("j") + "];\n";
      op += "Y[offset + " + FirstOfPair("j") + "] = c * x1 - sn * x2;\n";
      op += "Y[offset + " + SecondOfPair("j") + "] = sn * x1 + c * x2;\n";
      op += "} else {\n";
      op += "std::size_t const i = " + std::to_string(fRotaryDim) + " + (j - " + std::to_string(half) + ");\n";
      op += "Y[offset + i] = X[offset + i];\n";
      op += "}\n";
      op += "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      return "RotaryEmbeddingKernel_" + opName + " rotaryEmbeddingKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override
   {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE RotaryEmbedding GPU dispatch called without initialization");
      const size_t perHead = fRotaryDim / 2 + (fHeadSize - fRotaryDim);
      const std::string numWork = "(" + fBatch.GetVal() + ") * (" + fSeq.GetVal() + ") * " +
                                  std::to_string(fNumHeads * perHead);

      std::string args = "alpaka::getPtrNative(deviceBuf_" + fNX + "), alpaka::getPtrNative(deviceBuf_" + fNCos +
                         "), alpaka::getPtrNative(deviceBuf_" + fNSin + "), ";
      if (!fNPositionIds.empty())
         args += "alpaka::getPtrNative(deviceBuf_" + fNPositionIds + "), ";
      args += "alpaka::getPtrNative(deviceBuf_" + fNY + ")";
      for (auto &p : dynParamNames)
         args += ", static_cast<std::size_t>(" + p + ")";
      args += ", static_cast<Idx>(" + numWork + ")";

      std::stringstream out;
      out << "\n//------ ROTARY_EMBEDDING_GPU_ALPAKA\n";
      out << "auto const elementsPerGrid_" << opName << " = Vec::all(Idx{" << numWork << "});\n";
      out << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", rotaryEmbeddingKernel_" << opName << ", " << args << ");\n";
      out << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_ROTARYEMBEDDING
