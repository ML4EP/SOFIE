#ifndef SOFIE_ROPERATOR_TOPK
#define SOFIE_ROPERATOR_TOPK

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <algorithm>
#include <sstream>


namespace SOFIE {

template <typename T>
class ROperator_TopK final : public ROperator {

private:
   int fAttrAxis;
   int fAttrLargest;
   int fAttrSorted;

   Dim fTopKCount;      // k clamped to the axis dimension: a number for a static axis, a
                        // std::min(k, axis) expression for a dynamic one
   size_t fRequestedK;  // the model's requested k, unclamped; sizes the GPU register buffers
   bool fAxisIsDynamic = false;  // true if the axis dim (hence fTopKCount) is only known at runtime
   std::string fNK;
   std::string fNX;
   std::string fNVal;
   std::string fNInd;
   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;
   std::string fType;

   // GPU tuning knobs for the hierarchical kernel below. fStagingB is the size of each
   // thread's private register buffer. fBlockSize is the number of threads cooperating on
   // one slice.
   static constexpr size_t fStagingBCap = 64;
   size_t fStagingB = fStagingBCap;
   size_t fBlockSize = 256;

public:
   ROperator_TopK() {}
   ROperator_TopK(int attr_axis, int attr_largest, int attr_sorted, std::string nameK, std::string nameX, std::string nameVal, std::string nameInd)
      : fAttrAxis(attr_axis),
        fAttrLargest(attr_largest),
        fAttrSorted(attr_sorted),
        fNK(UTILITY::Clean_name(nameK)),
        fNX(UTILITY::Clean_name(nameX)),
        fNVal(UTILITY::Clean_name(nameVal)),
        fNInd(UTILITY::Clean_name(nameInd)){
            fInputTensorNames = { fNX, fNK };
            fOutputTensorNames = { fNVal, fNInd };
        }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      ETensorType ret = input[0];
      return {ret, ret};
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE TopK Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNK) == false) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE TopK Op Input Tensor i.e. K is not found in model");
      }

      fShapeX = model.GetDimTensorShape(fNX);
      auto kptr = static_cast<int64_t *>(model.GetInitializedTensorData(fNK).get());
      size_t kval = *kptr;
      fRequestedK = kval;
      model.SetNotWritableInitializedTensor(fNK);
      fAttrAxis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      if (static_cast<size_t>(fAttrAxis) >= fShapeX.size()) {
         throw std::runtime_error("SOFIE TopK op axis = " + std::to_string(fAttrAxis) +
            " value exceeds size of tensor " + fNX + " of size " + std::to_string(fShapeX.size()) + " .");
      }
      // fTopKCount cannot be larger than the axis dimension
      fAxisIsDynamic = fShapeX[fAttrAxis].isParam;
      if (fAxisIsDynamic) {
         fTopKCount = Dim{std::string("std::min(size_t(" + std::to_string(kval) + "), " + fShapeX[fAttrAxis].GetVal() + ")" ), static_cast<size_t>(-1) };
         // axis size unknown at codegen time - rsV/rsI must be a fixed-size array, so
         // fall back to the empirically-safe cap of 64.
         fStagingB = fStagingBCap;
      } else {
         fTopKCount = Dim { std::min(kval, fShapeX[fAttrAxis].dim) };
         // a thread never buffers more than its strided share of the axis, so B need
         // never exceed that share.
         size_t itemsPerThread = (fShapeX[fAttrAxis].dim + fBlockSize - 1) / fBlockSize;
         fStagingB = std::min(std::max<size_t>(itemsPerThread, 1), fStagingBCap);
      }

      // output shape is equal to input shape apart for value in fAttrAxis
      fShapeY = fShapeX;
      fShapeY[fAttrAxis] = Dim{fTopKCount};

      model.AddIntermediateTensor(fNVal, model.GetTensorType(fNX), fShapeY);

      // output indices should be an int64 tensor
      model.AddIntermediateTensor(fNInd, ETensorType::INT64, fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNX));

      if (model.Verbose()) {
         std::cout << "TopK " << fNX << "  " << ConvertDimShapeToString(fShapeX)
                   << "---> " << fNVal << " " << ConvertDimShapeToString(fShapeY) << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n" << SP << "//------ TopK\n";

      // we perform loop on dimension before sorted axis and after sorted axis
      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      auto sx = UTILITY::ComputeSliceInfo(fShapeX, axis), sy = UTILITY::ComputeSliceInfo(fShapeY, axis);   //X has n elements along the axis, Y has k: two layouts

      out << SP << "{\n"; // to define a separate scope for the operator code
      out << SP << "std::vector<std::pair<float,int64_t>> elements(" << sx.nElements << ");\n";
      // loop on elements before
      if (sx.nBefore != "1") {
         out << SP << "for (size_t i = 0; i < " << sx.nBefore << "; i++) {\n";
         out << SP << SP << "size_t xoffset = i*" << sx.strideBefore << ";\n";
         out << SP << SP << "size_t yoffset = i*" << sy.strideBefore << ";\n";
         out << SP;
      } else {
         out << SP << "size_t xoffset = 0;\n";
         out << SP << "size_t yoffset = 0;\n";
      }
      if (sx.nAfter != "1")
         out << SP << "for (size_t j = 0; j < " << sx.nAfter << "; j++) {\n";
      else
         out << SP << "const size_t j = 0;\n";

      // copy elements to be sorted in vector of pair
      out << SP << SP << "for (size_t l = 0; l < " << sx.nElements << "; l++) {\n";
      out << SP << SP << SP << "elements[l] = std::make_pair(tensor_" << fNX << "[xoffset + " << sx.strideAxis << "*l + j], l);\n";
      out << SP << SP << "}\n";

      if (fAttrSorted) {
         if (fAttrLargest)
            out << SP << SP << "std::partial_sort(elements.begin(),elements.begin()+" << fTopKCount << ",elements.end(),"
                << "[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first>b.first) : a.second < b.second;});\n";
         else
            out << SP << SP << "std::partial_sort(elements.begin(),elements.begin()+" << fTopKCount << ",elements.end(),"
                << "[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first<b.first) : a.second < b.second;});\n";
      } else
         // in this case we don't need to return sorted elements, so we keep same order as before
         out << SP << SP << "std::partial_sort(elements.begin(),elements.begin()+" << fTopKCount << ",elements.end());\n";

      // copy the selected elements in the output
      out << SP << SP << "for (size_t l = 0; l < " << fTopKCount << "; l++) {\n";
      out << SP << SP << SP << "tensor_" << fNVal << "[yoffset + " << sy.strideAxis << "*l + j] = elements[l].first;\n";
      out << SP << SP << SP << "tensor_" << fNInd << "[yoffset + " << sy.strideAxis << "*l + j] = elements[l].second;\n";
      out << SP << SP << "}\n";
      if (sx.nAfter != "1") out << SP << SP << "}\n";
      if (sx.nBefore != "1") out << SP << "}\n";
      out << SP << "}\n"; // end operator scope
      return out.str();
   }

   // Hierarchical GPU path: one block per slice, fBlockSize threads cooperating on that
   // slice's whole axis via a block-shared result (sVals/sInds, capacity K, K = the
   // model's requested k unclamped - the only value that needs to be a compile-time
   // constant here) guarded by a spinlock, plus a fill counter (sCount). Each thread
   // scans a strided subset of the axis into a tiny private register buffer (rsV/rsI,
   // size fStagingB - independent of K). This mirrors the shape of
   // real GPU top-k selection structures, simplified to use a
   // block-shared array + lock for the merge instead of warp-shuffle-only cooperation: 
   // a shared-memory merge is both simpler and the more portable
   // choice here. The spinlock is safe specifically because this is a *single-block*
   // design, unlike the cross-block case.

   // Admission before touching the shared state: a candidate only gets buffered locally
   // once it beats the current shared threshold (sVals[topKCount-1], once sCount reaches
   // topKCount) - anything worse is dropped immediately, for free, without acquiring the
   // lock. That read is deliberately lock-free: sCount and the shared array only ever get 
   // strictly tighter/more-populated over the scan, so an out-of-date value can make a 
   // thread buffer something that turns out unnecessary. It still has to be read
   // through volatile-qualified pointers.
   //
   // A thread's local buffer is flushed into the shared result whenever it fills, and
   // once more after the thread's own portion of the axis is exhausted -
   // required for correctness in general (a short axis, or many threads, means a
   // thread's last few buffered candidates may never trigger a fill-driven flush).
   //
   // fStagingB and fBlockSize are compile-time literals with no closed-form optimum
   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");

      std::string CMP = fAttrLargest ? ">" : "<";
      std::string K = std::to_string(fRequestedK);
      std::string B = std::to_string(fStagingB);
      std::string P = std::to_string(fBlockSize);
      std::string better = fAttrLargest
         ? "(v > sVals[p-1] || (v == sVals[p-1] && idx < sInds[p-1]))"
         : "(v < sVals[p-1] || (v == sVals[p-1] && idx < sInds[p-1]))";
      std::string beatsWorst = fAttrLargest
         ? "(v > sVals[topKCount-1] || (v == sVals[topKCount-1] && idx < sInds[topKCount-1]))"
         : "(v < sVals[topKCount-1] || (v == sVals[topKCount-1] && idx < sInds[topKCount-1]))";
      // Admission-check variant for the one read site that happens outside the lock
      std::string beatsWorstVolatile = fAttrLargest
         ? "(v >= vSVals[topKCount-1])"
         : "(v <= vSVals[topKCount-1])";
      std::string kname = "TopKKernel_" + fNVal;

      auto emitLockAcquire = [&](const std::string &ind) {
         std::string s;
         s += ind + "{\n";
         s += ind + SP + "unsigned int ns = 8u;\n";
         s += ind + SP + "while (alpaka::atomicCas(acc, &sLock, 0, 1, alpaka::hierarchy::Grids{}) != 0) {\n";
         s += ind + SP + SP + "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700\n";
         s += ind + SP + SP + "__nanosleep(ns);\n";
         s += ind + SP + SP + "if (ns < 256u) ns <<= 1;\n";
         s += ind + SP + SP + "#elif defined(__HIP_DEVICE_COMPILE__)\n";
         s += ind + SP + SP + "__builtin_amdgcn_s_sleep(2);\n";
         s += ind + SP + SP + "(void)ns;\n";
         s += ind + SP + SP + "#else\n";
         s += ind + SP + SP + "(void)ns;\n";
         s += ind + SP + SP + "#endif\n";
         s += ind + SP + "}\n";
         s += ind + "}\n";
         return s;
      };

      auto emitInsert = [&](const std::string &ind) {
         std::string s;
         s += ind + "if ((std::size_t)sCount < topKCount) {\n";
         s += ind + SP + "int p = sCount;\n";
         s += ind + SP + "while (p > 0 && " + better + ") { sVals[p] = sVals[p-1]; sInds[p] = sInds[p-1]; --p; }\n";
         s += ind + SP + "sVals[p] = v; sInds[p] = idx; sCount++;\n";
         s += ind + "} else if (" + beatsWorst + ") {\n";
         s += ind + SP + "int p = (int)topKCount - 1;\n";
         s += ind + SP + "while (p > 0 && " + better + ") { sVals[p] = sVals[p-1]; sInds[p] = sInds[p-1]; --p; }\n";
         s += ind + SP + "sVals[p] = v; sInds[p] = idx;\n";
         s += ind + "} else break;\n";
         return s;
      };

      std::string op = "\n//------ TopK_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ x,\n";
      op += SP + SP + SP + "T* __restrict__ vals,\n";
      op += SP + SP + SP + "int64_t* __restrict__ inds,\n";
      op += SP + SP + SP + "std::size_t const numSlices,\n";
      op += SP + SP + SP + "std::size_t const nAfter,\n";
      op += SP + SP + SP + "std::size_t const nElAxis,\n";
      op += SP + SP + SP + "std::size_t const topKCount,\n";
      op += SP + SP + SP + "std::size_t const strideXAxis,\n";
      op += SP + SP + SP + "std::size_t const strideXBefore,\n";
      op += SP + SP + SP + "std::size_t const strideYAxis,\n";
      op += SP + SP + SP + "std::size_t const strideYBefore) const {\n\n";

      op += SP + SP + SP + "auto const slice = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
      op += SP + SP + SP + "auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (slice >= numSlices) return;\n\n";

      op += SP + SP + SP + "std::size_t const i = slice / nAfter;\n";
      op += SP + SP + SP + "std::size_t const j = slice % nAfter;\n";
      op += SP + SP + SP + "std::size_t const xbase = i * strideXBefore + j;\n";
      op += SP + SP + SP + "std::size_t const ybase = i * strideYBefore + j;\n\n";

      op += SP + SP + SP + "auto& sVals  = alpaka::declareSharedVar<T[" + K + "], __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto& sInds  = alpaka::declareSharedVar<int64_t[" + K + "], __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto& sCount = alpaka::declareSharedVar<int, __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto& sLock  = alpaka::declareSharedVar<int, __COUNTER__>(acc);\n\n";

      op += SP + SP + SP + "if (tid == 0) { sCount = 0; sLock = 0; }\n";
      op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n\n";

      op += SP + SP + SP + "T const volatile* vSVals = sVals;\n";
      op += SP + SP + SP + "int const volatile* vSCount = &sCount;\n\n";

      op += SP + SP + SP + "T rsV[" + B + "]; int64_t rsI[" + B + "]; int rsN = 0;\n\n";

      op += SP + SP + SP + "for (std::size_t idx = tid; idx < nElAxis; idx += " + P + "u) {\n";
      op += SP + SP + SP + SP + "T v = x[xbase + strideXAxis * idx];\n";
      op += SP + SP + SP + SP + "bool admit;\n";
      op += SP + SP + SP + SP + "if ((std::size_t)*vSCount < topKCount) admit = true;\n";
      op += SP + SP + SP + SP + "else admit = " + beatsWorstVolatile + ";\n";
      op += SP + SP + SP + SP + "if (admit) {\n";
      op += SP + SP + SP + SP + SP + "int p = rsN;\n";
      op += SP + SP + SP + SP + SP + "while (p > 0 && v " + CMP + " rsV[p-1]) { rsV[p] = rsV[p-1]; rsI[p] = rsI[p-1]; --p; }\n";
      op += SP + SP + SP + SP + SP + "rsV[p] = v; rsI[p] = (int64_t)idx; rsN++;\n";
      op += SP + SP + SP + SP + SP + "if (rsN == " + B + ") {\n";
      op += emitLockAcquire(SP + SP + SP + SP + SP + SP);
      op += SP + SP + SP + SP + SP + SP + "alpaka::mem_fence(acc, alpaka::memory_scope::Device{});\n";
      op += SP + SP + SP + SP + SP + SP + "for (int r = 0; r < " + B + "; ++r) {\n";
      op += SP + SP + SP + SP + SP + SP + SP + "T v = rsV[r]; int64_t idx = rsI[r];\n";
      op += emitInsert(SP + SP + SP + SP + SP + SP + SP);
      op += SP + SP + SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + SP + SP + SP + "alpaka::mem_fence(acc, alpaka::memory_scope::Device{});\n";
      op += SP + SP + SP + SP + SP + SP + "alpaka::atomicExch(acc, &sLock, 0, alpaka::hierarchy::Grids{});\n";
      op += SP + SP + SP + SP + SP + SP + "rsN = 0;\n";
      op += SP + SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + "}\n\n";

      op += SP + SP + SP + "// drain: flush whatever's left, required for correctness in general\n";
      op += SP + SP + SP + "// (e.g. a short axis may never trigger a fill-driven flush above)\n";
      op += SP + SP + SP + "if (rsN > 0) {\n";
      op += emitLockAcquire(SP + SP + SP + SP);
      op += SP + SP + SP + SP + "alpaka::mem_fence(acc, alpaka::memory_scope::Device{});\n";
      op += SP + SP + SP + SP + "for (int r = 0; r < rsN; ++r) {\n";
      op += SP + SP + SP + SP + SP + "T v = rsV[r]; int64_t idx = rsI[r];\n";
      op += emitInsert(SP + SP + SP + SP + SP + SP);
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + SP + "alpaka::mem_fence(acc, alpaka::memory_scope::Device{});\n";
      op += SP + SP + SP + SP + "alpaka::atomicExch(acc, &sLock, 0, alpaka::hierarchy::Grids{});\n";
      op += SP + SP + SP + "}\n\n";

      op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n\n";
      op += SP + SP + SP + "for (std::size_t s = tid; s < topKCount; s += " + P + "u) {\n";
      op += SP + SP + SP + SP + "vals[ybase + strideYAxis * s] = sVals[s];\n";
      op += SP + SP + SP + SP + "inds[ybase + strideYAxis * s] = sInds[s];\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "TopKKernel_" + fNVal + " topKernel_" + fNVal + ";\n";
   }

   std::string GenerateInitCode_GPU_ALPAKA() override {
      if (!fAxisIsDynamic)
         return "";
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");

      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      auto sx = UTILITY::ComputeSliceInfo(fShapeX, axis);
      std::string maxLen = "((" + sx.nBefore + ") * (" + sx.nAfter + ") * " + std::to_string(fRequestedK) + "u)";

      std::string out;
      out += SP + "deviceBuf_" + fNVal + " = alpaka::allocBuf<" + fType + ", Idx>(devAcc, Ext1D::all(Idx{" + maxLen + "}));\n";
      out += SP + "deviceBuf_" + fNInd + " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{" + maxLen + "}));\n";
      return out;
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");

      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      auto sx = UTILITY::ComputeSliceInfo(fShapeX, axis), sy = UTILITY::ComputeSliceInfo(fShapeY, axis);   //X has n elements along the axis, Y has k: two layouts
      std::string numSlices = "((" + sx.nBefore + ")*(" + sx.nAfter + "))";
      std::string P = std::to_string(fBlockSize);
      std::string n = fNVal;

      std::stringstream out;
      out << "\n//-- TopK_GPU_ALPAKA (hierarchical)\n";
      out << SP << "{\n";
      out << SP << SP << "std::size_t const topkNumSlices_" << n << " = static_cast<std::size_t>(" << numSlices << ");\n";
      out << SP << SP << "std::size_t const topkNElAxis_"   << n << " = static_cast<std::size_t>(" << sx.nElements << ");\n";
      out << SP << SP << "std::size_t const topkTopKCount_" << n << " = static_cast<std::size_t>(" << fTopKCount.GetVal() << ");\n\n";

      out << SP << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << n << "(\n";
      out << SP << SP << SP << "Vec::all(Idx{topkNumSlices_" << n << "}),\n";
      out << SP << SP << SP << "Vec::all(Idx{" << P << "u}),\n";
      out << SP << SP << SP << "Vec::all(Idx{1u}));\n";
      out << SP << SP << "alpaka::exec<Acc>(queue, workDiv_" << n << ", topKernel_" << n
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNVal << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNInd << ")"
          << ", topkNumSlices_" << n
          << ", static_cast<std::size_t>(" << sx.nAfter << ")"
          << ", topkNElAxis_" << n
          << ", topkTopKCount_" << n
          << ", static_cast<std::size_t>(" << sx.strideAxis << ")"
          << ", static_cast<std::size_t>(" << sx.strideBefore << ")"
          << ", static_cast<std::size_t>(" << sy.strideAxis << ")"
          << ", static_cast<std::size_t>(" << sy.strideBefore << "));\n";
      out << SP << "}\n";
      return out.str();
   }

};

} // namespace SOFIE


#endif // SOFIE_ROPERATOR_TOPK
