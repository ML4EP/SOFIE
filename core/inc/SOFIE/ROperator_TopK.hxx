#ifndef SOFIE_ROPERATOR_TOPK
#define SOFIE_ROPERATOR_TOPK

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
namespace SOFIE {

template <typename T>
class ROperator_TopK final : public ROperator {

private:
   int fAttrAxis;
   int fAttrLargest;
   int fAttrSorted;

   Dim fK;
   size_t fRequestedK = 0;
   bool fKIsParam = false;
   bool fAxisIsDynamic = false;
   std::string fNK;
   std::string fNX;
   std::string fNVal;
   std::string fNInd;
   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;
   std::string fType;

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
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShapeX.empty();
      if (model.IsStridedInputTensor(fNK))
         throw std::runtime_error("SOFIE TopK - strided input is only supported for the data tensor X");
      Dim kdim;
      if (model.IsShapeTensor(fNK)) {
         auto &kvalues = model.GetShapeTensorValues(fNK);
         if (kvalues.size() != 1)
            throw std::runtime_error("SOFIE TopK Op input tensor K = " + fNK + " must be a single value");
         kdim = kvalues[0];
      } else if (model.IsInitializedTensor(fNK)) {
         auto kptr = static_cast<int64_t *>(model.GetInitializedTensorData(fNK).get());
         kdim = Dim{static_cast<size_t>(*kptr)};
         model.SetNotWritableInitializedTensor(fNK);
      } else {
         throw std::runtime_error("SOFIE TopK Op input tensor K = " + fNK +
                                  " must be known at initialization time");
      }
      fKIsParam = kdim.isParam;
      fRequestedK = kdim.isParam ? 0 : kdim.dim;
      fAttrAxis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      if(static_cast<size_t>(fAttrAxis) >=  fShapeX.size()){
         throw
            std::runtime_error("SOFIE ONNX TopK op axis = "+ std::to_string(fAttrAxis) +" value exeeds size of tensor " +fNX+" of size "+ std::to_string(fShapeX.size()) +" .");
      }
      if (kdim.isParam || fShapeX[fAttrAxis].isParam)
         fK = Dim{std::string("std::min(size_t(" + kdim.GetVal() + "), size_t(" + fShapeX[fAttrAxis].GetVal() + "))"),
                  static_cast<size_t>(-1)};
      else
         fK = Dim{std::min(kdim.dim, fShapeX[fAttrAxis].dim)};

      fAxisIsDynamic = fShapeX[fAttrAxis].isParam;
      if (!fAxisIsDynamic) {
         size_t itemsPerThread = (fShapeX[fAttrAxis].dim + fBlockSize - 1) / fBlockSize;
         fStagingB = std::min(std::max<size_t>(itemsPerThread, 1), fStagingBCap);
      }

      // output shape is equal to input shape apart for value in fAttrAxis
      fShapeY = fShapeX;
      fShapeY[fAttrAxis] = Dim{fK};

      model.AddIntermediateTensor(fNVal, model.GetTensorType(fNX), fShapeY);

      // output indices should be an int64 tensor
      model.AddIntermediateTensor(fNInd, ETensorType::INT64, fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      model.AddNeededStdLib("algorithm");
      model.AddNeededStdLib("cstdint");
      model.AddNeededStdLib("cstring");

      if (model.Verbose()) {
         std::cout << "TopK " << fNX << "  " << ConvertDimShapeToString(fShapeX)
                      << "---> " << fNVal << " " <<  ConvertDimShapeToString(fShapeY) << std::endl;
      }
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t size = fShapeX.size();
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;
      out << "\n" << SP << "//------ TopK\n";

      auto length=ConvertDimShapeToLength(fShapeX);
      auto strideX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto strideY = UTILITY::ComputeStrideFromShape(fShapeY);
      // we perform loop on dimension before sorted axis and after sorted axis
      std::vector<Dim> shape_before(fShapeX.begin(), fShapeX.begin() + axis);
      std::string n_before = (axis>0) ? ConvertDimShapeToLength(shape_before) : "1";
      std::string n_after = strideX[axis].GetVal();
      std::string n_elements = fShapeX[axis].GetVal();

      out << SP << "{\n"; // to define a separate scope for the operator code
      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      const std::string xIndex = "xoffset + " + strideX[axis].GetVal() + "*l + j";
      const std::string xRead = fHasStridedInput ? "tensor_" + fNX + "[xoff_" + OpName + "(" + xIndex + ")]"
                                                 : "tensor_" + fNX + "[" + xIndex + "]";
      if (fHasStridedInput)
         out << GenerateStridedOffsetLambda(OpName, fNX, fShapeX);

      //
      bool packed = (fType == "float");
      if (packed && !fShapeX[fAttrAxis].isParam && fShapeX[fAttrAxis].dim > 0xFFFFFFFFULL)
         packed = false;

      std::string pairType = "std::pair<" + fType + ",int64_t>";
      if (packed) {
         out << SP << "std::vector<uint64_t> elements(" << n_elements << ");\n";
         if (fShapeX[fAttrAxis].isParam) {
            out << SP << "if (static_cast<unsigned long long>(" << n_elements << ") > 0xFFFFFFFFULL)\n";
            out << SP << SP << "throw std::runtime_error(\"SOFIE TopK - reduced axis is longer "
                << "than the 2^32 limit of the packed index\");\n";
         }
      } else {
         out << SP << "std::vector<" << pairType << "> elements(" << n_elements << ");\n";
         out << SP << "auto " << OpName << "_cmp = [](const " << pairType << " &a, const " << pairType << " &b) {\n";
         out << SP << SP << "return (a.first != b.first) ? (a.first " << (fAttrLargest ? ">" : "<")
             << " b.first) : a.second < b.second;\n";
         out << SP << "};\n";
      }
      // loop on elements before
      if (n_before != "1") {
         out << SP << "for (size_t i = 0; i < " << n_before << "; i++) {\n";
         out << SP << SP << "size_t xoffset = i*" << strideX[axis-1] << ";\n";
         out << SP << SP << "size_t yoffset = i*" << strideY[axis-1] << ";\n";
         out << SP;
      } else {
         out << SP << "size_t xoffset = 0;\n";
         out << SP << "size_t yoffset = 0;\n";
      }
      if (n_after !=  "1")
         out << SP << "for (size_t j = 0; j < " << n_after << "; j++) {\n";
      else
         out << SP << "const size_t j = 0;\n";

      out << SP << SP << "for (size_t l = 0; l < " << n_elements << "; l++) {\n";
      if (packed) {
         out << SP << SP << SP << "uint32_t b_ = 0;\n";
         out << SP << SP << SP << "std::memcpy(&b_, &" << xRead << ", sizeof(b_));\n";
         out << SP << SP << SP << "b_ ^= (b_ & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;\n";
         if (fAttrLargest)
            out << SP << SP << SP << "b_ = ~b_;\n";
         out << SP << SP << SP << "elements[l] = (static_cast<uint64_t>(b_) << 32) | static_cast<uint32_t>(l);\n";
      } else {
         out << SP << SP << SP << "elements[l] = std::make_pair(" << xRead << ", l);\n";
      }
      out << SP << SP << "}\n";

      std::string cmp = packed ? "" : (", " + OpName + "_cmp");
      out << SP << SP << "std::nth_element(elements.begin(), elements.begin() + (" << fK << "), elements.end()" << cmp
          << ");\n";
      out << SP << SP << "std::sort(elements.begin(), elements.begin() + (" << fK << ")" << cmp << ");\n";

      // copy the selected elements in the output
      out << SP << SP << "for (size_t l = 0; l < " << fK << "; l++) {\n";
      if (packed) {
         out << SP << SP << SP << "uint32_t b_ = static_cast<uint32_t>(elements[l] >> 32);\n";
         if (fAttrLargest)
            out << SP << SP << SP << "b_ = ~b_;\n";
         out << SP << SP << SP << "b_ ^= (b_ & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;\n";
         out << SP << SP << SP << fType << " v_;\n";
         out << SP << SP << SP << "std::memcpy(&v_, &b_, sizeof(v_));\n";
         out << SP << SP << SP << "tensor_" << fNVal << "[yoffset + " << strideY[axis] << "*l + j] = v_;\n";
         out << SP << SP << SP << "tensor_" << fNInd << "[yoffset + " << strideY[axis]
             << "*l + j] = static_cast<int64_t>(static_cast<uint32_t>(elements[l]));\n";
      } else {
         out << SP << SP << SP << "tensor_" << fNVal << "[yoffset + " << strideY[axis]
             << "*l + j] = elements[l].first;\n";
         out << SP << SP << SP << "tensor_" << fNInd << "[yoffset + " << strideY[axis]
             << "*l + j] = elements[l].second;\n";
      }
      out << SP << SP << "}\n";
      if (n_after != "1") out << SP << SP << "}\n";
      if (n_before != "1") out << SP << "}\n";
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
      if (fKIsParam)
         throw std::runtime_error("SOFIE Operator TopK GPU code generation requires K to be known at initialization");

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
      op += SP + SP + SP + "std::size_t const strideYBefore";
      if (fHasStridedInput)
         op += ",\n" + SP + SP + SP + "sofie_strided_layout<" + std::to_string(fShapeX.size()) + "> const layoutX";
      op += ") const {\n\n";

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
      op += SP + SP + SP + SP + "T v = " + (fHasStridedInput ? StridedKernelRead("x", "layoutX", "xbase + strideXAxis * idx")
                                                          : std::string("x[xbase + strideXAxis * idx]")) + ";\n";
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
      if (!IsOutputPooled(fNVal))
         out += SP + "deviceBuf_" + fNVal + " = alpaka::allocBuf<" + fType + ", Idx>(devAcc, Ext1D::all(Idx{" + maxLen + "}));\n";
      if (!IsOutputPooled(fNInd))
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
      if (fHasStridedInput)
         out << GenerateStridedBroadcastLayout(opName + "_X", fNX, fShapeX, fShapeX.size(), fShapeX);
      out << SP << "{\n";
      out << SP << SP << "std::size_t const topkNumSlices_" << n << " = static_cast<std::size_t>(" << numSlices << ");\n";
      out << SP << SP << "std::size_t const topkNElAxis_"   << n << " = static_cast<std::size_t>(" << sx.nElements << ");\n";
      out << SP << SP << "std::size_t const topkTopKCount_" << n << " = static_cast<std::size_t>(" << fK.GetVal() << ");\n\n";

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
          << ", static_cast<std::size_t>(" << sy.strideBefore << ")"
          << (fHasStridedInput ? ", layout_" + opName + "_X" : std::string()) << ");\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_TOPK
