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

   /* GPU path: one block per slice, fBlockSize threads cooperating on that slice's
      whole axis. Every thread keeps the best K of the elements it scans in a sorted
      register list (K is the model's requested k, the one value that has to be a
      compile-time constant here). The lists are then combined without any shared
      writes during the scan: the lanes of a warp merge pairwise through shuffles
      until lane 0 holds the warp's best K, each warp parks that list in shared
      memory, and after one block barrier the first warp merges the per-warp lists
      the same way and writes the result.

      No thread ever waits for another thread's progress, only for the block
      barrier, so the kernel runs the same on hardware that schedules the lanes of a
      warp independently and on lockstep wavefronts. The result is deterministic:
      ties are broken on the lower index at every merge, matching the CPU path.

      Each merge of two sorted K-lists is a K-step two-pointer pass done by one
      lane. For K in the hundreds a bitonic network across lanes would be faster. */
   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");
      if (fKIsParam)
         throw std::runtime_error("SOFIE Operator TopK GPU code generation requires K to be known at initialization");

      std::string CMP = fAttrLargest ? ">" : "<";
      std::string K = std::to_string(fRequestedK);
      std::string P = std::to_string(fBlockSize);
      // unused list slots hold the value that loses every comparison
      std::string worst = fAttrLargest ? "std::numeric_limits<T>::lowest()" : "std::numeric_limits<T>::max()";
      std::string kname = "TopKKernel_" + fNVal;

      // "a beats b": the better value, or the lower index on a tie
      auto beats = [&](const std::string &aV, const std::string &aI, const std::string &bV, const std::string &bI) {
         return "(" + aV + " " + CMP + " " + bV + " || (" + aV + " == " + bV + " && " + aI + " < " + bI + "))";
      };

      /* Shuffle rounds over one warp: each round the lower half of the active lanes
         fetches its partner's list and merges it into its own, so after log2(warpSize)
         rounds lane 0 holds the best K of all the lanes' lists. */
      auto emitWarpMerge = [&](const std::string &ind) {
         std::string s;
         s += ind + "for (std::int32_t offset = warpSize / 2; offset > 0; offset >>= 1) {\n";
         s += ind + SP + "T oV[" + K + "]; int64_t oI[" + K + "];\n";
         s += ind + SP + "for (int p = 0; p < " + K + "; ++p) {\n";
         s += ind + SP + SP + "oV[p] = alpaka::warp::shfl_down(acc, myV[p], static_cast<std::uint32_t>(offset));\n";
         s += ind + SP + SP + "oI[p] = alpaka::warp::shfl_down(acc, myI[p], static_cast<std::uint32_t>(offset));\n";
         s += ind + SP + "}\n";
         s += ind + SP + "mergeBestK_" + fNVal + "(myV, myI, oV, oI);\n";
         s += ind + "}\n";
         return s;
      };

      std::string op = "\n//------ TopK_KERNEL_ALPAKA\n";

      /* Merge two sorted K-lists (best first) into the best K of their union, written
         back into the first. A K-step two-pointer pass; an exhausted side loses. */
      op += SP + "template<typename T>\n";
      op += SP + "ALPAKA_FN_ACC inline void mergeBestK_" + fNVal + "(T* aV, int64_t* aI, T const* bV, int64_t const* bI) {\n";
      op += SP + SP + "T rV[" + K + "]; int64_t rI[" + K + "]; int i = 0, j = 0;\n";
      op += SP + SP + "for (int p = 0; p < " + K + "; ++p) {\n";
      op += SP + SP + SP + "bool const takeA = (j == " + K + ") || (i < " + K + " && " + beats("aV[i]", "aI[i]", "bV[j]", "bI[j]") + ");\n";
      op += SP + SP + SP + "rV[p] = takeA ? aV[i] : bV[j]; rI[p] = takeA ? aI[i] : bI[j];\n";
      op += SP + SP + SP + "i += takeA; j += !takeA;\n";
      op += SP + SP + "}\n";
      op += SP + SP + "for (int p = 0; p < " + K + "; ++p) { aV[p] = rV[p]; aI[p] = rI[p]; }\n";
      op += SP + "}\n\n";

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
      op += SP + SP + SP + "std::int32_t const warpSize = alpaka::warp::getSize(acc);\n";
      op += SP + SP + SP + "std::int32_t const laneId = static_cast<std::int32_t>(tid) % warpSize;\n";
      op += SP + SP + SP + "if (slice >= numSlices) return;\n\n";

      op += SP + SP + SP + "std::size_t const i = slice / nAfter;\n";
      op += SP + SP + SP + "std::size_t const j = slice % nAfter;\n";
      op += SP + SP + SP + "std::size_t const xbase = i * strideXBefore + j;\n";
      op += SP + SP + SP + "std::size_t const ybase = i * strideYBefore + j;\n\n";

      op += SP + SP + SP + "std::int32_t const warpId = static_cast<std::int32_t>(tid) / warpSize;\n";
      op += SP + SP + SP + "std::int32_t const nWarps = " + P + " / warpSize;\n\n";

      // one K-list per warp, sized for 32-wide warps; 64-wide wavefronts use the first half
      op += SP + SP + SP + "auto& wVals = alpaka::declareSharedVar<T[(" + P + " / 32) * " + K + "], __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto& wInds = alpaka::declareSharedVar<int64_t[(" + P + " / 32) * " + K + "], __COUNTER__>(acc);\n\n";

      // each thread keeps the best K of the elements it scans, sorted best first
      op += SP + SP + SP + "T myV[" + K + "]; int64_t myI[" + K + "];\n";
      op += SP + SP + SP + "for (int p = 0; p < " + K + "; ++p) { myV[p] = " + worst + "; myI[p] = -1; }\n";
      op += SP + SP + SP + "for (std::size_t idx = tid; idx < nElAxis; idx += " + P + "u) {\n";
      op += SP + SP + SP + SP + "T const v = " + (fHasStridedInput ? StridedKernelRead("x", "layoutX", "xbase + strideXAxis * idx")
                                                                : std::string("x[xbase + strideXAxis * idx]")) + ";\n";
      op += SP + SP + SP + SP + "int64_t const vi = static_cast<int64_t>(idx);\n";
      op += SP + SP + SP + SP + "if (" + beats("v", "vi", "myV[" + K + "-1]", "myI[" + K + "-1]") + ") {\n";
      op += SP + SP + SP + SP + SP + "int p = " + K + " - 1;\n";
      op += SP + SP + SP + SP + SP + "while (p > 0 && " + beats("v", "vi", "myV[p-1]", "myI[p-1]") + ") { myV[p] = myV[p-1]; myI[p] = myI[p-1]; --p; }\n";
      op += SP + SP + SP + SP + SP + "myV[p] = v; myI[p] = vi;\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + "}\n\n";

      // merge across the warp, then park each warp's list in shared memory
      op += emitWarpMerge(SP + SP + SP);
      op += SP + SP + SP + "if (laneId == 0) {\n";
      op += SP + SP + SP + SP + "for (int p = 0; p < " + K + "; ++p) { wVals[warpId * " + K + " + p] = myV[p]; wInds[warpId * " + K + " + p] = myI[p]; }\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n\n";

      // the first warp merges the per-warp lists the same way and writes the result
      op += SP + SP + SP + "if (warpId == 0) {\n";
      op += SP + SP + SP + SP + "for (int p = 0; p < " + K + "; ++p) {\n";
      op += SP + SP + SP + SP + SP + "myV[p] = laneId < nWarps ? wVals[laneId * " + K + " + p] : " + worst + ";\n";
      op += SP + SP + SP + SP + SP + "myI[p] = laneId < nWarps ? wInds[laneId * " + K + " + p] : -1;\n";
      op += SP + SP + SP + SP + "}\n";
      op += emitWarpMerge(SP + SP + SP + SP);
      op += SP + SP + SP + SP + "if (laneId == 0) {\n";
      op += SP + SP + SP + SP + SP + "for (std::size_t s = 0; s < topKCount; ++s) {\n";
      op += SP + SP + SP + SP + SP + SP + "vals[ybase + strideYAxis * s] = myV[s];\n";
      op += SP + SP + SP + SP + SP + SP + "inds[ybase + strideYAxis * s] = myI[s];\n";
      op += SP + SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + SP + "}\n";
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
