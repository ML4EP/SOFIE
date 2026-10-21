#ifndef SOFIE_ROPERATOR_Reduce
#define SOFIE_ROPERATOR_Reduce

#include "SOFIE/utils/SOFIE_common.hxx"
#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/RModel.hxx"

#include <memory>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <limits>

namespace SOFIE{

enum EReduceOpMode { ReduceMean, ReduceSum, ReduceSumSquare, ReduceProd, ReduceL2, ReduceMax, ReduceMin, InvalidReduceOp };

template <EReduceOpMode Op>
class ROperator_Reduce final : public ROperator_ALPAKA
{
private:
    /* Attributes*/
    bool fInputDimShape = false;
    int fkeepdims = 1; //default value
    std::vector<int64_t> fAttrAxes;
    EReduceOpMode fReduceOpMode;
    std::string fNX;
    std::string fNAxes;
    std::string fNY;
    std::vector<Dim> fShapeX;
    std::vector<Dim> fShapeY;
    std::vector<Dim> fShapeYNotPruned; // needed for fKeepdims=0
    std::string fType;

    // GPU kernel launch tuning parameters
    struct GpuLaunchParams { std::size_t blockSize = 256; std::size_t groups = 1; };


    GpuLaunchParams ComputeGpuLaunchParams() const {
       GpuLaunchParams p;
       if (fInputDimShape || fShapeX.empty() || fShapeY.empty())
          return p;
       std::size_t inLen  = std::stoul(ConvertDimShapeToLength(fShapeX));
       std::size_t outLen = std::stoul(ConvertDimShapeToLength(fShapeY));
       if (outLen == 0)
          return p;
       std::size_t redLen = inLen / outLen;

       std::size_t bs = 32;
       while (bs < redLen && bs < 256) bs *= 2;
       p.blockSize = bs;

       constexpr std::size_t kTargetBlocks = 132;
       if (outLen < kTargetBlocks) {
          std::size_t maxGroups       = kTargetBlocks / outLen;
          std::size_t chunksAvailable = (redLen + bs - 1) / bs;
          p.groups = std::max<std::size_t>(1, std::min(maxGroups, chunksAvailable));
       }
       return p;
    }


public:

   std::string Name() {
      if (fReduceOpMode == ReduceMean)           return "ReduceMean";
      else if (fReduceOpMode == ReduceSumSquare) return "ReduceSumSquare";
      else if (fReduceOpMode == ReduceProd)      return "ReduceProd";
      else if (fReduceOpMode == ReduceSum)       return "ReduceSum";
      else if (fReduceOpMode == ReduceL2)        return "ReduceL2";
      else if (fReduceOpMode == ReduceMax)       return "ReduceMax";
      else if (fReduceOpMode == ReduceMin)       return "ReduceMin";
      return "Invalid";
   }

   std::vector<std::string> GetStdLibs() override {
      std::vector<std::string> libs = { std::string("memory") };
      if (fReduceOpMode == ReduceL2)
         libs.push_back("cmath");
      if (fReduceOpMode == ReduceMax || fReduceOpMode == ReduceMin)
         libs.push_back("limits");
      return libs;
   }

   ROperator_Reduce(){}
   ROperator_Reduce(int keepdims, std::vector<int64_t> attrAxes, std::string nameX, std::string nameAxes, std::string nameY):
   fkeepdims(keepdims), fAttrAxes(attrAxes), fNX(UTILITY::Clean_name(nameX)), fNAxes(UTILITY::Clean_name(nameAxes)), fNY(UTILITY::Clean_name(nameY)) {
      fReduceOpMode = Op;
      
      fInputTensorNames = { fNX };
      if(!fNAxes.empty()){
         fInputTensorNames.emplace_back(fNAxes);
      }

      fOutputTensorNames = { fNY };
   }

   // shape of output tensors given input tensors
   std::vector<Dim> ShapeInference(const std::vector<Dim> & input) {
      auto ret = input;
      auto & outputShape = ret;
      for (size_t j = 0; j < fAttrAxes.size(); j++) {
         if (fAttrAxes[j] < 0) fAttrAxes[j] += outputShape.size();
         if (fAttrAxes[j] < 0 || (size_t) fAttrAxes[j] >= outputShape.size())
            throw std::runtime_error("SOFIE Reduce Op - invalid axes values " + std::to_string(fAttrAxes[j]));
         outputShape[fAttrAxes[j]] = Dim{1};
      }
      fShapeYNotPruned = outputShape;
      if (fkeepdims == 0) {
         auto ax = fAttrAxes;
         std::sort(ax.begin(), ax.end());
         for (size_t j = 0; j < ax.size(); j++) {
            // erase reduced dimensions, but keep last one
            if (outputShape.size() > 0) {
               outputShape.erase(outputShape.begin() + ax[j]);
               for (size_t k = j+1; k < ax.size(); k++)
                  ax[k] -= 1;
            }
         }
      }
      return ret;
   }

   void Initialize(RModel& model) override {

      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE Reduce Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShapeX.empty();
      if (model.IsDynamicTensor(fNX))
         fInputDimShape = true;
      // check if tensor with axes is provided
      if (!fNAxes.empty()) {
         auto ax_shptr = model.GetInitializedTensorData(fNAxes);
         auto ax_ptr = static_cast<int64_t *>(ax_shptr.get());
         auto ax_shape = model.GetTensorShape(fNAxes);
         size_t ax_length = ConvertShapeToLength(ax_shape);
         fAttrAxes = std::vector<int64_t>(ax_ptr, ax_ptr+ax_length);
      } else if (fAttrAxes.empty()) {
         // in case no axes is passed assume full reduction
         fAttrAxes.resize(fShapeX.size());
         for (size_t i = 0; i < fAttrAxes.size(); i++)
            fAttrAxes[i] = i;
      }
      // find shape of Y and add it in the list of intermediate tensors
      fShapeY = ShapeInference(fShapeX);
      if (fkeepdims == 0 && fAttrAxes.size() == fShapeX.size())
         model.MarkScalarTensor(fNY);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      if (model.Verbose()){
         std::cout << Name() << " : " << fNX << " -> " << fNY << " shape " << ConvertDimShapeToString(fShapeY) << std::endl;
      }
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      model.AddNeededStdLib("algorithm");
      if (fReduceOpMode == ReduceMax || fReduceOpMode == ReduceMin)
         model.AddNeededStdLib("limits");
   }

   bool IsReducedAxis(size_t d) const {
      return std::find(fAttrAxes.begin(), fAttrAxes.end(), (int64_t)d) != fAttrAxes.end();
   }

   // number of input elements reduced into each output element, as generated-code text;
   // used by the CPU loops and passed to the GPU kernel by the launch
   std::string ReducedLengthExpr() const {
      return "(" + ConvertDimShapeToLength(fShapeX) + ") / (" + ConvertDimShapeToLength(fShapeY) + ")";
   }

   EFusionMappingType GetFusionMappingType() const override
   {
      if (fHasStridedInput || fInputDimShape || fShapeX.empty() || fShapeY.empty())
         return EFusionMappingType::Unsupported;
      return EFusionMappingType::ManyToMany;
   }

   std::vector<size_t> GetFusionDataInputIndices() const override
   {
      return {0};
   }

   bool IsFusionReduction() const override { return !fHasStridedInput && !fInputDimShape && !fShapeX.empty() && !fShapeY.empty(); }

   std::string GetFusionReductionInitExpr() const override
   {
      if (fReduceOpMode == ReduceProd)
         return "static_cast<T>(1)";
      if (fReduceOpMode == ReduceMax)
         return "std::numeric_limits<T>::lowest()";
      return "static_cast<T>(0)";
   }

   std::string GetFusionReductionAccumulateExpr(const std::string &accumulator, const std::string &value) const override
   {
      if (fReduceOpMode == ReduceProd)
         return "((" + accumulator + ") * (" + value + "))";
      if (fReduceOpMode == ReduceSumSquare || fReduceOpMode == ReduceL2)
         return "((" + accumulator + ") + (" + value + ") * (" + value + "))";
      if (fReduceOpMode == ReduceMax)
         return "((" + value + ") > (" + accumulator + ") ? (" + value + ") : (" + accumulator + "))";
      return "((" + accumulator + ") + (" + value + "))";
   }

   std::string GetFusionReductionCombineExpr(const std::string &left, const std::string &right) const override
   {
      if (fReduceOpMode == ReduceProd)
         return "((" + left + ") * (" + right + "))";
      if (fReduceOpMode == ReduceMax)
         return "((" + right + ") > (" + left + ") ? (" + right + ") : (" + left + "))";
      return "((" + left + ") + (" + right + "))";
   }

   std::string GetFusionReductionFinalizeExpr(const std::string &accumulator, const std::string &reducedLength) const override
   {
      if (fReduceOpMode == ReduceMean)
         return "((" + accumulator + ") / static_cast<T>(" + reducedLength + "))";
      if (fReduceOpMode == ReduceL2)
         return "std::sqrt(" + accumulator + ")";
      return accumulator;
   }

   // this fusion path only supports fully static shapes: fShapeX/fShapeYNotPruned are
   // Dim-valued to support the dynamic-shape codegen above, but the generic fusion
   // planner (RModel_Fusion_ALPAKA.cxx) only ever calls this once GetFusionMappingType
   // has confirmed the operator is static, so converting them down to plain sizes here
   // is safe.
   std::string GetFusionReductionInputIndexExpr(const std::string &outputIndex, const std::string &reductionIndex,
      const std::vector<Dim> &inputShapeDim, const std::vector<Dim> &outputShapeDim) const override
   {
      if (fInputDimShape || fShapeX.empty() || fShapeY.empty())
         return "";

      std::vector<size_t> inputShape(inputShapeDim.size());
      for (size_t i = 0; i < inputShapeDim.size(); ++i)
         inputShape[i] = inputShapeDim[i].dim;
      std::vector<size_t> outputShape(outputShapeDim.size());
      for (size_t i = 0; i < outputShapeDim.size(); ++i)
         outputShape[i] = outputShapeDim[i].dim;

      std::vector<size_t> shapeX(fShapeX.size());
      for (size_t i = 0; i < fShapeX.size(); ++i)
         shapeX[i] = fShapeX[i].dim;
      std::vector<size_t> shapeYNotPruned(fShapeYNotPruned.size());
      for (size_t i = 0; i < fShapeYNotPruned.size(); ++i)
         shapeYNotPruned[i] = fShapeYNotPruned[i].dim;

      if (inputShape != shapeX || outputShape.size() != fShapeY.size())
         return "";
      for (size_t i = 0; i < outputShape.size(); ++i)
         if (outputShape[i] != fShapeY[i].dim)
            return "";

      const auto inputStrides = UTILITY::ComputeStrideFromShape(shapeX);
      const auto outputStrides = UTILITY::ComputeStrideFromShape(shapeYNotPruned);
      std::vector<size_t> reducedAxes;

      for (size_t dim = 0; dim < shapeX.size(); ++dim) {
         if (IsReducedAxis(dim))
            reducedAxes.push_back(dim);
      }

      if (reducedAxes.empty())
         return "";

      std::vector<size_t> reductionStrides(reducedAxes.size(), 1);
      for (int axisIdx = static_cast<int>(reducedAxes.size()) - 2; axisIdx >= 0; --axisIdx)
         reductionStrides[axisIdx] = reductionStrides[axisIdx + 1] * shapeX[reducedAxes[axisIdx + 1]];

      std::string expression;

      for (size_t dim = 0; dim < shapeX.size(); ++dim) {
         const auto reducedIt = std::find(reducedAxes.begin(), reducedAxes.end(), dim);
         std::string coordinate;

         if (reducedIt != reducedAxes.end()) {
            const size_t reducedIdx = static_cast<size_t>(std::distance(reducedAxes.begin(), reducedIt));
            coordinate = "((" + reductionIndex + " / " + std::to_string(reductionStrides[reducedIdx]) + "u) % " +
                         std::to_string(shapeX[dim]) + "u)";
         } else {
            coordinate = "((" + outputIndex + " / " + std::to_string(outputStrides[dim]) + "u) % " +
                         std::to_string(shapeYNotPruned[dim]) + "u)";
         }

         if (!expression.empty())
            expression += " + ";

         expression += coordinate;
         if (inputStrides[dim] != 1)
            expression += " * " + std::to_string(inputStrides[dim]) + "u";
      }

      return expression;
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty() || fShapeY.empty()) {
         throw std::runtime_error("SOFIE Reduce Op called to Generate without being initialized first");
      }

      std::string inputLength = SOFIE::ConvertDimShapeToLength(fShapeX);
      std::string outputLength = SOFIE::ConvertDimShapeToLength(fShapeY);

      // output stride (or not pruned vector)
      auto outputStrides = SOFIE::UTILITY::ComputeStrideFromShape(fShapeYNotPruned);

      // write here according to size of shape
      // in generation code can be done automatically
      // i0 =  i / stride0  % shape0; i1 = i / stride1 % shape1 and so on
      // and we have for the inverse
      // i = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 ....

      // don't need to divide by last stride s[n-1] since it is 1 by definition

      std::stringstream out;
      out << "\n//----  operator " << Name() << "  " << opName << "\n";
      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      auto readX = [&](const std::string &index) {
         return fHasStridedInput ? "tensor_" + fNX + "[xoff_" + opName + "(" + index + ")]"
                                 : "tensor_" + fNX + "[" + index + "]";
      };
      if (fHasStridedInput)
         out << GenerateStridedOffsetLambda(opName, fNX, fShapeX);
      // check where is reduced axes are first or last one. In these case we can do a faster implementation
      enum EReduceDim {kFirst, kLast, kMiddle};
      EReduceDim reduceDims = kLast;
      int kmin = fShapeX.size()-fAttrAxes.size();
      for (int k = fShapeX.size()-1; k >= kmin; k--) {
         // if k is not a reduced axis is not last ones
         if (!IsReducedAxis(k)) {
            reduceDims = kMiddle;
            break;
         }
      }
      if (reduceDims == kMiddle) {
         reduceDims = kFirst;
         // check if at the beginning
         for (size_t k = 0; k < fAttrAxes.size(); k++) {
            // if k is not a reduced axis is not first ones
            if (!IsReducedAxis(k)) {
               reduceDims = kMiddle;
               break;
            }
         }
      }
      std::string initValue = "0";
      if (fReduceOpMode == ReduceProd)
         initValue = "1";
      else if (fReduceOpMode == ReduceMax)
         initValue = "std::numeric_limits<" + fType + ">::lowest()";
      else if (fReduceOpMode == ReduceMin)
         initValue = "std::numeric_limits<" + fType + ">::max()";

      std::string reducedLength;
      if (fInputDimShape) {
         reducedLength = "reducedLength_" + opName;
         out << "size_t " << reducedLength << " = " << ReducedLengthExpr() << ";\n";
      } else {
         int rLength = std::stoi(inputLength) / std::stoi(outputLength);
         reducedLength = std::to_string(rLength);
      }
      auto accumulate = [&](const std::string &y, const std::string &x) {
         if (fReduceOpMode == ReduceMax)
            return y + " = std::max(" + y + ", " + x + ");\n";
         if (fReduceOpMode == ReduceMin)
            return y + " = std::min(" + y + ", " + x + ");\n";
         if (fReduceOpMode == ReduceProd)
            return y + " *= " + x + ";\n";
         if (fReduceOpMode == ReduceSumSquare || fReduceOpMode == ReduceL2)
            return y + " += " + x + " * " + x + ";\n";
         return y + " += " + x + ";\n";
      };
      if (reduceDims == kLast) {

         out << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
         out << "tensor_" << fNY << "[i] = " << initValue << ";\n";
         out << "for (size_t j = 0; j < " << reducedLength << "; j++) {\n";
         out << accumulate("tensor_" + fNY + "[i]", readX("i * " + reducedLength + " + j"));
         out << "}\n";
         if(fReduceOpMode == ReduceMean)
            out << "tensor_" << fNY << "[i] /= static_cast<float>(" << reducedLength << ");\n";
         else if (fReduceOpMode == ReduceL2)
            out << "tensor_" << fNY << "[i] = std::sqrt(tensor_" << fNY << "[i]);\n";

         out << "}\n";
      } else if (reduceDims == kFirst) {
         out << "std::fill(tensor_" << fNY << ", tensor_" << fNY << " + " << outputLength << ", " << initValue
             << ");\n";

         out << "for (size_t i = 0; i < " << reducedLength << "; i++) {\n";
         out << "for (size_t j = 0; j < " << outputLength << "; j++) {\n";
         out << accumulate("tensor_" + fNY + "[j]", readX("i * " + outputLength + " + j"));
         out << "}\n";
         out << "}\n";
         if(fReduceOpMode == ReduceMean) {
            out << "for (size_t j = 0; j < " << outputLength << "; j++) {\n";
            out << "tensor_" << fNY << "[j] /= static_cast<float>(" << reducedLength << ");\n";
            out << "}\n";
         } else if (fReduceOpMode == ReduceL2) {
            out << "for (size_t j = 0; j < " << outputLength << "; j++) {\n";
            out << "tensor_" << fNY << "[j] = std::sqrt(tensor_" << fNY << "[j]);\n";
            out << "}\n";
         }
      }
      else
      {
         out << "std::fill(tensor_" << fNY << ", tensor_" << fNY << " + " << outputLength << ", " << initValue
             << ");\n";

         size_t dim = fShapeX.size();

         out << "{\n";
         out << "size_t inputIndex = 0;\n";
         std::string outputIndex = "0";
         for (size_t k = 0; k < dim; k++) {
            out << "for (size_t i_" << k << " = 0; i_" << k << " < (" << fShapeX[k].GetVal() << "); i_" << k << "++) {\n";
            if (!IsReducedAxis(k)) {
               std::string next = "outputIndex_" + std::to_string(k);
               out << "size_t " << next << " = " << outputIndex << " + i_" << k << " * (" << outputStrides[k].GetVal() << ");\n";
               outputIndex = next;
            }
         }
         out << accumulate("tensor_" + fNY + "[" + outputIndex + "]", readX("inputIndex"));
         out << "inputIndex++;\n";
         for (size_t k = dim; k > 0; k--) {
            out << "}\n";
         }
         out << "}\n";
         if (fReduceOpMode == ReduceMean) {
            out << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
            out << "tensor_" << fNY << "[i] /= static_cast<float>(" << reducedLength << ");\n";
            out << "}\n";
         } else if (fReduceOpMode == ReduceL2) {
            out << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
            out << "tensor_" << fNY << "[i] = std::sqrt(tensor_" << fNY << "[i]);\n";
            out << "}\n";
         }
      }

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/, const std::vector<std::string> &dynParamNames) override {
      if (fShapeX.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Reduce Op called to Generate without being initialized first");

      const std::size_t Dx        = fShapeX.size();
      auto inputStrides            = UTILITY::ComputeStrideFromShape(fShapeX);
      auto outputStrides           = UTILITY::ComputeStrideFromShape(fShapeYNotPruned);

      // Partition axes into keep (non-reduced) and reduce sets.
      std::vector<std::size_t> redAxes, keepAxes;
      for (std::size_t d = 0; d < Dx; ++d) {
         if (IsReducedAxis(d))
            redAxes.push_back(d);
         else
            keepAxes.push_back(d);
      }


      enum EReduceDim { kFirst, kLast, kMiddle };
      EReduceDim reduceDims = kLast;
      {
         int kmin = (int)Dx - (int)redAxes.size();
         for (int k = (int)Dx - 1; k >= kmin; k--) {
            if (!IsReducedAxis(k)) { reduceDims = kMiddle; break; }
         }
      }
      if (reduceDims == kMiddle) {
         reduceDims = kFirst;
         for (std::size_t k = 0; k < redAxes.size(); k++) {
            if (!IsReducedAxis(k)) { reduceDims = kMiddle; break; }
         }
      }

      // row-major strides for decomposing the flat reduction index into coordinates
      // redStrides[i] = product of fShapeX[redAxes[j]] for j > i
      std::vector<std::string> redStrides(redAxes.size(), "1");
      for (int ri = (int)redAxes.size() - 2; ri >= 0; --ri)
         redStrides[ri] = "(" + redStrides[ri + 1] + " * " + fShapeX[redAxes[ri + 1]].GetVal() + ")";

      bool dyn = fInputDimShape;
      GpuLaunchParams lp = ComputeGpuLaunchParams();
      // Dynamic shapes always take the groups-capable, symbolic-block-size
      // form since the split decision can only be made at runtime.
      bool useGroupsDecomp = dyn || lp.groups > 1;
      std::string bsExpr     = dyn ? "blockDim"      : (std::to_string(lp.blockSize) + "u");
      std::string halfBsExpr = dyn ? "(blockDim / 2)" : (std::to_string(lp.blockSize / 2) + "u");
      std::string shmemCap   = dyn ? "256"            : std::to_string(lp.blockSize);

      std::string kname = "ReduceKernel_" + Name() + "_" + fNY;

      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      const std::string xin = fHasStridedInput ? "input[sofie_strided_offset(layoutX, in_idx)]" : "input[in_idx]";

      std::string op;
      op  = "\n//------ " + Name() + "_KERNEL_ALPAKA (block parallel reduction)\n";
      op += "struct " + kname + " {\n";
      op += "template<typename TAcc, typename T>\n";
      op += "ALPAKA_FN_ACC void operator()(\n";
      op += "TAcc const& acc,\n";
      op += "T const* __restrict__ input,\n";
      op += "T* __restrict__ output,\n";
      op += "std::size_t const reducedLength,\n";
      op += "std::size_t const outputLength";
      if (useGroupsDecomp)
         op += ",\nstd::size_t const groups";
      for (auto &p : dynParamNames)
         op += ",\nstd::size_t const " + p;
      if (fHasStridedInput)
         op += ",\nsofie_strided_layout<" + std::to_string(Dx) + "> const layoutX";
      op += ") const {\n\n";

      // ---- shared memory ----
      op += "auto& shmem = alpaka::declareSharedVar<T[" + shmemCap + "], __COUNTER__>(acc);\n\n";

      // ---- block/thread addressing ----
      op += "auto const thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads >(acc)[0];\n";
      if (dyn)
         op += "auto const blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      if (useGroupsDecomp) {
         op += "auto const flat_block = alpaka::getIdx<alpaka::Grid, alpaka::Blocks >(acc)[0];\n";
         op += "auto const out_idx    = flat_block % outputLength;\n";
         op += "auto const group_id   = flat_block / outputLength;\n";
         op += "if (flat_block >= outputLength * groups) return;\n\n";
      } else {
         op += "auto const out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks >(acc)[0];\n";
         op += "if (out_idx >= outputLength) return;\n\n";
      }

      // ---- decode output (keep-axis) coordinates from out_idx ----
      // (only needed for the generic kMiddle path; kFirst/kLast compute
      // in_idx directly from out_idx without per-axis coordinates)
      if (reduceDims == kMiddle) {
         for (std::size_t d = 0; d < Dx; ++d) {
            if (!IsReducedAxis(d)) {
               op += "std::size_t const oy_" + std::to_string(d)
                     + " = (out_idx / (" + outputStrides[d].GetVal() + ")) % ("
                     + fShapeYNotPruned[d].GetVal() + ");\n";
            }
         }
         op += "\n";
      }

      // ---- thread-stride partial accumulation over reduction axis ----
      std::string startVal;
      if (Op == ReduceProd)       startVal = "static_cast<T>(1)";
      else if (Op == ReduceMax)   startVal = "std::numeric_limits<T>::lowest()";
      else if (Op == ReduceMin)   startVal = "std::numeric_limits<T>::max()";
      else                        startVal = "static_cast<T>(0)";
      op += "T partial = " + startVal + ";\n";
      if (useGroupsDecomp)
         op += "for (std::size_t r = thread_id + group_id * " + bsExpr + "; r < reducedLength; r += "
               + bsExpr + " * groups) {\n";
      else
         op += "for (std::size_t r = thread_id; r < reducedLength; r += " + bsExpr + ") {\n";

      if (reduceDims == kLast) {
         // reduced axes are the trailing contiguous axes: input is laid out
         // as [outputLength, reducedLength] row-major.
         op += "std::size_t const in_idx = out_idx * reducedLength + r;\n";
      } else if (reduceDims == kFirst) {
         // reduced axes are the leading contiguous axes: input is laid out
         // as [reducedLength, outputLength] row-major.
         op += "std::size_t const in_idx = r * outputLength + out_idx;\n";
      } else {
         // Decode flat reduction index r into per-axis coordinates.
         for (std::size_t ri = 0; ri < redAxes.size(); ++ri) {
            std::size_t rd = redAxes[ri];
            op += "std::size_t const r_" + std::to_string(rd)
                  + " = (r / (" + redStrides[ri] + ")) % ("
                  + fShapeX[rd].GetVal() + ");\n";
         }

         // Compute flat input index.
         op += "std::size_t const in_idx =\n";
         for (std::size_t d = 0; d < Dx; ++d) {
            std::string coord = IsReducedAxis(d) ? "r_" + std::to_string(d) : "oy_" + std::to_string(d);
            op += coord + " * (" + inputStrides[d].GetVal() + ")";
            op += (d + 1 < Dx) ? " +\n" : ";\n";
         }
      }

      // Partial accumulation step.
      if (Op == ReduceProd)
         op += "partial *= " + xin + ";\n";
      else if (Op == ReduceSum || Op == ReduceMean)
         op += "partial += " + xin + ";\n";
      else if (Op == ReduceSumSquare || Op == ReduceL2)
         op += "partial += " + xin + " * " + xin + ";\n";
      else if (Op == ReduceMax)
         op += "if (" + xin + " > partial) partial = " + xin + ";\n";
      else if (Op == ReduceMin)
         op += "if (" + xin + " < partial) partial = " + xin + ";\n";

      op += "}\n\n"; // end thread-stride loop

      // ---- store in shared memory and synchronise ----
      op += "shmem[thread_id] = partial;\n";
      op += "alpaka::syncBlockThreads(acc);\n\n";

      // ---- shared-memory tree reduction down to one warp's worth of partials ----
      // Stops once the active thread count fits in a single warp; the final
      // warp-width combine below uses shuffles instead of shared memory.
      op += "std::int32_t const warpSize = alpaka::warp::getSize(acc);\n";
      op += "for (std::size_t s = " + halfBsExpr
            + "; s >= static_cast<std::size_t>(warpSize); s >>= 1u) {\n";
      op += "if (thread_id < s) {\n";
      if (Op == ReduceProd)
         op += "shmem[thread_id] *= shmem[thread_id + s];\n";
      else if (Op == ReduceMax)
         op += "if (shmem[thread_id + s] > shmem[thread_id]) shmem[thread_id] = shmem[thread_id + s];\n";
      else if (Op == ReduceMin)
         op += "if (shmem[thread_id + s] < shmem[thread_id]) shmem[thread_id] = shmem[thread_id + s];\n";
      else
         op += "shmem[thread_id] += shmem[thread_id + s];\n";
      op += "}\n";
      op += "alpaka::syncBlockThreads(acc);\n";
      op += "}\n\n";

      op += "if (thread_id < static_cast<std::size_t>(warpSize)) {\n";
      op += "T val = shmem[thread_id];\n";
      op += "for (std::int32_t offset = warpSize / 2; offset > 0; offset >>= 1) {\n";
      op += "T const other = alpaka::warp::shfl_down(acc, val, static_cast<std::uint32_t>(offset));\n";
      if (Op == ReduceProd)
         op += "val *= other;\n";
      else if (Op == ReduceMax)
         op += "if (other > val) val = other;\n";
      else if (Op == ReduceMin)
         op += "if (other < val) val = other;\n";
      else
         op += "val += other;\n";
      op += "}\n\n";

      op += "if (thread_id == 0u) {\n";
      if (dyn) {
         op += "T result = val;\n";
         op += "if (groups == 1u) {\n";
         if (Op == ReduceMean)
            op += "result /= static_cast<T>(reducedLength);\n";
         else if (Op == ReduceL2)
            op += "result = std::sqrt(result);\n";
         op += "}\n";
         op += "output[flat_block] = result;\n";
      } else if (lp.groups > 1) {
         op += "output[flat_block] = val;\n";
      } else {
         op += "T result = val;\n";
         if (Op == ReduceMean)
            op += "result /= static_cast<T>(reducedLength);\n";
         else if (Op == ReduceL2)
            op += "result = std::sqrt(result);\n";
         op += "output[out_idx] = result;\n";
      }
      op += "}\n";
      op += "}\n"; // end if (thread_id < warpSize)

      op += "}\n"; // end operator()
      op += "};\n";     // end struct

      if (useGroupsDecomp) {
         std::string kname2 = "ReduceFinalizeKernel_" + Name() + "_" + fNY;
         std::string groupsDesc = dyn ? "a runtime-determined number of" : std::to_string(lp.groups);
         op += "\n//------ " + Name() + "_FINALIZE_KERNEL_ALPAKA (combine " + groupsDesc
               + " partial blocks per output)\n";
         op += "struct " + kname2 + " {\n";
         op += "template<typename TAcc, typename T>\n";
         op += "ALPAKA_FN_ACC void operator()(\n";
         op += "TAcc const& acc,\n";
         op += "T const* __restrict__ partial,\n";
         op += "T* __restrict__ output,\n";
         op += "std::size_t const reducedLength,\n";
         op += "std::size_t const outputLength,\n";
         op += "std::size_t const groups) const {\n\n";
         op += "auto const out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += "if (out_idx >= outputLength) return;\n\n";
         op += "T result = " + startVal + ";\n";
         op += "for (std::size_t g = 0; g < groups; ++g) {\n";
         op += "T const v = partial[g * outputLength + out_idx];\n";
         if (Op == ReduceProd)
            op += "result *= v;\n";
         else if (Op == ReduceMax)
            op += "if (v > result) result = v;\n";
         else if (Op == ReduceMin)
            op += "if (v < result) result = v;\n";
         else
            op += "result += v;\n";
         op += "}\n\n";
         if (Op == ReduceMean)
            op += "result /= static_cast<T>(reducedLength);\n";
         else if (Op == ReduceL2)
            op += "result = std::sqrt(result);\n";
         op += "output[out_idx] = result;\n";
         op += "}\n"; // end operator()
         op += "};\n";     // end struct
      }

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      bool dyn = fInputDimShape;
      GpuLaunchParams lp = ComputeGpuLaunchParams();
      bool needsScratch = dyn || lp.groups > 1;
      std::string kname = "ReduceKernel_" + Name() + "_" + fNY;
      std::string op = kname + " reduceKernel_" + Name() + "_" + fNY + ";\n";
      if (needsScratch) {
         std::string kname2 = "ReduceFinalizeKernel_" + Name() + "_" + fNY;
         op += kname2 + " reduceFinalizeKernel_" + Name() + "_" + fNY + ";\n";
         // The scratch buffer holds partial reduction results, so it must be
         // T's actual element type — a boolean-derived reduction like ReduceSum
         // over an int64 tensor would otherwise get a float scratch buffer,
         // which fails to compile against the int64 input/output kernel args.
         op += "std::unique_ptr<alpaka::Buf<Acc, " + fType + ", Dim, Idx>> reduceScratch_" + fNY + ";\n";
         op += "std::size_t reduceScratchCapacity_" + fNY + " = 0;\n";
      }
      return op;
   }

   std::string Generate_GPU_ALPAKA(std::string /*opName*/, const std::vector<std::string> &dynParamNames) override {
      if (fShapeX.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Reduce Op called to Generate without being initialized first");

      std::string outputLength  = ConvertDimShapeToLength(fShapeY);
      std::string reducedLength = ReducedLengthExpr();
      std::string kname = "reduceKernel_" + Name() + "_" + fNY;

      bool dyn = fInputDimShape;
      GpuLaunchParams lp = ComputeGpuLaunchParams();
      std::string bs = std::to_string(lp.blockSize);

      std::string dynArgs;
      for (auto &p : dynParamNames) dynArgs += ", static_cast<std::size_t>(" + p + ")";
      // the layout of a strided input is the last argument of the reduction kernel
      if (fHasStridedInput)
         dynArgs += ", layout_" + fNY + "_X";

      std::stringstream out;
      out << "\n//------ " << Name() << "_GPU_ALPAKA\n";
      if (fHasStridedInput)
         out << GenerateStridedBroadcastLayout(fNY + "_X", fNX, fShapeX, fShapeX.size(), fShapeX);

      if (dyn) {
         std::string rl = "reduceLaunch_" + fNY;
         out << "std::size_t const " << rl << "_reducedLength = " << reducedLength << ";\n";
         out << "std::size_t const " << rl << "_outputLength  = " << outputLength << ";\n";
         out << "std::size_t " << rl << "_blockSize = 32u;\n";
         out << "while (" << rl << "_blockSize < " << rl << "_reducedLength && " << rl
             << "_blockSize < 256u) " << rl << "_blockSize *= 2u;\n";
         out << "std::size_t " << rl << "_groups = 1u;\n";
         out << "if (" << rl << "_outputLength > 0 && " << rl << "_outputLength < 132u) {\n";
         out << "std::size_t const " << rl << "_maxGroups = 132u / " << rl << "_outputLength;\n";
         out << "std::size_t const " << rl << "_chunks    = (" << rl << "_reducedLength + " << rl
             << "_blockSize - 1u) / " << rl << "_blockSize;\n";
         out << rl << "_groups = std::max<std::size_t>(1u, std::min(" << rl << "_maxGroups, " << rl
             << "_chunks));\n";
         out << "}\n\n";

         out << "if (" << rl << "_groups > 1u) {\n";
         out << "std::size_t const " << rl << "_capacity = " << rl << "_outputLength * " << rl
             << "_groups;\n";
         out << "if (!reduceScratch_" << fNY << " || reduceScratchCapacity_" << fNY << " < " << rl
             << "_capacity) {\n";
         out << "reduceScratch_" << fNY
             << " = std::make_unique<alpaka::Buf<Acc, " << fType << ", Dim, Idx>>(alpaka::allocBuf<"
             << fType << ", Idx>(devAcc, Ext1D::all(Idx{" << rl
             << "_capacity})));\n";
         out << "reduceScratchCapacity_" << fNY << " = " << rl << "_capacity;\n";
         out << "}\n";
         out << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNY << "(\n";
         out << "Vec::all(Idx{" << rl << "_capacity}),\n";
         out << "Vec::all(Idx{" << rl << "_blockSize}),\n";
         out << "Vec::all(Idx{1u}));\n";
         out << "alpaka::exec<Acc>(queue, workDiv_" << fNY
             << ", " << kname
             << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
             << ", alpaka::getPtrNative(*reduceScratch_" << fNY << ")"
             << ", " << rl << "_reducedLength"
             << ", " << rl << "_outputLength"
             << ", " << rl << "_groups"
             << dynArgs << ");\n\n";

         std::string kname2 = "reduceFinalizeKernel_" + Name() + "_" + fNY;
         out << "auto const elementsPerThread_" << fNY << "_finalize = Vec::all(static_cast<Idx>(1));\n";
         out << "auto const elementsPerGrid_" << fNY << "_finalize   = Vec::all(Idx{" << rl
             << "_outputLength});\n";
         out << "auto const workDiv_" << fNY << "_finalize = sofie_workdiv(elementsPerGrid_" << fNY
             << "_finalize);\n";
         out << "alpaka::exec<Acc>(queue, workDiv_" << fNY << "_finalize"
             << ", " << kname2
             << ", alpaka::getPtrNative(*reduceScratch_" << fNY << ")"
             << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
             << ", " << rl << "_reducedLength"
             << ", " << rl << "_outputLength"
             << ", " << rl << "_groups);\n";
         out << "} else {\n";
         out << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNY << "(\n";
         out << "Vec::all(Idx{" << rl << "_outputLength}),\n";
         out << "Vec::all(Idx{" << rl << "_blockSize}),\n";
         out << "Vec::all(Idx{1u}));\n";
         out << "alpaka::exec<Acc>(queue, workDiv_" << fNY
             << ", " << kname
             << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
             << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
             << ", " << rl << "_reducedLength"
             << ", " << rl << "_outputLength"
             << ", " << rl << "_groups"
             << dynArgs << ");\n";
         out << "}\n";
      } else if (lp.groups > 1) {
         std::string totalBlocks = std::to_string(std::stoul(outputLength) * lp.groups);
         out << "if (!reduceScratch_" << fNY << ") {\n";
         out << "reduceScratch_" << fNY
             << " = std::make_unique<alpaka::Buf<Acc, " << fType << ", Dim, Idx>>(alpaka::allocBuf<"
             << fType << ", Idx>(devAcc, Ext1D::all(Idx{" << totalBlocks
             << "})));\n";
         out << "reduceScratchCapacity_" << fNY << " = " << totalBlocks << ";\n";
         out << "}\n";

         out << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNY << "(\n";
         out << "Vec::all(Idx{" << totalBlocks << "u}),\n";
         out << "Vec::all(Idx{" << bs << "u}),\n";
         out << "Vec::all(Idx{1u}));\n";
         out << "alpaka::exec<Acc>(queue, workDiv_" << fNY
             << ", " << kname
             << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
             << ", alpaka::getPtrNative(*reduceScratch_" << fNY << ")"
             << ", static_cast<std::size_t>(" << reducedLength << ")"
             << ", static_cast<std::size_t>(" << outputLength << ")"
             << ", static_cast<std::size_t>(" << lp.groups << "u)"
             << dynArgs << ");\n";

         std::string kname2 = "reduceFinalizeKernel_" + Name() + "_" + fNY;
         out << "auto const elementsPerThread_" << fNY << "_finalize = Vec::all(static_cast<Idx>(1));\n";
         out << "auto const elementsPerGrid_" << fNY << "_finalize   = Vec::all(Idx{" << outputLength << "});\n";
         out << "auto const workDiv_" << fNY << "_finalize = sofie_workdiv(elementsPerGrid_" << fNY << "_finalize);\n";
         out << "alpaka::exec<Acc>(queue, workDiv_" << fNY << "_finalize"
             << ", " << kname2
             << ", alpaka::getPtrNative(*reduceScratch_" << fNY << ")"
             << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
             << ", static_cast<std::size_t>(" << reducedLength << ")"
             << ", static_cast<std::size_t>(" << outputLength << ")"
             << ", static_cast<std::size_t>(" << lp.groups << "u));\n";
      } else {
         // Grid: one block per output element; Block: threads cooperate to
         // reduce the corresponding slice.
         out << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNY << "(\n";
         out << "Vec::all(Idx{" << outputLength << "}),\n";
         out << "Vec::all(Idx{" << bs << "u}),\n";
         out << "Vec::all(Idx{1u}));\n";
         out << "alpaka::exec<Acc>(queue, workDiv_" << fNY
             << ", " << kname
             << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
             << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
             << ", static_cast<std::size_t>(" << reducedLength << ")"
             << ", static_cast<std::size_t>(" << outputLength << ")"
             << dynArgs << ");\n";
      }

      return out.str();
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_Reduce

