#ifndef SOFIE_ROPERATOR_ALPAKA
#define SOFIE_ROPERATOR_ALPAKA

#include "SOFIE/operators/ROperator.hxx"

#include <algorithm>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace SOFIE {

inline std::set<OperatorKind> FusableKinds = { OperatorKind::RELU, OperatorKind::LAYERNORM, OperatorKind::BATCHNORM};

/// Interface of the operators which can be generated for the GPU with alpaka: the
/// generation of their kernels and of their launch.
class ROperator_ALPAKA : public ROperator {

public:
   virtual std::string Generate_GPU_ALPAKA(std::string OpName){ return "";} //expect unique opName for each operator within the same RModel
   //dynParamNames: the model's shape parameters in infer-argument order
   virtual std::string Generate_GPU_ALPAKA(std::string OpName, const std::vector<std::string> & /*dynParamNames*/) {
      return Generate_GPU_ALPAKA(OpName);
   }
   virtual std::string GenerateInitCode_GPU_ALPAKA() { return "";};
   // generate code to reset recurrent/stateful buffers (called once per file boundary)
   virtual std::string GenerateResetStateCode_GPU_ALPAKA() { return ""; }
   virtual std::vector<std::string> GetPersistentTensorNames_GPU_ALPAKA() const { return {}; }
   virtual std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) { return ""; }
   virtual std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> & /*dynParamNames*/) {
      return Generate_GPU_Kernel_ALPAKA(opName);
   }
   virtual std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) { return ""; }
   virtual std::string GetFusableOutputTensorName() { return "";}
   virtual std::string GetBlasConfig() { return ""; }
   virtual std::vector<std::string> GetBlasConfigs() {
      auto c = GetBlasConfig();
      if (c.empty())
         return {};
      return {c};
   }
   virtual void UpdateFusableTensorName(std::string, const std::function<void(const std::string&)>& removal_func){ return;};

   // Elementwise kernel fusion interface
   virtual bool IsElementwise() const { return false; }
   // Returns the C++ expression applying this op to inputVar (a local T variable) for fused kernel generation
   virtual std::string GetElementwiseExpr(const std::string& /*inputVar*/) const { return ""; }

   virtual EFusionMappingType GetFusionMappingType() const
   {
      return IsElementwise() ? EFusionMappingType::OneToOne : EFusionMappingType::Unsupported;
   }

   // Returns the expression produced by this operator from its input
   // expressions. The default implementation adapts the existing unary
   // GetElementwiseExpr interface
   virtual std::string GetFusionExpr(const std::vector<std::string> &inputs) const
   {
      if (inputs.size() != 1)
         return "";

      return GetElementwiseExpr(inputs[0]);
   }

   virtual bool SupportsFusionTypes(const std::vector<ETensorType> &inputTypes, ETensorType outputType) const
   {
      if (outputType != ETensorType::FLOAT)
         return false;

      return std::all_of(inputTypes.begin(), inputTypes.end(), [](ETensorType type) {
         return type == ETensorType::FLOAT;
      });
   }

   virtual std::string GetFusionInputIndexExpr(size_t /*inputIndex*/, const std::string &/*outputIndex*/,
                                                const std::vector<Dim> &/*inputShape*/,
                                                const std::vector<Dim> &/*outputShape*/) const
   {
      return "";
   }

   virtual std::string GetFusionInputIndexExprForOutput(size_t inputIndex, size_t /*outputTensorIndex*/,
                                                         const std::string &outputIndex,
                                                         const std::vector<Dim> &inputShape,
                                                         const std::vector<Dim> &outputShape) const
   {
      return GetFusionInputIndexExpr(inputIndex, outputIndex, inputShape, outputShape);
   }

   virtual std::string GetFusionInputConditionExpr(size_t /*inputIndex*/, const std::string &/*outputIndex*/,
      const std::vector<Dim> &/*inputShape*/, const std::vector<Dim> &/*outputShape*/) const
   {
      return "";
   }

   // Optional aggregation semantics for ManyToMany operators such as reductions.
   // The mapping category remains ManyToMany; these methods describe how the
   // many input values are accumulated when fused into a cooperative kernel.
   virtual bool IsFusionReduction() const { return false; }
   virtual std::string GetFusionReductionInitExpr() const { return ""; }
   virtual std::string GetFusionReductionAccumulateExpr(const std::string &/*accumulator*/, const std::string &/*value*/) const { return ""; }
   virtual std::string GetFusionReductionCombineExpr(const std::string &/*left*/, const std::string &/*right*/) const { return ""; }
   virtual std::string GetFusionReductionFinalizeExpr(const std::string &/*accumulator*/, const std::string &/*reducedLength*/) const { return ""; }
   virtual std::string GetFusionReductionInputIndexExpr(const std::string &/*outputIndex*/, const std::string &/*reductionIndex*/,
      const std::vector<Dim> &/*inputShape*/, const std::vector<Dim> &/*outputShape*/) const
   {
      return "";
   }

   virtual std::vector<size_t> GetFusionDataInputIndices() const
   {
      std::vector<size_t> indices;
      indices.reserve(fInputTensorNames.size());

      for (size_t i = 0; i < fInputTensorNames.size(); ++i)
         indices.push_back(i);

      return indices;
   }

   void MarkOutputAsPooled(std::string_view name) { fPooledOutputNames.insert(std::string(name)); }
   bool IsOutputPooled(std::string_view name) const { return fPooledOutputNames.count(std::string(name)) > 0; }

protected:
   std::set<std::string> fPooledOutputNames;

   /// CPU: GPU kernels read a strided input X as `X[sofie_strided_offset(layoutX, logicalIndex)]`
   static std::string StridedKernelRead(const std::string &ptr, const std::string &layout, const std::string &index)
   {
      return ptr + "[sofie_strided_offset(" + layout + ", " + index + ")]";
   }

   /// GPU kernel: `code` reads the strided inputs as <param>[index] and its signature ends with `lastParam) const {`;
   /// returns the code of a kernel receiving the layouts of the strided inputs as additional last parameters
   std::string RewriteKernelStridedInputs(std::string code, const std::vector<StridedInputInfo> &inputs,
                                          const std::string &lastParam) const
   {
      std::string layouts;
      for (const auto &in : inputs) {
         if (!in.strided)
            continue;
         code = RewriteIndexedReads(code, in.param, [&](const std::string &index) {
            return "sofie_strided_offset(layout_" + in.param + ", " + index + ")";
         });
         layouts += ",\nsofie_strided_layout<" + std::to_string(in.shape.size()) + "> const layout_" +
                    in.param;
      }
      const std::string signatureEnd = lastParam + ") const {";
      for (size_t pos = code.find(signatureEnd); pos != std::string::npos; pos = code.find(signatureEnd, pos + 1))
         code.replace(pos, signatureEnd.size(), lastParam + layouts + ") const {");
      return code;
   }

   /// GPU launch: code defining the layouts of the strided inputs; kernel arguments are given by StridedLayoutArgs
   std::string StridedLaunchLayouts(const std::string &id, const std::vector<StridedInputInfo> &inputs) const
   {
      std::string code;
      for (const auto &in : inputs)
         if (in.strided)
            code += GenerateStridedBroadcastLayout(id + "_" + in.param, in.tensor, in.shape, in.shape.size(), in.shape);
      return code;
   }

   /// GPU launch: the arguments to append to the kernel call for the layouts of the strided inputs
   static std::string StridedLayoutArgs(const std::string &id, const std::vector<StridedInputInfo> &inputs)
   {
      std::string args;
      for (const auto &in : inputs)
         if (in.strided)
            args += ", layout_" + id + "_" + in.param;
      return args;
   }

   /// GPU: source of an elementwise kernel reading its input through a strided layout
   /// `expr` maps the name of a variable holding the input element to the expression of the output element;
   /// `extraParams` declares additional kernel parameters
   std::string GenerateStridedUnaryKernel(const std::string &kernelName, const std::string &title,
                                          const std::function<std::string(const std::string &)> &expr,
                                          const std::string &extraParams = "") const
   {
      std::string op = "\n//------ " + title + "_STRIDED_KERNEL_ALPAKA\n";
      op += "struct " + kernelName + " {\n";
      op += "template<typename TAcc, typename T, std::size_t R>\n";
      op += "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, "
                 "std::size_t numElements, sofie_strided_layout<R> const layout" + extraParams + ") const {\n";
      op += "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += "if (idx < numElements) {\n";
      op += "auto const src = sofie_strided_offset(layout, idx);\n";
      op += "T const x = data[src];\n";
      op += "out[idx] = " + expr("x") + ";\n";
      op += "}\n";
      op += "}\n";
      op += "};\n";
      return op;
   }

   /// GPU: launch of a strided elementwise kernel (member `kernelMember`) on the graph input nX
   /// `extraArgs` are the additional kernel arguments matching `extraParams` of the kernel
   std::string GenerateStridedUnaryLaunch(const std::string &opName, const std::string &kernelMember,
                                          const std::string &title, const std::string &nX, const std::string &nY,
                                          const std::vector<Dim> &shape, const std::string &extraArgs = "") const
   {
      const size_t rank = shape.size();
      const std::string length = ConvertDimShapeToLength(shape);
      std::stringstream out;
      out << "\n//------ " << title << "_STRIDED_GPU_ALPAKA\n";
      out << GenerateInputStrideCode(opName, nX, shape);
      out << "sofie_strided_layout<" << rank << "> layout_" << opName << ";\n";
      for (size_t d = 0; d < rank; d++) {
         out << "layout_" << opName << ".shape[" << d << "] = static_cast<std::size_t>(" << shape[d].GetVal()
             << ");\n";
         out << "layout_" << opName << ".stride[" << d << "] = stride_" << opName << "[" << d << "];\n";
      }
      out << "auto const elementsPerGrid_" << opName << " = Vec::all(Idx{" << length << "});\n";
      out << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName << ", "
          << kernelMember << ", alpaka::getPtrNative(deviceBuf_" << nX << "), alpaka::getPtrNative(deviceBuf_" << nY
          << "), static_cast<Idx>(" << length << "), layout_" << opName << extraArgs << ");\n";
      out << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   /// GPU: code defining `layout_<id>` (sofie_strided_layout<outRank>) holding the output shape and, for each output
   /// dim, the stride of the input X (0 for the broadcast dims), so that sofie_strided_offset(layout, outputIndex)
   /// is the offset of the input element read for that output element.
   std::string GenerateStridedBroadcastLayout(const std::string &id, const std::string &name,
                                              const std::vector<Dim> &paddedShape, size_t origRank,
                                              const std::vector<Dim> &shapeY) const
   {
      const size_t outRank = shapeY.size();
      const size_t pad = paddedShape.size() - origRank;
      const std::vector<Dim> origShape(paddedShape.begin() + pad, paddedShape.end());
      std::stringstream out;
      out << GenerateInputStrideCode(id, name, origShape);
      out << "sofie_strided_layout<" << outRank << "> layout_" << id << ";\n";
      for (size_t d = 0; d < outRank; d++) {
         out << "layout_" << id << ".shape[" << d << "] = static_cast<std::size_t>(" << shapeY[d].GetVal() << ");\n";
         // dim d of the output is dim (d - (outRank - paddedShape.size()) - pad) of the original input
         const long k = static_cast<long>(d) - static_cast<long>(outRank - paddedShape.size()) - static_cast<long>(pad);
         if (k < 0 || paddedShape[k + pad].GetVal() == "1")
            out << "layout_" << id << ".stride[" << d << "] = 0;\n";
         else
            out << "layout_" << id << ".stride[" << d << "] = stride_" << id << "[" << k << "];\n";
      }
      return out.str();
   }
};

/// the alpaka interface of an operator known to have one
inline ROperator_ALPAKA &AlpakaOp(ROperator &op) { return dynamic_cast<ROperator_ALPAKA &>(op); }

/// the alpaka interface of an operator, or nullptr if it has no alpaka implementation
inline ROperator_ALPAKA *GetAlpaka(ROperator &op) { return dynamic_cast<ROperator_ALPAKA *>(&op); }
inline const ROperator_ALPAKA *GetAlpaka(const ROperator &op) { return dynamic_cast<const ROperator_ALPAKA *>(&op); }

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_ALPAKA
