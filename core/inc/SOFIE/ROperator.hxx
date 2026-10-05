#ifndef SOFIE_ROPERATOR
#define SOFIE_ROPERATOR

#include "SOFIE/SOFIE_common.hxx"

#include <algorithm>
#include <functional>
#include <set>
#include <span>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

namespace SOFIE{

class RModel;

enum class OperatorKind {
   GEMM = 0,
   LAYERNORM = 1,
   RELU = 2,
   CONSTANT = 3,
   CONSTANTOFSHAPE = 4,
   UNDEFINED = 5,
   CONV=6,
   BATCHNORM=7,
   CAST=8,
   COMPARISON=9,
   EINSUM=10,
   ELU=11,
   SIGMOID=12,
   TANH=13,
   SOFTMAX=14,
   LEAKYRELU=15,
   UNARY_RECIPROCAL=16,
   UNARY_SQRT=17,
   UNARY_NEG=18,
   UNARY_EXP=19,
   UNARY_LOG=20,
   UNARY_SIN=21,
   UNARY_COS=22,
   UNARY_ABS=23,
   CLIP=24,
   NOT=25,
   UNARY_SOFTPLUS=26,
   UNARY_ATAN=27,
   UNARY_FLOOR=28,
   L2NORMALIZATION=29,
   POOL=30,
   SELU=31,
   RMSNORM=32,
   GROUPNORM=33,
   CUMSUM=34,
   SDPA=35,
   MAMBA_SCAN=36,
   RWKV_WKV6=37,
   GRIFFIN_RGLRU=38
};

enum class EFusionMappingType {
   OneToOne,
   OneToMany,
   ManyToMany,
   Reorganize,
   Shuffle,
   Unsupported
};

inline const char *toString(EFusionMappingType type)
{
   switch (type) {
      case EFusionMappingType::OneToOne:
         return "OneToOne";
      case EFusionMappingType::OneToMany:
         return "OneToMany";
      case EFusionMappingType::ManyToMany:
         return "ManyToMany";
      case EFusionMappingType::Reorganize:
         return "Reorganize";
      case EFusionMappingType::Shuffle:
         return "Shuffle";
      case EFusionMappingType::Unsupported:
         return "Unsupported";
   }
   return "Unsupported";
}

inline const char* toString(OperatorKind kind) {
   switch (kind) {
       case OperatorKind::GEMM:       return "GEMM";
       case OperatorKind::LAYERNORM:  return "LAYERNORM";
       case OperatorKind::RELU:       return "RELU";
       case OperatorKind::CONSTANT:        return "CONSTANT";
       case OperatorKind::CONSTANTOFSHAPE: return "CONSTANTOFSHAPE";
       case OperatorKind::BATCHNORM:       return "BATCHNORM";  
       case OperatorKind::CONV:       return "CONV";
       case OperatorKind::UNARY_SOFTPLUS: return "UNARY_SOFTPLUS";
       case OperatorKind::UNARY_ATAN:     return "UNARY_ATAN";
       case OperatorKind::UNARY_FLOOR:    return "UNARY_FLOOR";
       case OperatorKind::UNDEFINED:  return "UNDEFINED";
       case OperatorKind::L2NORMALIZATION: return "L2NORMALIZATION";
       default:                       return "UNKNOWN";
   }
}

/// name of the Session member holding the user-provided strides (in elements) of a graph-input tensor;
/// it is empty when the input is contiguous (Options::kStridedInput)
inline std::string InputStrideMemberName(const std::string &inputName) { return "fInputStride_" + inputName; }

inline std::set<OperatorKind> FusableKinds = { OperatorKind::RELU, OperatorKind::LAYERNORM, OperatorKind::BATCHNORM};

class ROperator {

public:
   virtual std::vector<std::string> GetBlasRoutines() { return {}; }
   virtual std::vector<std::string> GetStdLibs() { return {}; }
   virtual void Initialize(RModel&) = 0;
   virtual std::string Generate(std::string OpName) = 0;  //expect unique opName for each operator within the same RModel
   virtual std::string Generate_GPU_ALPAKA(std::string OpName){ return "";} //expect unique opName for each operator within the same RModel
   //dynParamNames: the model's shape parameters in infer-argument order
   virtual std::string Generate_GPU_ALPAKA(std::string OpName, const std::vector<std::string> & /*dynParamNames*/) {
      return Generate_GPU_ALPAKA(OpName);
   }
   // true if the operator can read its graph-input tensors through the strides given to the Session
   // (Options::kStridedInput). An operator reading a strided input without supporting it is rejected at generation.
   virtual bool SupportsStridedInput() const { return false; }
   // generate initialization code for session constructor
   virtual std::string GenerateInitCode() { return "";}
   virtual std::string GenerateInitCode_GPU_ALPAKA() { return "";};
   // generate code to reset recurrent/stateful buffers (called once per file boundary)
   virtual std::string GenerateResetStateCode_GPU_ALPAKA() { return ""; }
   virtual std::vector<std::string> GetPersistentTensorNames_GPU_ALPAKA() const { return {}; }
   // generate some specific declaration code for Session
   virtual std::string GenerateDeclCode() { return "";}
   // generate session data members specific to operator
   virtual std::string GenerateSessionMembersCode(std::string /*opName*/) { return ""; }
   virtual std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) { return ""; }
   virtual std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> & /*dynParamNames*/) {
      return Generate_GPU_Kernel_ALPAKA(opName);
   }
   virtual std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) { return ""; }
   virtual std::string Header() { return "";}
   virtual std::string GetFusableOutputTensorName() { return "";}
   virtual std::string GetBlasConfig() { return ""; }
   // most operators issue a single cuBLASLt GEMM call and so need at most one layout
   // config; operators that chain multiple GEMM calls of different shapes (e.g. a
   // low-rank factorized Gemm) override this to register one config per call.
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

   // DNNFusion-style input/output mapping classification.
   // One-To-One, One-To-Many, Many-To-Many, Reorganize, Shuffle
   virtual EFusionMappingType GetFusionMappingType() const
   {
      return IsElementwise() ? EFusionMappingType::OneToOne : EFusionMappingType::Unsupported;
   }

   // Returns the expression produced by this operator from its input
   // expressions. The default implementation adapts the existing unary
   // GetElementwiseExpr interface.
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

   bool IsOutputConstant() const { return fIsOutputConstant; }

   //virtual void Forward_reference() = 0;
   //virtual void Forward_blas() = 0;
   virtual ~ROperator(){}

   std::string fName = "UnnamedOperator";
   const std::string &Name() const { return fName; }

protected:
   OperatorKind fKind = OperatorKind::UNDEFINED;
   size_t fOpOrder = 0;
   const std::string SP = "   ";    ///< space used to correctly indent the generated C++ code
   bool fIsOutputConstant = false;  ///< flag to identify if operator has a constant output (no need to generate code)
   bool fIsOutputParamShape = false;     ///< flag to identify of the output represents a parametric shape (can be known at compile time)

   mutable std::vector<std::string_view> fInputTensorNames;
   mutable std::vector<std::string_view> fOutputTensorNames;

   std::set<std::string> fPooledOutputNames;

protected:
   bool fHasStridedInput = false; ///< the (first) input is a graph input read through user-provided strides

   /// Code defining the local array `stride_<id>` with the strides (in elements) of the graph input `name`:
   /// the ones given to the Session if any, else the contiguous ones of the current (runtime) shape.
   /// Requires a non-scalar shape.
   std::string GenerateInputStrideCode(const std::string &id, const std::string &name,
                                       const std::vector<Dim> &shape) const
   {
      return GenerateInputStrideArray("stride_" + id, name, shape);
   }

   /// same as GenerateInputStrideCode, with the array being named `var`
   std::string GenerateInputStrideArray(const std::string &var, const std::string &name,
                                        const std::vector<Dim> &shape) const
   {
      const size_t rank = shape.size();
      const std::string member = "this->" + InputStrideMemberName(name);
      std::stringstream out;
      out << SP << "size_t " << var << "[" << rank << "];\n";
      out << SP << "if (!" << member << ".empty()) {\n";
      for (size_t d = 0; d < rank; d++)
         out << SP << SP << var << "[" << d << "] = " << member << "[" << d << "];\n";
      out << SP << "} else {\n";
      out << SP << SP << var << "[" << rank - 1 << "] = 1;\n";
      for (size_t d = rank - 1; d > 0; d--)
         out << SP << SP << var << "[" << d - 1 << "] = " << var << "[" << d << "] * (" << shape[d].GetVal() << ");\n";
      out << SP << "}\n";
      return out.str();
   }

   /// CPU: code defining the lambda `xoff_<id>(logical)` giving the offset in the memory of the graph input `name`,
   /// read through the strides given to the Session, of the element with logical (contiguous row-major) index `logical`.
   /// Operators computing a contiguous index of their input can read it as tensor_X[xoff_<id>(index)].
   std::string GenerateStridedOffsetLambda(const std::string &id, const std::string &name,
                                           const std::vector<Dim> &shape) const
   {
      const size_t rank = shape.size();
      std::stringstream out;
      out << GenerateInputStrideCode(id, name, shape);
      out << SP << "const size_t shape_" << id << "[" << rank << "] = {";
      for (size_t d = 0; d < rank; d++)
         out << (d ? ", " : "") << "static_cast<size_t>(" << shape[d].GetVal() << ")";
      out << "};\n";
      out << SP << "auto xoff_" << id << " = [&](size_t logical) {\n";
      out << SP << SP << "size_t off = 0;\n";
      out << SP << SP << "for (int d = " << rank << " - 1; d >= 0; --d) {\n";
      out << SP << SP << SP << "off += (logical % shape_" << id << "[d]) * stride_" << id << "[d];\n";
      out << SP << SP << SP << "logical /= shape_" << id << "[d];\n";
      out << SP << SP << "}\n";
      out << SP << SP << "return off;\n";
      out << SP << "};\n";
      return out.str();
   }

   /// CPU: GPU kernels read a strided input X as `X[sofie_strided_offset(layoutX, logicalIndex)]`
   static std::string StridedKernelRead(const std::string &ptr, const std::string &layout, const std::string &index)
   {
      return ptr + "[sofie_strided_offset(" + layout + ", " + index + ")]";
   }

   /// Rewrites in generated code every read `name[index]` (a whole identifier `name` followed by a bracketed index) as
   /// `name[mapIndex(index)]`, e.g. to read through the strides of a tensor from the contiguous index computed by the code
   static std::string RewriteIndexedReads(std::string code, const std::string &name,
                                          const std::function<std::string(const std::string &)> &mapIndex)
   {
      const std::string token = name + "[";
      auto isIdentChar = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
      size_t pos = 0;
      while ((pos = code.find(token, pos)) != std::string::npos) {
         if (pos > 0 && isIdentChar(code[pos - 1])) {
            pos += token.size();
            continue;
         }
         size_t begin = pos + token.size(), depth = 1, end = begin;
         while (end < code.size() && depth > 0) {
            if (code[end] == '[')
               depth++;
            else if (code[end] == ']')
               depth--;
            end++;
         }
         // [begin, end - 1) is the index
         const std::string mapped = mapIndex(code.substr(begin, end - 1 - begin));
         code.replace(begin, end - 1 - begin, mapped);
         pos = begin + mapped.size();
      }
      return code;
   }

   /// A graph input of an operator whose code reads it with the contiguous (logical) index of its elements
   struct StridedInputInfo {
      std::string tensor;      ///< name of the tensor
      std::string param;       ///< name of the kernel parameter (pointer) holding it
      std::vector<Dim> shape;  ///< shape of the tensor
      bool strided = false;    ///< it is read through the strides given to the Session
   };

   /// CPU: `code` reads the strided inputs as tensor_<name>[index]; returns the code reading them through their strides
   std::string RewriteCpuStridedReads(const std::string &id, const std::vector<StridedInputInfo> &inputs,
                                      std::string code) const
   {
      std::string lambdas;
      for (const auto &in : inputs) {
         if (!in.strided)
            continue;
         lambdas += GenerateStridedOffsetLambda(id + "_" + in.param, in.tensor, in.shape);
         code = RewriteIndexedReads(code, "tensor_" + in.tensor, [&](const std::string &index) {
            return "xoff_" + id + "_" + in.param + "(" + index + ")";
         });
      }
      return lambdas + code;
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
         layouts += ",\n" + SP + SP + SP + "sofie_strided_layout<" + std::to_string(in.shape.size()) + "> const layout_" +
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

   /// CPU: elementwise Y[k] = f(X[offset(k)]) for a graph input X read through its strides, Y contiguous.
   /// `expr` maps the name of a variable holding the input element to the expression of the output element.
   std::string GenerateStridedUnaryLoop(const std::string &id, const std::string &nX, const std::string &nY,
                                        const std::vector<Dim> &shape,
                                        const std::function<std::string(const std::string &)> &expr) const
   {
      const size_t rank = shape.size();
      std::stringstream out;
      out << SP << "{\n";
      out << GenerateInputStrideCode(id, nX, shape);
      out << SP << "size_t k_" << id << " = 0;\n";
      std::string offset;
      for (size_t d = 0; d < rank; d++) {
         out << SP << "for (size_t i" << d << "_" << id << " = 0; i" << d << "_" << id << " < (" << shape[d].GetVal()
             << "); i" << d << "_" << id << "++) {\n";
         offset += (d ? " + " : "") + std::string("i") + std::to_string(d) + "_" + id + " * stride_" + id + "[" +
                   std::to_string(d) + "]";
      }
      out << SP << "size_t off_" << id << " = " << offset << ";\n";
      out << SP << "const auto x_" << id << " = tensor_" << nX << "[off_" << id << "];\n";
      out << SP << "tensor_" << nY << "[k_" << id << "++] = " << expr("x_" + id) << ";\n";
      for (size_t d = 0; d < rank; d++)
         out << SP << "}\n";
      out << SP << "}\n";
      return out.str();
   }

   /// CPU: elementwise Y[k] = f(X_0, X_1, ...) over the same shape for several inputs, those flagged in `strided` being
   /// graph inputs read through their strides (shape of the original input = `shape`), the others contiguous.
   /// `expr` maps the names of the variables holding the input elements to the expression of the output element.
   std::string GenerateStridedNaryLoop(const std::string &id, const std::vector<std::string> &names,
                                       const std::vector<bool> &strided, const std::string &nY,
                                       const std::vector<Dim> &shape,
                                       const std::function<std::string(const std::vector<std::string> &)> &expr) const
   {
      const size_t rank = shape.size();
      std::stringstream out;
      out << SP << "{\n";
      std::vector<std::string> vars;
      for (size_t n = 0; n < names.size(); n++) {
         const std::string sid = id + "_" + std::to_string(n);
         if (strided[n])
            out << GenerateInputStrideCode(sid, names[n], shape);
         vars.push_back("x_" + sid);
      }
      out << SP << "size_t k_" << id << " = 0;\n";
      std::vector<std::string> offsets(names.size());
      for (size_t d = 0; d < rank; d++) {
         out << SP << "for (size_t i" << d << "_" << id << " = 0; i" << d << "_" << id << " < (" << shape[d].GetVal()
             << "); i" << d << "_" << id << "++) {\n";
         for (size_t n = 0; n < names.size(); n++)
            if (strided[n])
               offsets[n] += (d ? " + " : "") + std::string("i") + std::to_string(d) + "_" + id + " * stride_" + id +
                             "_" + std::to_string(n) + "[" + std::to_string(d) + "]";
      }
      for (size_t n = 0; n < names.size(); n++) {
         const std::string index = strided[n] ? offsets[n] : "k_" + id;
         out << SP << "const auto " << vars[n] << " = tensor_" << names[n] << "[" << index << "];\n";
      }
      out << SP << "tensor_" << nY << "[k_" << id << "++] = " << expr(vars) << ";\n";
      for (size_t d = 0; d < rank; d++)
         out << SP << "}\n";
      out << SP << "}\n";
      return out.str();
   }

   /// GPU: source of an elementwise kernel reading its input through a strided layout (see sofie_strided_layout)
   /// `expr` maps the name of a variable holding the input element to the expression of the output element;
   /// `extraParams` declares additional kernel parameters, e.g. ", T alpha"
   std::string GenerateStridedUnaryKernel(const std::string &kernelName, const std::string &title,
                                          const std::function<std::string(const std::string &)> &expr,
                                          const std::string &extraParams = "") const
   {
      std::string op = "\n//------ " + title + "_STRIDED_KERNEL_ALPAKA\n";
      op += "struct " + kernelName + " {\n";
      op += SP + "template<typename TAcc, typename T, std::size_t R>\n";
      op += SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, "
                 "std::size_t numElements, sofie_strided_layout<R> const layout" + extraParams + ") const {\n";
      op += SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + "if (idx < numElements) {\n";
      op += SP + SP + SP + "auto const src = sofie_strided_offset(layout, idx);\n";
      op += SP + SP + SP + "T const x = data[src];\n";
      op += SP + SP + SP + "out[idx] = " + expr("x") + ";\n";
      op += SP + SP + "}\n";
      op += SP + "}\n";
      op += "};\n";
      return op;
   }

   /// GPU: launch of a strided elementwise kernel (member `kernelMember`) on the graph input nX
   /// `extraArgs` are the additional kernel arguments matching `extraParams` of the kernel, e.g. ", op_0_alpha"
   std::string GenerateStridedUnaryLaunch(const std::string &opName, const std::string &kernelMember,
                                          const std::string &title, const std::string &nX, const std::string &nY,
                                          const std::vector<Dim> &shape, const std::string &extraArgs = "") const
   {
      const size_t rank = shape.size();
      const std::string length = ConvertDimShapeToLength(shape);
      std::stringstream out;
      out << "\n//------ " << title << "_STRIDED_GPU_ALPAKA\n";
      out << GenerateInputStrideCode(opName, nX, shape);
      out << SP << "sofie_strided_layout<" << rank << "> layout_" << opName << ";\n";
      for (size_t d = 0; d < rank; d++) {
         out << SP << "layout_" << opName << ".shape[" << d << "] = static_cast<std::size_t>(" << shape[d].GetVal()
             << ");\n";
         out << SP << "layout_" << opName << ".stride[" << d << "] = stride_" << opName << "[" << d << "];\n";
      }
      out << SP << "auto const elementsPerGrid_" << opName << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName << ", "
          << kernelMember << ", alpaka::getPtrNative(deviceBuf_" << nX << "), alpaka::getPtrNative(deviceBuf_" << nY
          << "), static_cast<Idx>(" << length << "), layout_" << opName << extraArgs << ");\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   /// Operators with broadcasting read input X (original rank `origRank`, rank-padded shape `paddedShape` with leading 1s,
   /// output rank `outRank`) at the position of the output multi-index (idx_0, ..., idx_{outRank-1}).
   /// CPU: offset expression sum(idx_j * stride[k]) over the non-broadcast dims, strides in the array `strideVar`
   /// (see GenerateInputStrideCode, defined for the original shape of X).
   std::string GenerateStridedBroadcastIndex(const std::string &strideVar, const std::vector<Dim> &paddedShape,
                                             size_t origRank, size_t outRank) const
   {
      std::string expr;
      const size_t pad = paddedShape.size() - origRank;
      for (size_t i = pad; i < paddedShape.size(); i++) {
         if (paddedShape[i].GetVal() == "1")
            continue;
         expr += "idx_" + std::to_string(i + outRank - paddedShape.size()) + " * " + strideVar + "[" +
                 std::to_string(i - pad) + "] + ";
      }
      if (expr.empty())
         return "0";
      expr.erase(expr.size() - 3);
      return expr;
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
      out << SP << "sofie_strided_layout<" << outRank << "> layout_" << id << ";\n";
      for (size_t d = 0; d < outRank; d++) {
         out << SP << "layout_" << id << ".shape[" << d << "] = static_cast<std::size_t>(" << shapeY[d].GetVal() << ");\n";
         // dim d of the output is dim (d - (outRank - paddedShape.size()) - pad) of the original input
         const long k = static_cast<long>(d) - static_cast<long>(outRank - paddedShape.size()) - static_cast<long>(pad);
         if (k < 0 || paddedShape[k + pad].GetVal() == "1")
            out << SP << "layout_" << id << ".stride[" << d << "] = 0;\n";
         else
            out << SP << "layout_" << id << ".stride[" << d << "] = stride_" << id << "[" << k << "];\n";
      }
      return out.str();
   }

public:
   /// true if the operator reads a graph input through its strides: it then needs its own kernel and cannot be fused
   bool HasStridedInput() const { return fHasStridedInput; }

   void MarkOutputAsPooled(std::string_view name) { fPooledOutputNames.insert(std::string(name)); }
   bool IsOutputPooled(std::string_view name) const { return fPooledOutputNames.count(std::string(name)) > 0; }

   std::span<const std::string_view> GetOpInputTensors() const {
      return fInputTensorNames;
   }

   std::span<const std::string_view> GetOpOutputTensors() const {
      return fOutputTensorNames;
   }

   OperatorKind GetKind() const { return fKind; }

   void RegisterOperatorOrder(const size_t ord){
      fOpOrder = ord;
   }
   size_t GetOpOrder(){
      return fOpOrder;
   }

};

}

#endif //SOFIE_OPERATOR
