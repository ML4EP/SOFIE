#ifndef SOFIE_ROPERATOR
#define SOFIE_ROPERATOR

#include "SOFIE/utils/SOFIE_common.hxx"

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
/// it is empty when the input is contiguous
inline std::string InputStrideMemberName(const std::string &inputName) { return "fInputStride_" + inputName; }

class ROperator {

public:
   virtual std::vector<std::string> GetBlasRoutines() { return {}; }
   virtual std::vector<std::string> GetStdLibs() { return {}; }
   virtual std::vector<std::string> GetHelperFunctions() { return {}; }
   virtual void Initialize(RModel&) = 0;
   virtual std::string Generate(std::string OpName) = 0;  //expect unique opName for each operator within the same RModel
   // true if the operator can read its graph-input tensors through the strides given to the Session
   //. An operator reading a strided input without supporting it is rejected at generation.
   virtual bool SupportsStridedInput() const { return false; }
   // generate initialization code for session constructor
   virtual std::string GenerateInitCode() { return "";}
   // generate some specific declaration code for Session
   virtual std::string GenerateDeclCode() { return "";}
   // generate session data members specific to operator
   virtual std::string GenerateSessionMembersCode(std::string /*opName*/) { return ""; }
   virtual std::string Header() { return "";}

   bool IsOutputConstant() const { return fIsOutputConstant; }

   //virtual void Forward_reference() = 0;
   //virtual void Forward_blas() = 0;
   virtual ~ROperator(){}

   std::string fName = "UnnamedOperator";
   const std::string &Name() const { return fName; }

protected:
   OperatorKind fKind = OperatorKind::UNDEFINED;
   size_t fOpOrder = 0;
   bool fIsOutputConstant = false;  ///< flag to identify if operator has a constant output (no need to generate code)
   bool fIsOutputParamShape = false;     ///< flag to identify of the output represents a parametric shape (can be known at compile time)

   mutable std::vector<std::string_view> fInputTensorNames;
   mutable std::vector<std::string_view> fOutputTensorNames;


protected:
   bool fHasStridedInput = false;

   /// Code defining the local array `stride_<id>` with the strides of the graph input `name`:
   /// the ones given to the Session if any, else the contiguous ones of the current shape.
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
      out << "size_t " << var << "[" << rank << "];\n";
      out << "if (!" << member << ".empty()) {\n";
      for (size_t d = 0; d < rank; d++)
         out << var << "[" << d << "] = " << member << "[" << d << "];\n";
      out << "} else {\n";
      out << var << "[" << rank - 1 << "] = 1;\n";
      for (size_t d = rank - 1; d > 0; d--)
         out << var << "[" << d - 1 << "] = " << var << "[" << d << "] * (" << shape[d].GetVal() << ");\n";
      out << "}\n";
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
      out << "const size_t shape_" << id << "[" << rank << "] = {";
      for (size_t d = 0; d < rank; d++)
         out << (d ? ", " : "") << "static_cast<size_t>(" << shape[d].GetVal() << ")";
      out << "};\n";
      out << "auto xoff_" << id << " = [&](size_t logical) {\n";
      out << "size_t off = 0;\n";
      out << "for (int d = " << rank << " - 1; d >= 0; --d) {\n";
      out << "off += (logical % shape_" << id << "[d]) * stride_" << id << "[d];\n";
      out << "logical /= shape_" << id << "[d];\n";
      out << "}\n";
      out << "return off;\n";
      out << "};\n";
      return out.str();
   }

   /// Rewrites in generated code every read `name[index]` (a whole identifier `name` followed by a bracketed index) as
   /// `name[mapIndex(index)]`
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

   /// A graph input of an operator whose code reads it with the contiguous index of its elements
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

   /// CPU: elementwise Y[k] = f(X[offset(k)]) for a graph input X read through its strides, Y contiguous.
   /// `expr` maps the name of a variable holding the input element to the expression of the output element.
   std::string GenerateStridedUnaryLoop(const std::string &id, const std::string &nX, const std::string &nY,
                                        const std::vector<Dim> &shape,
                                        const std::function<std::string(const std::string &)> &expr) const
   {
      const size_t rank = shape.size();
      std::stringstream out;
      out << "{\n";
      out << GenerateInputStrideCode(id, nX, shape);
      out << "size_t k_" << id << " = 0;\n";
      std::string offset;
      for (size_t d = 0; d < rank; d++) {
         out << "for (size_t i" << d << "_" << id << " = 0; i" << d << "_" << id << " < (" << shape[d].GetVal()
             << "); i" << d << "_" << id << "++) {\n";
         offset += (d ? " + " : "") + std::string("i") + std::to_string(d) + "_" + id + " * stride_" + id + "[" +
                   std::to_string(d) + "]";
      }
      out << "size_t off_" << id << " = " << offset << ";\n";
      out << "const auto x_" << id << " = tensor_" << nX << "[off_" << id << "];\n";
      out << "tensor_" << nY << "[k_" << id << "++] = " << expr("x_" + id) << ";\n";
      for (size_t d = 0; d < rank; d++)
         out << "}\n";
      out << "}\n";
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
      out << "{\n";
      std::vector<std::string> vars;
      for (size_t n = 0; n < names.size(); n++) {
         const std::string sid = id + "_" + std::to_string(n);
         if (strided[n])
            out << GenerateInputStrideCode(sid, names[n], shape);
         vars.push_back("x_" + sid);
      }
      out << "size_t k_" << id << " = 0;\n";
      std::vector<std::string> offsets(names.size());
      for (size_t d = 0; d < rank; d++) {
         out << "for (size_t i" << d << "_" << id << " = 0; i" << d << "_" << id << " < (" << shape[d].GetVal()
             << "); i" << d << "_" << id << "++) {\n";
         for (size_t n = 0; n < names.size(); n++)
            if (strided[n])
               offsets[n] += (d ? " + " : "") + std::string("i") + std::to_string(d) + "_" + id + " * stride_" + id +
                             "_" + std::to_string(n) + "[" + std::to_string(d) + "]";
      }
      for (size_t n = 0; n < names.size(); n++) {
         const std::string index = strided[n] ? offsets[n] : "k_" + id;
         out << "const auto " << vars[n] << " = tensor_" << names[n] << "[" << index << "];\n";
      }
      out << "tensor_" << nY << "[k_" << id << "++] = " << expr(vars) << ";\n";
      for (size_t d = 0; d < rank; d++)
         out << "}\n";
      out << "}\n";
      return out.str();
   }

   /// Operators with broadcasting read input X (original rank `origRank`, rank-padded shape `paddedShape` with leading 1s,
   /// output rank `outRank`) at the position of the output multi-index (idx_0, ..., idx_{outRank-1}).
   /// CPU: offset expression sum(idx_j * stride[k]) over the non-broadcast dims, strides in the array `strideVar`
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


public:
   /// true if the operator reads a graph input through its strides: it then needs its own kernel and cannot be fused
   bool HasStridedInput() const { return fHasStridedInput; }


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
