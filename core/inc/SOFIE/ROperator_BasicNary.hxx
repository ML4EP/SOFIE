#ifndef SOFIE_ROPERATOR_BASICNARY
#define SOFIE_ROPERATOR_BASICNARY

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <algorithm>
#include <sstream>
#include <vector>
namespace SOFIE{

enum class EBasicNaryOperator {Max, Min, Mean, Sum};

template<typename T, EBasicNaryOperator Op>
struct NaryOperatorTraits {};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Max> {
   static const std::string Name() {return "Max";}
   static std::string Expr(const std::vector<std::string> &inputs)
   {
      std::stringstream out;
      out << "std::max({ " << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << ", " << inputs[i];
      }
      out << "})";
      return out.str();
   }
   static std::string Op(const std::string &res, std::vector<std::string> &inputs)
   {
      return res + " = " + Expr(inputs) + ";\n";
   }
   static size_t Func(const std::vector<size_t> &values) { return *std::max_element(values.begin(), values.end()); }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Min> {
   static const std::string Name() {return "Min";}
   static std::string Expr(const std::vector<std::string> &inputs)
   {
      std::stringstream out;
      out << "std::min({ " << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << ", " << inputs[i];
      }
      out << "})";
      return out.str();
   }
   static std::string Op(const std::string &res, std::vector<std::string> &inputs)
   {
      return res + " = " + Expr(inputs) + ";\n";
   }
   static size_t Func(const std::vector<size_t> &values) { return *std::min_element(values.begin(), values.end()); }
};

template <typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Mean> {
   static const std::string Name() {return "Mean";}
   static std::string Expr(const std::vector<std::string> &inputs)
   {
      std::stringstream out;
      out << "((" << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << " + " << inputs[i];
      }
      out << ") / " << ConvertTypeToString(GetTemplatedType(T{})) << "(" << inputs.size() << "))";
      return out.str();
   }
   static std::string Op(const std::string &res, std::vector<std::string> &inputs)
   {
      return res + " = " + Expr(inputs) + ";\n";
   }
   static size_t Func(const std::vector<size_t> &values)
   {
      size_t sum = 0;
      for (auto &v : values)
         sum += v;
      return sum / values.size();
   }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Sum> {
   static const std::string Name() {return "Sum";}
   static std::string Expr(const std::vector<std::string> &inputs)
   {
      std::stringstream out;
      out << "(" << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << " + " << inputs[i];
      }
      out << ")";
      return out.str();
   }
   static std::string Op(const std::string &res, std::vector<std::string> &inputs)
   {
      return res + " = " + Expr(inputs) + ";\n";
   }
   static size_t Func(const std::vector<size_t> &values)
   {
      size_t sum = 0;
      for (auto &v : values)
         sum += v;
      return sum;
   }
};

template <typename T, EBasicNaryOperator Op>
class ROperator_BasicNary final : public ROperator
{

private:

   std::vector<std::string> fNInputs;
   std::string fNY;
   std::vector<std::vector<Dim>> fShapeInputs;

   std::vector<std::string> fNBroadcastedInputs;
   std::vector<size_t> fShapeY;
   std::vector<Dim> fDimShapeY;

   bool fBroadcast = false;
   std::vector<bool> fStrided; ///< inputs which are graph inputs read through their strides (Options::kStridedInput)

   std::string fType;

public:
   ROperator_BasicNary(){}

   ROperator_BasicNary( const std::vector<std::string> & inputNames, const std::string& nameY):
   fNY(UTILITY::Clean_name(nameY)){
      fNInputs.reserve(inputNames.size());
      for (auto & name : inputNames)
         fNInputs.push_back(UTILITY::Clean_name(name));

      fInputTensorNames.resize(fNInputs.size());
      std::transform(fNInputs.begin(), fNInputs.end(), fInputTensorNames.begin(),
                  [](const std::string& s) -> std::string_view { return s; });
      fOutputTensorNames = { fNY };
   }

   bool InitializeShapeTensorOutput(RModel &model)
   {
      bool hasShapeTensor = false;
      bool isScalar = true;
      size_t length = 1;
      for (auto &name : fNInputs) {
         if (model.GetTensorType(name) != ETensorType::INT64)
            return false;
         if (!model.IsShapeTensor(name) && !model.IsInitializedTensor(name))
            return false;
         hasShapeTensor |= model.IsShapeTensor(name);
         auto shape = model.GetTensorShape(name);
         if (shape.size() > 1)
            return false;
         if (!shape.empty()) {
            isScalar = false;
            if (shape[0] != 1 && length != 1 && shape[0] != length)
               return false;
            length = std::max(length, shape[0]);
         }
      }
      if (!hasShapeTensor)
         return false;

      std::vector<std::vector<Dim>> values(fNInputs.size(), std::vector<Dim>(length));
      for (size_t i = 0; i < fNInputs.size(); i++) {
         auto &name = fNInputs[i];
         if (model.IsShapeTensor(name)) {
            auto &dims = model.GetShapeTensorValues(name);
            for (size_t j = 0; j < length; j++)
               values[i][j] = (dims.size() == 1) ? dims[0] : dims[j];
         } else {
            auto data = static_cast<int64_t *>(model.GetInitializedTensorData(name).get());
            size_t n = ConvertShapeToLength(model.GetTensorShape(name));
            for (size_t j = 0; j < length; j++)
               values[i][j] = Dim{static_cast<size_t>(data[(n == 1) ? 0 : j])};
         }
      }

      std::vector<Dim> outputValues(length);
      for (size_t j = 0; j < length; j++) {
         bool isConstant = true;
         std::vector<size_t> dims(fNInputs.size());
         std::vector<std::string> exprs(fNInputs.size());
         for (size_t i = 0; i < fNInputs.size(); i++) {
            isConstant &= !values[i][j].isParam;
            dims[i] = values[i][j].dim;
            exprs[i] = "size_t(" + values[i][j].GetVal() + ")";
         }
         if (isConstant)
            outputValues[j] = Dim{NaryOperatorTraits<T, Op>::Func(dims)};
         else
            outputValues[j] = Dim{NaryOperatorTraits<T, Op>::Expr(exprs), static_cast<size_t>(-1)};
      }
      model.AddShapeTensor(fNY, outputValues, isScalar);
      fIsOutputConstant = true;
      if (model.Verbose()) {
         std::cout << NaryOperatorTraits<T, Op>::Name() << " : --> " << fNY << " "
                   << ConvertDimShapeToString(outputValues) << " (shape)" << std::endl;
      }
      return true;
   }

   void Initialize(RModel& model) override {
      std::vector<std::vector<size_t>> inputShapes;
      for (auto &it : fNInputs) {
         if (!model.CheckIfTensorAlreadyExist(it)) {
            throw std::runtime_error("SOFIE BasicNary Op Input Tensor " + it + " is not found in model");
         }
      }
      if (InitializeShapeTensorOutput(model))
         return;
      for (auto &it : fNInputs) {
         fShapeInputs.push_back(model.GetDimTensorShape(it));
         fStrided.push_back(model.IsStridedInputTensor(it) && !fShapeInputs.back().empty());
         fHasStridedInput |= fStrided.back();
         if (fNInputs.size()> 2) {
            if (model.IsDimInputTensor(it))
               throw std::runtime_error("SOFIE BasicNary : supports only 2 inputs for dynamic tensors");
            else
               inputShapes.push_back(model.GetTensorShape(it));
         }
      }
      if (fShapeInputs.size() > 2 ) {
         auto shapeY = UTILITY::MultidirectionalBroadcastShape(inputShapes);
         fDimShapeY = ConvertShapeToDim(shapeY);
      } else if (fShapeInputs.size() == 2 ) {
         auto ret  = UTILITY::MultidirectionalBroadcastShape(fShapeInputs[0], fShapeInputs[1]);
         fBroadcast = ret.first;
         fDimShapeY = ret.second;
         if (ret.first & 4) {
            auto IsInputDimParam = [&](const std::string &p) {
               auto inputNames = model.GetInputTensorNames();
               for (auto &input : inputNames) {
                  for (auto &i_s : model.GetDimTensorShape(input)) {
                     if (i_s.isParam && i_s.param == p)
                        return true;
                  }
               }
               return false;
            };
            auto & shapeA = fShapeInputs[0];
            auto & shapeB = fShapeInputs[1];
            for (size_t i = 0; i < fDimShapeY.size(); i++) {
               auto &s = fDimShapeY[i];
               if (s.isParam && s.param.find("std::max") != std::string::npos) {
                  if (IsInputDimParam(shapeA[i].param)) {
                     if (shapeA[i].dim != 1)
                        s = shapeA[i];
                     else
                        s = shapeB[i];
                  } else if (IsInputDimParam(shapeB[i].param)) {
                     if (shapeB[i].dim != 1)
                        s = shapeB[i];
                     else
                        s = shapeA[i];
                  }
               }
            }
         }
      } else if  (fShapeInputs.size() == 1 ) {
         fDimShapeY = fShapeInputs[0];
      }
      if (!fShapeY.empty())
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNInputs[0]), fShapeY);
      else
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNInputs[0]), fDimShapeY);


      fType = ConvertTypeToString(model.GetTensorType(fNInputs[0]));

      if (model.Verbose()) {
         std::cout << NaryOperatorTraits<T, Op>::Name() << " : ";
         if (fNInputs.size() == 2)
            std::cout << ConvertDimShapeToString(fShapeInputs[0]) << " , "
                      << ConvertDimShapeToString(fShapeInputs[1]);
         std::cout << " --> " << ConvertDimShapeToString(fDimShapeY) << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant)
         return "";
      OpName = "op_" + OpName;
      if (fDimShapeY.empty()) {
         throw std::runtime_error("SOFIE BasicNary called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fDimShapeY);
      out << SP << "\n//------ BasicNary operator\n";

      int nInputs = fNInputs.size();

      if (nInputs == 1 && fStrided[0]) {
         out << GenerateStridedUnaryLoop(OpName, fNInputs[0], fNY, fShapeInputs[0], [](const std::string &v) { return v; });
      } else if (nInputs == 1) {
         out << SP << "std::copy(tensor_" << fNInputs[0] << ", tensor_" << fNInputs[0] << " + ";
         out << length << ", tensor_" << fNY << ");\n";
      } else {

         std::vector<std::vector<Dim>> inputStrides(nInputs);
         for (int i = 0; i < nInputs; i++)
            inputStrides[i] = UTILITY::ComputeStrideFromShape(fShapeInputs[i]);

         auto stridesY = UTILITY::ComputeStrideFromShape(fDimShapeY);

         // inputs read through the strides given to the Session
         for (int i = 0; i < nInputs; i++)
            if (fStrided[i])
               out << GenerateInputStrideCode(OpName + "_" + std::to_string(i), fNInputs[i], fShapeInputs[i]);

         std::string compute_idx_Y;
         int nloop = 0;
         if (fDimShapeY.empty() ||
               std::all_of(fDimShapeY.begin(), fDimShapeY.end(), [](Dim d) { return d.dim == 1 || d.GetVal() == "1"; })) {
            compute_idx_Y = "0";
         } else {
            for (size_t i = 0; i < fDimShapeY.size(); ++i) {
               if (fDimShapeY[i].dim != 1 && fDimShapeY[i].GetVal() != "1") {
                  nloop++;
                  for (int j = 0; j < nloop; j++) out << SP;
                  out << "for (size_t idx_" << i << " = 0; idx_" << i << " < " << fDimShapeY[i]
                      << "; ++idx_" << i << "){\n";
                  compute_idx_Y += "idx_" + std::to_string(i);
                  if (stridesY[i].GetVal() != "1")
                     compute_idx_Y += " * " + stridesY[i].GetVal();
                  compute_idx_Y += " + ";
               }
            }
            for (int j = 0; j < 3; j++)
               compute_idx_Y.pop_back();
         }
         std::vector<std::string> inputs(nInputs);
         for (int ipt = 0; ipt < nInputs; ipt++ ) {
            std::string compute_idx_X;
            auto & shape = fShapeInputs[ipt];
            auto & stride = inputStrides[ipt];
            if (shape.empty() ||
                std::all_of(shape.begin(), shape.end(), [](Dim d) { return d.dim == 1 || d.GetVal() == "1"; })) {
               compute_idx_X = "0";
            } else {
               for (size_t i = 0; i < shape.size(); ++i) {
                  if (shape[i].dim == 1 || shape[i].GetVal() == "1")
                     continue;
                  compute_idx_X += "idx_" + std::to_string(i + (fDimShapeY.size() - shape.size()));
                  if (stride[i].GetVal() != "1")
                     compute_idx_X += " * " + stride[i].GetVal();
                  compute_idx_X += " + ";
               }
               for (int j = 0; j < 3; j++)
                  compute_idx_X.pop_back();
            }
            if (fStrided[ipt] && compute_idx_X != "0")
               compute_idx_X = GenerateStridedBroadcastIndex("stride_" + OpName + "_" + std::to_string(ipt), shape,
                                                             shape.size(), fDimShapeY.size());
            inputs[ipt] = "tensor_" + fNInputs[ipt] + "[" + compute_idx_X + "]";
         }

         for (int j = 0; j < nloop + 1; j++) out << SP;
         std::string output = "tensor_" + fNY + "[" + compute_idx_Y + "]";
         out << NaryOperatorTraits<T,Op>::Op(output, inputs);

         for (int i = nloop; i > 0; i--) {
            for (int j = 0; j < i; j++) out << SP;
            out << "}\n";
         }
      }
      return out.str();
   }

   bool SupportsStridedInput() const override { return true; }

   std::vector<std::string> GetStdLibs() override {return { std::string("cmath") }; }

   std::string GetGPUCombine(const std::string& acc_v, const std::string& val) const {
      if (Op == EBasicNaryOperator::Max)
         return acc_v + " = (" + acc_v + " > " + val + ") ? " + acc_v + " : " + val + ";";
      if (Op == EBasicNaryOperator::Min)
         return acc_v + " = (" + acc_v + " < " + val + ") ? " + acc_v + " : " + val + ";";
      return acc_v + " = " + acc_v + " + " + val + ";"; // Sum and Mean both accumulate
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string OpName, const std::vector<std::string> &dynParamNames) override {
      if (fIsOutputConstant) return "";
      OpName = "op_" + OpName;
      size_t nIn = fNInputs.size();
      const std::size_t D = fDimShapeY.size();
      std::vector<std::vector<Dim>> fPaddedInputs = fShapeInputs;
      for (auto &ps : fPaddedInputs)
         if (ps.size() < D)
            ps.insert(ps.begin(), D - ps.size(), Dim{1});
      auto stridesY = UTILITY::ComputeStrideFromShape(fDimShapeY);
      auto isOne = [](const Dim &d) { return d.GetVal() == "1"; };
      std::vector<bool> isScalar(nIn), isContiguous(nIn), isPartial(nIn);
      std::vector<std::vector<Dim>> strides(nIn);
      bool anyPartial = false;
      for (size_t i = 0; i < nIn; i++) {
         isScalar[i]     = ConvertDimShapeToLength(fPaddedInputs[i]) == "1";
         isContiguous[i] = UTILITY::AreSameShape(fPaddedInputs[i], fDimShapeY);
         isPartial[i]    = !isScalar[i] && !isContiguous[i];
         if (isPartial[i]) {
            strides[i] = UTILITY::ComputeStrideFromShape(fPaddedInputs[i]);
            anyPartial = true;
         }
      }
      // inputs read through the strides given to the Session (a scalar input has no strides to apply)
      std::vector<bool> strided(nIn);
      for (size_t i = 0; i < nIn; i++)
         strided[i] = fStrided[i] && !isScalar[i];

      std::string op;
      op += "\n//------ BASICNARY_KERNEL_ALPAKA\n";
      op += SP + "struct BasicNaryKernel_" + OpName + " {\n";
      op += SP + SP + "template<typename TAcc, typename TOut";
      for (size_t i = 0; i < nIn; i++)
         op += ", typename Tin" + std::to_string(i);
      op += ">\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc";
      for (size_t i = 0; i < nIn; i++)
         op += ", Tin" + std::to_string(i) + " const* in" + std::to_string(i);
      op += ", TOut* out";
      for (auto &p : dynParamNames)
         op += ", std::size_t const " + p;
      for (size_t i = 0; i < nIn; i++)
         if (strided[i])
            op += ", sofie_strided_layout<" + std::to_string(D) + "> const layout" + std::to_string(i);
      op += ", std::size_t n) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= n) return;\n";

      // decompose idx into output coords, skipping broadcast dims (stride 0)
      if (anyPartial) {
         op += SP + SP + SP + "std::size_t remaining = idx;\n";
         op += SP + SP + SP + "std::size_t coord;\n";
         for (size_t i = 0; i < nIn; i++)
            if (isPartial[i])
               op += SP + SP + SP + "std::size_t idx" + std::to_string(i) + " = 0;\n";
         for (std::size_t d = 0; d < D; d++) {
            std::string sY = "(" + stridesY[d].GetVal() + ")";
            op += SP + SP + SP + "coord = remaining / " + sY + ";\n";
            if (d + 1 < D)
               op += SP + SP + SP + "remaining -= coord * " + sY + ";\n";
            for (size_t i = 0; i < nIn; i++) {
               if (!isPartial[i] || isOne(fPaddedInputs[i][d])) continue;
               op += SP + SP + SP + "idx" + std::to_string(i) + " += coord * (" + strides[i][d].GetVal() + ");\n";
            }
         }
      }
      auto index = [&](size_t i) -> std::string {
         if (strided[i]) return "sofie_strided_offset(layout" + std::to_string(i) + ", idx)";
         if (isContiguous[i]) return "idx";
         if (isScalar[i]) return "0";
         return "idx" + std::to_string(i);
      };
      op += SP + SP + SP + "TOut v = static_cast<TOut>(in0[" + index(0) + "]);\n";
      for (size_t i = 1; i < nIn; i++)
         op += SP + SP + SP + "{ TOut w = static_cast<TOut>(in" + std::to_string(i) + "[" + index(i) + "]); " + GetGPUCombine("v", "w") + " }\n";
      if (Op == EBasicNaryOperator::Mean)
         op += SP + SP + SP + "v = v / static_cast<TOut>(" + std::to_string(nIn) + ");\n";
      op += SP + SP + SP + "out[idx] = v;\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string OpName) override {
      if (fIsOutputConstant) return "";
      OpName = "op_" + OpName;
      return SP + "BasicNaryKernel_" + OpName + " basicNaryKernel_" + OpName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName, const std::vector<std::string> &dynParamNames) override {
      if (fIsOutputConstant) return "";
      if (fDimShapeY.empty())
         throw std::runtime_error("SOFIE BasicNary Op called to Generate without being initialized first");
      OpName = "op_" + OpName;
      std::stringstream out;
      std::string length = ConvertDimShapeToLength(fDimShapeY);
      out << "\n//------ BASICNARY_GPU_ALPAKA\n";
      std::vector<bool> strided(fNInputs.size());
      for (size_t i = 0; i < fNInputs.size(); i++) {
         strided[i] = fStrided[i] && ConvertDimShapeToLength(fShapeInputs[i]) != "1";
         if (!strided[i])
            continue;
         std::vector<Dim> padded = fShapeInputs[i];
         if (padded.size() < fDimShapeY.size())
            padded.insert(padded.begin(), fDimShapeY.size() - padded.size(), Dim{1});
         out << GenerateStridedBroadcastLayout(OpName + "_" + std::to_string(i), fNInputs[i], padded,
                                               fShapeInputs[i].size(), fDimShapeY);
      }
      out << SP << "auto const elementsPerGrid_" << OpName << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << OpName << " = sofie_workdiv(elementsPerGrid_" << OpName << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << OpName
          << ", basicNaryKernel_" << OpName;
      for (auto &in : fNInputs)
         out << ", alpaka::getPtrNative(deviceBuf_" << in << ")";
      out << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")";
      for (auto &p : dynParamNames)
         out << ", static_cast<std::size_t>(" << p << ")";
      for (size_t i = 0; i < fNInputs.size(); i++)
         if (strided[i])
            out << ", layout_" << OpName << "_" << i;
      out << ", static_cast<std::size_t>(" << length << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }
};

}//SOFIE


#endif //SOFIE_ROPERATOR_BasicNary
