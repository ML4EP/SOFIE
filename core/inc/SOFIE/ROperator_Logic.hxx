#ifndef SOFIE_ROPERATOR_LOGIC
#define SOFIE_ROPERATOR_LOGIC

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SOFIE {

enum class ELogicBinaryOp {
   // logical (bool / uint8)
   And,
   Or,
   Xor,
   // bitwise (integer types)
   BitwiseAnd,
   BitwiseOr,
   BitwiseXor,
};

template <typename T, ELogicBinaryOp Op>
struct LogicBinaryTrait {};

template <typename T>
struct LogicBinaryTrait<T, ELogicBinaryOp::And> {
   static std::string Name()   { return "And"; }
   static std::string KernelName() { return "AndKernel"; }
   static std::string Expr(const std::string &a, const std::string &b) {
      return "(" + a + " && " + b + ")";
   }
   static T Eval(T a, T b) { return static_cast<T>(a && b); }
   static ETensorType OutputType() { return ETensorType::BOOL; }
};

template <typename T>
struct LogicBinaryTrait<T, ELogicBinaryOp::Or> {
   static std::string Name()   { return "Or"; }
   static std::string KernelName() { return "OrKernel"; }
   static std::string Expr(const std::string &a, const std::string &b) {
      return "(" + a + " || " + b + ")";
   }
   static T Eval(T a, T b) { return static_cast<T>(a || b); }
   static ETensorType OutputType() { return ETensorType::BOOL; }
};

template <typename T>
struct LogicBinaryTrait<T, ELogicBinaryOp::Xor> {
   static std::string Name()   { return "Xor"; }
   static std::string KernelName() { return "XorKernel"; }
   static std::string Expr(const std::string &a, const std::string &b) {
      return "(" + a + " != " + b + ")";
   }
   static T Eval(T a, T b) { return static_cast<T>(a != b); }
   static ETensorType OutputType() { return ETensorType::BOOL; }
};

template <typename T>
struct LogicBinaryTrait<T, ELogicBinaryOp::BitwiseAnd> {
   static std::string Name()   { return "BitwiseAnd"; }
   static std::string KernelName() { return "BitwiseAndKernel"; }
   static std::string Expr(const std::string &a, const std::string &b) {
      return "(" + a + " & " + b + ")";
   }
   static T Eval(T a, T b) { return static_cast<T>(a & b); }
   static ETensorType OutputType() { return GetTemplatedType(T()); }
};

template <typename T>
struct LogicBinaryTrait<T, ELogicBinaryOp::BitwiseOr> {
   static std::string Name()   { return "BitwiseOr"; }
   static std::string KernelName() { return "BitwiseOrKernel"; }
   static std::string Expr(const std::string &a, const std::string &b) {
      return "(" + a + " | " + b + ")";
   }
   static T Eval(T a, T b) { return static_cast<T>(a | b); }
   static ETensorType OutputType() { return GetTemplatedType(T()); }
};

template <typename T>
struct LogicBinaryTrait<T, ELogicBinaryOp::BitwiseXor> {
   static std::string Name()   { return "BitwiseXor"; }
   static std::string KernelName() { return "BitwiseXorKernel"; }
   static std::string Expr(const std::string &a, const std::string &b) {
      return "(" + a + " ^ " + b + ")";
   }
   static T Eval(T a, T b) { return static_cast<T>(a ^ b); }
   static ETensorType OutputType() { return GetTemplatedType(T()); }
};

template <typename T, ELogicBinaryOp Op>
class ROperator_LogicBinary final : public ROperator {
private:
   std::string fNA;
   std::string fNB;
   std::string fNY;
   std::vector<Dim> fShape;
   bool fStridedA = false; ///< input A is a graph input read through its strides (Options::kStridedInput)
   bool fStridedB = false;

   using Trait = LogicBinaryTrait<T, Op>;

public:
   ROperator_LogicBinary() {}

   ROperator_LogicBinary(std::string nameA, std::string nameB, std::string nameY)
      : fNA(UTILITY::Clean_name(nameA)),
        fNB(UTILITY::Clean_name(nameB)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames  = { fNA, fNB };
      fOutputTensorNames = { fNY };
   }

   // ── Type / shape inference ────────────────────────────────────────────────

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNA))
         throw std::runtime_error("SOFIE " + Trait::Name() + ": input A '" +
                                  fNA + "' not found in model");
      if (!model.CheckIfTensorAlreadyExist(fNB))
         throw std::runtime_error("SOFIE " + Trait::Name() + ": input B '" +
                                  fNB + "' not found in model");

      fShape = model.GetDimTensorShape(fNA);
      fStridedA = model.IsStridedInputTensor(fNA) && !fShape.empty();
      fStridedB = model.IsStridedInputTensor(fNB) && !fShape.empty();
      fHasStridedInput = fStridedA || fStridedB;
      auto length = ConvertShapeToLength(fShape);
      // Constant-fold: if both inputs are constant, compute output at init time.
      if (model.IsConstantTensor(fNA) && model.IsConstantTensor(fNB)) {
         auto dataA  = static_cast<T*>(model.GetInitializedTensorData(fNA).get());
         auto dataB  = static_cast<T*>(model.GetInitializedTensorData(fNB).get());
         std::vector<T> dataY(length);
         for (size_t i = 0; i < length; ++i)
            dataY[i] = Trait::Eval(dataA[i], dataB[i]);
         std::vector<size_t> outShape = (length == 1) ?
            std::vector<size_t>{} : std::vector<size_t>{ length };
         model.AddConstantTensor<T>(fNY, outShape, dataY.data());
         fIsOutputConstant = true;
      } else {
         model.AddIntermediateTensor(fNY, Trait::OutputType(), fShape);
      }

      if (model.Verbose()) {
         std::cout << Trait::Name() << " : " << fNA << " , " << fNB
                   << " -> " << fNY << " " << ConvertDimShapeToString(fShape)
                   << (fIsOutputConstant ? " [constant-folded]" : "") << std::endl;
      }
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant) return "";
      OpName = "op_" + OpName;
      auto length = ConvertDimShapeToLength(fShape);
      std::stringstream out;
      out << "\n//------ " << Trait::Name() << "\n";
      if (fHasStridedInput) {
         out << GenerateStridedNaryLoop(OpName, {fNA, fNB}, {fStridedA, fStridedB}, fNY, fShape,
                                        [](const std::vector<std::string> &v) { return Trait::Expr(v[0], v[1]); });
         return out.str();
      }
      out << SP << "for (std::size_t id = 0; id < " << length << "u; ++id) {\n";
      out << SP << SP << "tensor_" << fNY << "[id] = "
          << Trait::Expr("tensor_" + fNA + "[id]", "tensor_" + fNB + "[id]")
          << ";\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string OpName) override {
      if (fIsOutputConstant) return "";
      OpName = "op_" + OpName;
      std::stringstream op;
      op << "\n//------ " << Trait::Name() << "_KERNEL_ALPAKA\n";
      op << "struct " << Trait::KernelName() << "_" << OpName << " {\n";
      const std::string layoutType = "sofie_strided_layout<" + std::to_string(fShape.size()) + ">";
      op << SP << "template<typename TAcc, typename T>\n";
      op << SP << "ALPAKA_FN_ACC void operator()("
               << "TAcc const& acc, "
               << "T const* __restrict__ A, "
               << "T const* __restrict__ B, "
               << "T* __restrict__ C, "
               << "std::size_t const N"
               << (fStridedA ? ", " + layoutType + " const layoutA" : "")
               << (fStridedB ? ", " + layoutType + " const layoutB" : "") << ") const {\n";
      op << SP << SP << "auto const idx = "
               << "alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op << SP << SP << "if (idx >= N) return;\n";
      op << SP << SP << "C[idx] = "
         << Trait::Expr(fStridedA ? "A[sofie_strided_offset(layoutA, idx)]" : "A[idx]",
                        fStridedB ? "B[sofie_strided_offset(layoutB, idx)]" : "B[idx]")
         << ";\n";
      op << SP << "}\n";
      op << "};\n";
      return op.str();
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string OpName) override {
      if (fIsOutputConstant) return "";
      std::string clean = "op_" + OpName;
      return SP + Trait::KernelName() + "_" + clean + " logic_" + clean + "Kernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      if (fIsOutputConstant) return "";
      std::string cleanOp = "op_" + OpName;
      auto length = ConvertDimShapeToLength(fShape);
      std::stringstream out;
      out << "\n//------ " << Trait::Name() << "_GPU_ALPAKA\n";
      if (fStridedA)
         out << GenerateStridedBroadcastLayout(cleanOp + "_A", fNA, fShape, fShape.size(), fShape);
      if (fStridedB)
         out << GenerateStridedBroadcastLayout(cleanOp + "_B", fNB, fShape, fShape.size(), fShape);
      out << SP << "auto const elementsPerThread_" << fNY
          << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_" << fNY
          << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << fNY
          << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << SP << "auto task_" << cleanOp
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
          << ", logic_" << cleanOp << "Kernel"
          << ", alpaka::getPtrNative(deviceBuf_" << fNA << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNB << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", static_cast<Idx>(" << length << ")"
          << (fStridedA ? ", layout_" + cleanOp + "_A" : "")
          << (fStridedB ? ", layout_" + cleanOp + "_B" : "") << ");\n";
      out << SP << "alpaka::enqueue(queue, task_" << cleanOp << ");\n";
      return out.str();
   }

   bool IsElementwise() const override
   {
      return !fIsOutputConstant && !fHasStridedInput;
   }

   EFusionMappingType GetFusionMappingType() const override
   {
      return (fIsOutputConstant || fHasStridedInput) ? EFusionMappingType::Unsupported : EFusionMappingType::OneToOne;
   }

   bool SupportsFusionTypes(const std::vector<ETensorType> &inputTypes, ETensorType outputType) const override
   {
      if (inputTypes.size() != 2)
         return false;

      if constexpr (Op == ELogicBinaryOp::And ||
                    Op == ELogicBinaryOp::Or ||
                    Op == ELogicBinaryOp::Xor) {
         return inputTypes[0] == ETensorType::BOOL &&
                inputTypes[1] == ETensorType::BOOL &&
                outputType == ETensorType::BOOL;
                    } else {
                       const auto type = GetTemplatedType(T{});

                       return inputTypes[0] == type &&
                              inputTypes[1] == type &&
                              outputType == type;
                    }
   }

   std::string GetFusionExpr(const std::vector<std::string> &inputs) const override
   {
      if (fIsOutputConstant || inputs.size() != 2)
         return "";

      return Trait::Expr(inputs[0], inputs[1]);
   }
};

template <typename T>
class ROperator_BitwiseNot final : public ROperator {
private:
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_BitwiseNot() {}

   ROperator_BitwiseNot(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames  = { fNX };
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE BitwiseNot: input tensor '" + fNX +
                                  "' not found in model");
      fShape = model.GetDimTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShape.empty();
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      if (model.Verbose())
         std::cout << "BitwiseNot: " << fNX << " -> " << fNY
                   << " " << ConvertDimShapeToString(fShape) << std::endl;
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      auto length = ConvertDimShapeToLength(fShape);
      std::stringstream out;
      out << "\n//------ BITWISE_NOT\n";
      if (fHasStridedInput) {
         out << GenerateStridedUnaryLoop(OpName, fNX, fNY, fShape, [](const std::string &v) { return "~" + v; });
         return out.str();
      }
      out << SP << "for (std::size_t id = 0; id < " << length << "u; ++id) {\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ~tensor_" << fNX << "[id];\n";
      out << SP << "}\n";
      return out.str();
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate_GPU_Kernel_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fHasStridedInput)
         return GenerateStridedUnaryKernel("BitwiseNotKernel_" + OpName, "BITWISE_NOT",
                                           [](const std::string &v) { return "~" + v; });
      std::stringstream op;
      op << "\n//------ BITWISE_NOT_KERNEL_ALPAKA\n";
      op << "struct BitwiseNotKernel_" << OpName << " {\n";
      op << SP << "template<typename TAcc, typename T>\n";
      op << SP << "ALPAKA_FN_ACC void operator()("
               << "TAcc const& acc, "
               << "T const* __restrict__ input, "
               << "T* __restrict__ output, "
               << "std::size_t const N) const {\n";
      op << SP << SP << "auto const idx = "
               << "alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op << SP << SP << "if (idx >= N) return;\n";
      op << SP << SP << "output[idx] = ~input[idx];\n";
      op << SP << "}\n";
      op << "};\n";
      return op.str();
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string OpName) override {
      std::string clean = "op_" + OpName;
      return SP + "BitwiseNotKernel_" + clean + " bitwiseNotKernel_" + clean + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      std::string cleanOp = "op_" + OpName;
      if (fHasStridedInput)
         return GenerateStridedUnaryLaunch(cleanOp, "bitwiseNotKernel_" + cleanOp, "BITWISE_NOT", fNX, fNY, fShape);
      auto length = ConvertDimShapeToLength(fShape);
      std::stringstream out;
      out << "\n//------ BITWISE_NOT_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNY
          << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_" << fNY
          << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << fNY
          << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << SP << "auto task_" << cleanOp
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
          << ", bitwiseNotKernel_" << cleanOp
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", static_cast<Idx>(" << length << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << cleanOp << ");\n";
      return out.str();
   }

   bool IsElementwise() const override { return !fHasStridedInput; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return "~" + v;
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_LOGIC
