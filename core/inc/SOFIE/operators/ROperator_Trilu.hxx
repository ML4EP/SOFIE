#ifndef SOFIE_ROPERATOR_TRILU
#define SOFIE_ROPERATOR_TRILU

#include "SOFIE/utils/SOFIE_common.hxx"
#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/model/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace SOFIE {

template <typename T>
class ROperator_Trilu final : public ROperator_ALPAKA {
private:
   int      fUpper = 1;

   int64_t  fK        = 0;
   bool     fKIsStatic= true;

   std::string fNX;
   std::string fNK;
   std::string fNY;

   std::vector<Dim> fShape;
   Dim fM;
   Dim fN;
   std::string fTotal;  // ConvertDimShapeToLength(fShape)

public:
   ROperator_Trilu() {}

   ROperator_Trilu(int upper, std::string nameX, std::string nameY)
      : fUpper(upper),
        fNX(UTILITY::Clean_name(nameX)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames  = { fNX };
      fOutputTensorNames = { fNY };
   }

   ROperator_Trilu(int upper, std::string nameX, std::string nameK, std::string nameY)
      : fUpper(upper),
        fNX(UTILITY::Clean_name(nameX)),
        fNK(UTILITY::Clean_name(nameK)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames  = { fNX, fNK };
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE Trilu: input tensor '" + fNX +
                                  "' not found in model");

      fShape = model.GetDimTensorShape(fNX);
      if (fShape.size() < 2)
         throw std::runtime_error("SOFIE Trilu: input tensor '" + fNX +
                                  "' must have at least 2 dimensions, got " +
                                  std::to_string(fShape.size()));

      fHasStridedInput = model.IsStridedInputTensor(fNX);
      if (!fNK.empty() && model.IsStridedInputTensor(fNK))
         throw std::runtime_error("SOFIE Trilu - strided input is only supported for the data tensor");
      fN = fShape.back();
      fM = fShape[fShape.size() - 2];
      fTotal = ConvertDimShapeToLength(fShape);

      if (!fNK.empty()) {
         if (model.IsInitializedTensor(fNK) || model.IsConstantTensor(fNK)) {
            // Bake the constant value into generated code.
            auto data_ptr = static_cast<int64_t*>(
               model.GetInitializedTensorData(fNK).get());
            fK        = data_ptr[0];
            fKIsStatic = true;
         } else {
            fKIsStatic = false;
         }
      }

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);

      if (model.Verbose()) {
         std::cout << "Trilu: " << fNX
                   << " upper=" << fUpper << " k=";
         if (fKIsStatic) std::cout << fK;
         else            std::cout << "dyn(" << fNK << ")";
         std::cout << " -> " << fNY
                   << " " << ConvertDimShapeToString(fShape) << std::endl;
      }
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty())
         throw std::runtime_error(
            "SOFIE Trilu: Generate called before Initialize");

      std::stringstream out;
      out << "\n//------ TRILU\n";
      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      if (fHasStridedInput)
         out << GenerateStridedOffsetLambda(OpName, fNX, fShape);

      if (fKIsStatic) {
         out << "const int64_t k_" << OpName << " = " << fK << "LL;\n";
      } else {
         out << "const int64_t k_" << OpName
             << " = static_cast<int64_t>(tensor_" << fNK << "[0]);\n";
      }

      out << "for (std::size_t id = 0; id < static_cast<std::size_t>(" << fTotal << "); ++id) {\n";
      out << "const std::size_t mat_id = id % static_cast<std::size_t>("
                      << fM.GetVal() << " * " << fN.GetVal() << ");\n";
      out << "const std::ptrdiff_t row = "
                      << "static_cast<std::ptrdiff_t>(mat_id / static_cast<std::size_t>(" << fN.GetVal() << "));\n";
      out << "const std::ptrdiff_t col = "
                      << "static_cast<std::ptrdiff_t>(mat_id % static_cast<std::size_t>(" << fN.GetVal() << "));\n";
      if (fUpper) {
         out << "const bool keep = (col >= row + k_" << OpName << ");\n";
      } else {
         out << "const bool keep = (col <= row + k_" << OpName << ");\n";
      }
      out << "tensor_" << fNY << "[id] = keep ? tensor_" << fNX
                      << (fHasStridedInput ? "[xoff_" + OpName + "(id)]" : std::string("[id]"))
                      << " : static_cast<" << TensorType<T>::Name() << ">(0);\n";
      out << "}\n";

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty())
         throw std::runtime_error(
            "SOFIE Trilu: Generate_GPU_Kernel_ALPAKA called before Initialize");

      std::stringstream op;
      op << "\n//------ TRILU_KERNEL_ALPAKA\n";
      op << "struct TriluKernel_" << OpName << " {\n";
      op << "template<typename TAcc, typename T>\n";
      op << "ALPAKA_FN_ACC void operator()("
               << "TAcc const& acc, "
               << "T const* __restrict__ input, "
               << "T* __restrict__ output, "
               << "const std::size_t total, "
               << "const std::size_t M, "
               << "const std::size_t N, "
               << "const std::ptrdiff_t k"
               << (fHasStridedInput ? ", sofie_strided_layout<" + std::to_string(fShape.size()) + "> const layoutX" : std::string())
               << ") const {\n";
      op << "auto const idx = "
               << "alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op << "if (idx >= total) return;\n";
      op << "const std::size_t    MN     = M * N;\n";
      op << "const std::size_t    mat_id = idx % MN;\n";
      op << "const std::ptrdiff_t row    = "
               << "static_cast<std::ptrdiff_t>(mat_id / N);\n";
      op << "const std::ptrdiff_t col    = "
               << "static_cast<std::ptrdiff_t>(mat_id % N);\n";
      if (fUpper) {
         op << "const bool keep = (col >= row + k);\n";
      } else {
         op << "const bool keep = (col <= row + k);\n";
      }
      op << "output[idx] = keep ? "
         << (fHasStridedInput ? StridedKernelRead("input", "layoutX", "idx") : std::string("input[idx]")) << " : T(0);\n";
      op << "}\n";
      op << "};\n";
      return op.str();
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string OpName) override {
      std::string cleaned = "op_" + OpName;
      return "TriluKernel_" + cleaned + " triluKernel_" + cleaned + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      if (fShape.empty())
         throw std::runtime_error(
            "SOFIE Trilu: Generate_GPU_ALPAKA called before Initialize");

      std::string cleanOp = "op_" + OpName;
      std::stringstream out;
      out << "\n//------ TRILU_GPU_ALPAKA\n";
      if (fHasStridedInput)
         out << GenerateStridedBroadcastLayout(cleanOp + "_X", fNX, fShape, fShape.size(), fShape);

      if (fKIsStatic) {
         out << "const std::ptrdiff_t k_" << cleanOp
             << " = static_cast<std::ptrdiff_t>(" << fK << "LL);\n";
      } else {
         out << "std::ptrdiff_t k_" << cleanOp << ";\n";
         out << "{\n";
         out << "auto hostK = alpaka::allocBuf<int64_t, Idx>(host, Ext1D::all(Idx{1}));\n";
         out << "alpaka::memcpy(queue, hostK, deviceBuf_" << fNK << ");\n";
         out << "alpaka::wait(queue);\n";
         out << "k_" << cleanOp << " = static_cast<std::ptrdiff_t>("
             << "*reinterpret_cast<const int64_t*>(alpaka::getPtrNative(hostK)));\n";
         out << "}\n";
      }

      out << "auto const elementsPerThread_" << fNY
          << " = Vec::all(static_cast<Idx>(1));\n";
      out << "auto const elementsPerGrid_" << fNY
          << " = Vec::all(Idx{static_cast<Idx>(" << fTotal << ")});\n";
      out << "auto const workDiv_" << fNY
          << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << "auto task_" << cleanOp
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
          << ", triluKernel_" << cleanOp
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", static_cast<std::size_t>(" << fTotal << ")"
          << ", static_cast<std::size_t>(" << fM.GetVal() << ")"
          << ", static_cast<std::size_t>(" << fN.GetVal() << ")"
          << ", k_" << cleanOp << (fHasStridedInput ? ", layout_" + cleanOp + "_X" : std::string()) << ");\n";
      out << "alpaka::enqueue(queue, task_" << cleanOp << ");\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_TRILU
