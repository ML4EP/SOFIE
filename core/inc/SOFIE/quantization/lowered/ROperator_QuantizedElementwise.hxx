#ifndef SOFIE_ROPERATOR_QUANTIZEDELEMENTWISE
#define SOFIE_ROPERATOR_QUANTIZEDELEMENTWISE

// Quantized elementwise Add / Mul on operands in any decided form (int8/uint8/float8 codes decoded
// on their own grid, floats read directly), computed in double and stored per a QuantStorePlan.
// Multidirectional broadcasting follows ROperator_BasicBinary: operand extents are right-aligned
// and a size-1 axis broadcasts through a zero stride.

#include "SOFIE/quantization/lowered/ROperator_QuantLowered.hxx"

namespace SOFIE {

class ROperator_QuantizedElementwise final : public ROperator_QuantLowered {
public:
   enum class EOp { Add, Mul };

private:
   EOp fOp;
   QuantBinding fA, fB;
   std::string fY;
   QuantStorePlan fStore;
   std::vector<std::size_t> fShapeA, fShapeB, fShapeY;

   // Returns the operand offset for output element `idx`: `idx` itself when the operand has the
   // output's shape, the broadcast mixed-radix offset otherwise.
   std::string OffsetExpression(const std::vector<std::size_t> &operand) const
   {
      if (operand == fShapeY)
         return "idx";
      const std::size_t rank = fShapeY.size();
      std::vector<std::size_t> extent(rank, 1);
      for (std::size_t i = 0; i < operand.size(); ++i)
         extent[rank - operand.size() + i] = operand[i];
      std::string expr = "0";
      std::size_t outStride = 1, opStride = 1;
      for (std::size_t axis = rank; axis-- > 0;) {
         if (extent[axis] != 1)
            expr += " + ((idx / " + std::to_string(outStride) + ") % " + std::to_string(fShapeY[axis]) + ") * " +
                    std::to_string(opStride);
         opStride *= extent[axis];
         outStride *= fShapeY[axis];
      }
      return "(" + expr + ")";
   }

public:
   ROperator_QuantizedElementwise(EOp op, QuantBinding a, QuantBinding b, std::string y, QuantStorePlan store,
                                  std::vector<std::size_t> shapeA, std::vector<std::size_t> shapeB,
                                  std::vector<std::size_t> shapeY)
      : fOp(op), fA(std::move(a)), fB(std::move(b)), fY(std::move(y)), fStore(std::move(store)),
        fShapeA(std::move(shapeA)), fShapeB(std::move(shapeB)), fShapeY(std::move(shapeY))
   {
      fInputTensorNames = {fA.tensor, fB.tensor};
      fOutputTensorNames = {fY};
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      std::string op = "\n//------ QUANTIZED_ELEMENTWISE_KERNEL_ALPAKA " + opName + "\n";
      op += SP + "struct QuantizedElementwiseKernel_" + opName + " {\n";
      op += SP + SP + "template<typename TAcc, typename TA, typename TB, typename TOut>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TA const * a, TB const * b, TOut * y, "
                      "std::size_t const n) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= n) return;\n";
      op += SP + SP + SP + "double const va = " + QuantLoadExpression(fA, "a[" + OffsetExpression(fShapeA) + "]") + ";\n";
      op += SP + SP + SP + "double const vb = " + QuantLoadExpression(fB, "b[" + OffsetExpression(fShapeB) + "]") + ";\n";
      op += SP + SP + SP + "double v = " + std::string(fOp == EOp::Add ? "va + vb" : "va * vb") + ";\n";
      op += fStore.Statements("v", "y[idx]", SP + SP + SP);
      op += SP + SP + "}\n" + SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      return SP + "QuantizedElementwiseKernel_" + opName + " quantizedElementwiseKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      return "\n//------ QUANTIZED_ELEMENTWISE " + opName + "\n" +
             Launch1D(opName, "quantizedElementwiseKernel_" + opName, ConvertShapeToLength(fShapeY),
                      Ptr(fA.tensor) + ", " + Ptr(fB.tensor) + ", " + Ptr(fY));
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZEDELEMENTWISE
