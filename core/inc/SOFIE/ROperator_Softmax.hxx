#ifndef SOFIE_ROPERATOR_Softmax
#define SOFIE_ROPERATOR_Softmax

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>

namespace SOFIE {

class ROperator_Softmax final : public ROperator {

private:
   bool fLogSoftmax;  // for the logsoftmax case
   bool fUseVDT = false;
   int64_t fAttrAxis;

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

   std::string fType;

   // Output tensor of an absorbed Clip (and optional QuantizeLinear), written in place of fNY.
   std::string fFusedOutputTensor; // empty: write fNY
   // Grid of an absorbed QuantizeLinear; when set, pass 3 writes a low-precision carrier.
   // One QuantizationGrid encode covers int8 and FP8.
   std::optional<QuantizationGrid> fOutputGrid;
   bool fQuantHasClip = false;
   double fQuantClipLow = 0.0;
   double fQuantClipHigh = 0.0;

   // Grid of an absorbed DequantizeLinear; when set, the carrier input decodes at the load.
   std::optional<QuantizationGrid> fInputGrid;

   bool ReadsCarrierInput() const { return fInputGrid.has_value(); }

   // Input parameter, load expression and working type of pass 1. Held together because a
   // mismatched parameter and load still compiles.
   struct SoftmaxInputBinding {
      std::string templateParams;   // after `typename TAcc, `, before `, typename TOut`
      std::string parameter;        // the input parameter declaration
      std::string workingType;      // T, when it cannot be deduced from the input; else empty
      std::function<std::string(const std::string &)> load;   // one element, by index expression
   };

   SoftmaxInputBinding InputBinding() const
   {
      // Decodes through the same portable codec as a QuantDecode boundary, so the result is
      // bit-identical. T is the decoded type, not the carrier type.
      if (ReadsCarrierInput()) {
         const std::string scale = ExactDoubleLiteral(fInputGrid->scale);
         return {"typename TIn", "TIn const* __restrict__ X", "float",
                 [scale](const std::string &at) {
                    return "static_cast<T>(SOFIE::QuantCodec::DecodeE4M3(static_cast<std::uint8_t>(X[" + at +
                           "])) * " + scale + ")";
                 }};
      }
      return {"typename T", "T const* __restrict__ X", "",
              [](const std::string &at) { return "X[" + at + "]"; }};
   }

public:
   ROperator_Softmax() {}
   ROperator_Softmax(int64_t attr_axis, std::string nameX, std::string nameY, bool logSoftmax = false)
      : fLogSoftmax(logSoftmax),
      fAttrAxis(attr_axis), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))

   {
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
   }

   // Gates every fusion hook below. Only the register-resident emission reads each element
   // once and keeps it; the staged variant re-reads X in pass 2 and uses Y as scratch.
   bool CanFuseClip() const { return SoftmaxUsesRegisterResidentRows(); }
   // Returns false for log-softmax, whose output is not on the boundary's grid.
   bool CanFuseOutputOnGrid(EQuantizedOutputEmit mode) const override
   {
      return mode == EQuantizedOutputEmit::Carrier && CanFuseClip() && !fLogSoftmax;
   }

   // Encodes onto `grid` with no clamp. FuseQuantizedOutput covers the Softmax-Clip case.
   void FuseOutputOnGrid(const std::string &output, const QuantizationGrid &grid,
                         EQuantizedOutputEmit mode) override
   {
      if (mode != EQuantizedOutputEmit::Carrier)
         return ROperator::FuseOutputOnGrid(output, grid, mode);
      FuseQuantizedOutput(output, grid, false, 0.0, 0.0);
   }

   // Absorbs a following Clip and writes its output tensor in place of fNY.
   void FuseClip(std::string clipOutput, double clipLow, double clipHigh)
   {
      fFusedOutputTensor = std::move(clipOutput);
      fOutputTensorNames = { fFusedOutputTensor };
      fQuantHasClip = true;
      fQuantClipLow = clipLow;
      fQuantClipHigh = clipHigh;
   }

   // Absorbs a following Clip and encodes onto `grid`, writing `carrier` as a low-precision code.
   void FuseQuantizedOutput(std::string carrier, const QuantizationGrid &grid, bool hasClip,
                            double clipLow, double clipHigh)
   {
      FuseClip(std::move(carrier), clipLow, clipHigh);
      fQuantHasClip = hasClip;
      fOutputGrid = grid;
   }


   // Returns whether the carrier input can decode at the load, removing the DequantizeLinear
   // kernel and its float round trip. Gated as CanFuseClip.
   bool CanFuseDequantizedInput() const override
   {
      return SoftmaxUsesRegisterResidentRows();
   }

   bool FuseDequantizedInput(const std::string &from, const std::string &carrier,
                             const QuantizationGrid &grid) override
   {
      // Accepts per-tensor E4M3 only: an affine grid adds a zero point, and E5M2 has its own decode.
      if (!IsPerTensorE4M3(grid) || fNX != from || fInputGrid)
         return false;
      fNX = carrier;
      fInputGrid = grid;
      fInputTensorNames = {fNX};
      return true;
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; // suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) ==
          false) { // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE Softmax Op Input Tensor is not found in model");
      }
      fShape = model.GetDimTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      if (model.Verbose()) {
         std::cout << "Softmax -> " << fNY << " " << ConvertDimShapeToString(fShape) << std::endl;
      }
      fUseVDT = model.UseVDT();
      if (fUseVDT) {
         model.AddNeededCustomHeader("vdt/exp.h");
         if (fLogSoftmax)
            model.AddNeededCustomHeader("vdt/log.h");
      }
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator Softmax called to Generate without being initialized first");
      }
      std::stringstream out;
       out << "///------- Softmax " << opName << " ---> "  // << fNY << " "
           << ConvertDimShapeToString(fShape) << "\n" << std::endl;
      size_t size = fShape.size();
      auto length_str = ConvertDimShapeToLength(fShape);
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;

      std::string expFunction = (fUseVDT) ? "vdt::fast_expf" : "std::exp";
      std::string logFunction = (fUseVDT) ? "vdt::fast_logf" : "std::log";

      // Check if this is the special case where memory is contiguous.
      if (axis == size - 1) {
         std::string axis_size = fShape[axis].GetVal();
         std::string num_rows;
         if (IsInteger(length_str) && IsInteger(axis_size)) {
            num_rows = std::to_string(std::stoul(length_str) / std::stoul(axis_size));
         } else {
            num_rows = "(" + length_str + ") / (" + axis_size + ")";
         }

         out << SP << "//-----  softmax axis is last one - " << axis << "\n";
         out << SP << "for (int i = 0; i < " << num_rows << "; ++i) {\n";
         out << SP << SP << "size_t offset = i * " << axis_size << ";\n";
         out << SP << SP << fType << " const * x_ptr = &tensor_" << fNX << "[offset];\n";
         out << SP << SP << fType << " * y_ptr = &tensor_" << fNY << "[offset];\n";

         out << SP << SP << fType << " vmax = x_ptr[0];\n";
         out << SP << SP << "for (int j = 1; j < " << axis_size << "; ++j) {\n";
         out << SP << SP << SP << "if (x_ptr[j] > vmax) vmax = x_ptr[j];\n";
         out << SP << SP << "}\n";

         out << SP << SP << fType << " sum = 0.0;\n";
         out << SP << SP << "for (int j = 0; j < " << axis_size << "; ++j) {\n";
         out << SP << SP << SP << "y_ptr[j] = " << expFunction << "(x_ptr[j] - vmax);\n";
         out << SP << SP << SP << "sum += y_ptr[j];\n";
         out << SP << SP << "}\n";

         out << SP << SP << fType << " inv_sum = 1.0f / sum;\n";
         out << SP << SP << "for (int j = 0; j < " << axis_size << "; ++j) {\n";
         out << SP << SP << SP << "y_ptr[j] *= inv_sum;\n";
         if (fLogSoftmax)
            out << SP << SP << SP << "y_ptr[j] = " << logFunction << "(y_ptr[j]);\n";
         out << SP << SP << "}\n";
         out << SP << "}\n";

      } else {
         // generic case for any axis
         auto stride = UTILITY::ComputeStrideFromShape(fShape);
         size_t k = 0;
         std::vector<std::string> l(size);
         for (size_t i = 0; i < size; i++) {
            if (i != axis) {
               for (size_t j = 0; j < k; j++) out << SP;
               l[i] = std::string("i") + std::to_string(i);
               out << SP << "for (int " << l[i] << " = 0; " << l[i] << " < " << fShape[i] << "; " << l[i] << "++) {\n";
               k++;
            }
         }
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << fType << " sum = 0.;\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "size_t index = ";
         bool first = true;
         for (size_t i = 0; i < size; i++) {
            if (i == axis) continue;
            if (!first) out << " + ";
            if (stride[i].GetVal() != "1")
               out << stride[i] << "*";
            out << l[i];
            first = false;
         }
         out << ";\n";
         // find maximum looping along reduced axis
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << fType << " vmax = tensor_" << fNX << "[index];\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 1; i < " << fShape[axis] << "; i++) {\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << fType << " x = tensor_" << fNX << "[index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << "];\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "if (x > vmax) vmax = x;\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         // compute softmax
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 0; i < " << fShape[axis] << "; i++) {\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "size_t id = index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << ";\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "tensor_" << fNY << "[id] = " << expFunction << "(tensor_" << fNX << "[id] - vmax);\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "sum += tensor_" << fNY << "[id];\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         // normalize
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 0; i < " << fShape[axis] << "; i++) {\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "size_t id = index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << ";\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "tensor_" << fNY << "[id] /= sum;\n";
         if (fLogSoftmax) {
            for (size_t j = 0; j < size; j++) out << SP;
            out << "tensor_" << fNY << "[id] = " << logFunction << "(tensor_" << fNY << "[id]);\n";
         }
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         //end loops
         for (int i = static_cast<int>(k) - 1; i >= 0; i--) {
            for (int j = 0; j < i; j++) out << SP;
            out << "}\n";
         }
      }
      return out.str();
   }

   // GPU/Alpaka codegen: two kernels selected by `inner`, one thread per row when
   // inner > 1 and one block per row when inner == 1; the block path reduces pairwise.
private:
   // Threads per block for the contiguous path; a power of two for the tree reductions.
   static constexpr std::size_t kSoftmaxBlockThreads = 256;
   // Cap on the per-thread register array; above it the values spill to local memory and
   // the emitter uses the staged path instead.
   static constexpr std::size_t kSoftmaxMaxRegisters = 8;

   // Elements each thread owns when one block covers a row. Zero means the axis length
   // is a runtime parameter, so no compile-time register array can be sized.
   std::size_t SoftmaxElementsPerThread() const
   {
      if (fShape.empty())
         return 0;
      const auto &axisDim = fShape[SoftmaxAxis()];
      if (axisDim.isParam || axisDim.dim == 0)
         return 0;
      return (axisDim.dim + kSoftmaxBlockThreads - 1) / kSoftmaxBlockThreads;
   }

   // True when each thread can hold its elements in registers across all three passes
   // instead of staging them through Y. Visits elements in the strided loops' order.
   bool SoftmaxUsesRegisterResidentRows() const
   {
      const std::size_t perThread = SoftmaxElementsPerThread();
      return SoftmaxRowIsContiguous() && perThread > 0 && perThread <= kSoftmaxMaxRegisters;
   }

   std::size_t SoftmaxAxis() const
   {
      const std::size_t size = fShape.size();
      return fAttrAxis < 0 ? size + fAttrAxis : static_cast<std::size_t>(fAttrAxis);
   }
   std::string SoftmaxDimProduct(std::size_t lo, std::size_t hi) const
   {
      std::string product;
      for (std::size_t i = lo; i < hi; ++i)
         product += (product.empty() ? "(" : "*(") + fShape[i].GetVal() + ")";
      return product.empty() ? "1" : product;
   }
   // Structural, so it holds for parametric dims too: inner is the product of the axes
   // after the softmax axis, hence 1 exactly when that axis is last.
   bool SoftmaxRowIsContiguous() const
   {
      return !fShape.empty() && SoftmaxAxis() + 1 == fShape.size();
   }
   static std::string SubstitutePlaceholder(std::string text, const std::string &token,
                                            const std::string &value)
   {
      for (auto at = text.find(token); at != std::string::npos; at = text.find(token, at))
         text.replace(at, token.size(), value);
      return text;
   }

   // Block-wide reduction over kSoftmaxBlockThreads, unrolled at codegen time. In `combine`,
   // "@A" is the accumulator and "@B" the partner; both halves substitute their own names.
   std::string SoftmaxBlockReduce(const std::string &indent, const std::string &combine) const
   {
      const auto step = [&combine](const std::string &a, const std::string &b) {
         return SubstitutePlaceholder(SubstitutePlaceholder(combine, "@A", a), "@B", b);
      };
      std::string out;
      // Shared-memory tree only while the survivors span more than one warp; below 32 a
      // block-wide barrier buys nothing. `alpaka::warp::shfl_down`, not `__shfl_down_sync`, so
      // CPU and HIP keep their WarpSingleThread fallback. Lane `tid` still pairs with
      // `tid + half` in the same order, so the result is bit-identical to the tree.
      constexpr std::size_t kWarp = 32;
      for (std::size_t half = kSoftmaxBlockThreads / 2; half >= kWarp; half >>= 1) {
         const std::string h = std::to_string(half) + "u";
         out += indent + "if (tid < " + h + ") { " + step("shmem[tid]", "shmem[tid + " + h + "]") + " }\n";
         out += indent + "alpaka::syncBlockThreads(acc);\n";
      }
      out += indent + "if (tid < " + std::to_string(kWarp) + "u) {\n";
      out += indent + "   auto v = shmem[tid];\n";
      for (std::size_t half = kWarp / 2; half > 0; half >>= 1) {
         out += indent + "   { const auto other = alpaka::warp::shfl_down(acc, v, " +
                std::to_string(half) + "u); " + step("v", "other") + " }\n";
      }
      out += indent + "   if (tid == 0u) shmem[0] = v;\n";
      out += indent + "}\n";
      out += indent + "alpaka::syncBlockThreads(acc);\n";
      return out;
   }

public:
   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      const std::string kname = "SoftmaxKernel_" + opName;
      std::string op;
      op = "\n//------ SOFTMAX_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      // TOut is deduced: T for a plain Softmax, the carrier type with a fused output quantization.
      const auto binding = InputBinding();
      std::string templateParams = "typename TAcc";
      if (!binding.templateParams.empty())
         templateParams += ", " + binding.templateParams;
      templateParams += ", typename TOut";
      op += SP + SP + "template<" + templateParams + ">\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, " + binding.parameter +
            ", TOut* __restrict__ Y, std::size_t numRows, std::size_t axisLen, "
            "std::size_t inner) const {\n";
      if (!binding.workingType.empty())
         op += SP + SP + SP + "using T = " + binding.workingType + ";\n";

      if (SoftmaxRowIsContiguous()) {
         const std::string I3 = SP + SP + SP;
         const std::string I4 = SP + SP + SP + SP;
         const std::string bt = std::to_string(kSoftmaxBlockThreads) + "u";

         op += I3 + "// Contiguous rows (inner == 1): one block per row, " + bt +
               " threads striding\n";
         op += I3 + "// the axis so a warp reads one cache line instead of 32.\n";
         op += I3 + "auto& shmem = alpaka::declareSharedVar<T[" +
               std::to_string(kSoftmaxBlockThreads) + "], __COUNTER__>(acc);\n";
         // row is a BLOCK index, so every thread in the block takes this branch together
         // and the syncBlockThreads below are never reached by a partial block.
         op += I3 + "auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
         op += I3 + "auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
         op += I3 + "(void)inner;\n";
         op += I3 + "if (row >= numRows) return;\n";
         op += I3 + "const std::size_t base = row * axisLen;\n\n";

         const std::size_t perThread = SoftmaxElementsPerThread();
         const bool registerResident = SoftmaxUsesRegisterResidentRows();
         // k = tid + j * blockThreads visits elements in the strided loop's order, so
         // the max and the sum accumulate in the same sequence.
         if (registerResident) {
            const std::string pt = std::to_string(perThread);
            op += I3 + "// Row fits in registers (" + pt + " element(s) per thread), so the\n";
            op += I3 + "// exponentials never round-trip through Y. Same access order as the\n";
            op += I3 + "// strided form, hence the same reduction order and the same result.\n";
            op += I3 + "T vals[" + pt + "];\n\n";

            op += I3 + "// Pass 1: block-wide max. Threads with no element contribute the\n";
            op += I3 + "// identity, which is what makes axisLen < " + bt + " safe.\n";
            op += I3 + "T vmax = std::numeric_limits<T>::lowest();\n";
            op += I3 + "#pragma unroll\n";
            op += I3 + "for (std::size_t j = 0; j < " + pt + "; ++j) {\n";
            op += I4 + "const std::size_t k = tid + j * " + bt + ";\n";
            op += I4 + "if (k < axisLen) { vals[j] = " + binding.load("base + k") +
                  "; if (vals[j] > vmax) vmax = vals[j]; }\n";
            op += I3 + "}\n";
            op += I3 + "shmem[tid] = vmax;\n";
            op += I3 + "alpaka::syncBlockThreads(acc);\n";
            op += SoftmaxBlockReduce(I3, "if (@B > @A) @A = @B;");
            op += I3 + "vmax = shmem[0];\n";
            op += I3 + "alpaka::syncBlockThreads(acc);\n\n";

            op += I3 + "// Pass 2: exp onto the shifted grid, kept in registers, block-wide sum.\n";
            op += I3 + "T sum = static_cast<T>(0);\n";
            op += I3 + "#pragma unroll\n";
            op += I3 + "for (std::size_t j = 0; j < " + pt + "; ++j) {\n";
            op += I4 + "const std::size_t k = tid + j * " + bt + ";\n";
            op += I4 + "if (k < axisLen) { vals[j] = static_cast<T>(exp(vals[j] - vmax)); sum += vals[j]; }\n";
            op += I3 + "}\n";
            op += I3 + "shmem[tid] = sum;\n";
            op += I3 + "alpaka::syncBlockThreads(acc);\n";
            op += SoftmaxBlockReduce(I3, "@A += @B;");
            op += I3 + "const T inv = static_cast<T>(1) / shmem[0];\n";
            op += I3 + "alpaka::syncBlockThreads(acc);\n\n";

            op += I3 + "// Pass 3: normalise straight out of registers into Y.\n";
            op += I3 + "#pragma unroll\n";
            op += I3 + "for (std::size_t j = 0; j < " + pt + "; ++j) {\n";
            op += I4 + "const std::size_t k = tid + j * " + bt + ";\n";
            op += I4 + "if (k >= axisLen) continue;\n";
            op += I4 + "T v = vals[j] * inv;\n";
            if (fLogSoftmax)
               op += I4 + "v = static_cast<T>(log(v));\n";
            // The absorbed Clip clamps in T, the type and position the Clip kernel used.
            if (fQuantHasClip) {
               op += I4 + "v = v < static_cast<T>(" + ExactDoubleLiteral(fQuantClipLow) +
                     ") ? static_cast<T>(" + ExactDoubleLiteral(fQuantClipLow) + ") : (v > static_cast<T>(" +
                     ExactDoubleLiteral(fQuantClipHigh) + ") ? static_cast<T>(" +
                     ExactDoubleLiteral(fQuantClipHigh) + ") : v);\n";
            }
            if (fOutputGrid && fOutputGrid->IsFloatingPoint()) {
               // FP8: divides by the scale in double, rounds to float, then applies the
               // portable saturating E4M3 encode. No zero point.
               op += I4 + "auto q = SOFIE::QuantCodec::EncodeE4M3(static_cast<double>(static_cast<float>("
                     "static_cast<double>(v) / " + ExactDoubleLiteral(fOutputGrid->scale) + ")));\n";
               op += I4 + "Y[base + k] = static_cast<TOut>(q);\n";
            } else if (fOutputGrid) {
               // Matches the QuantEncode boundary: round half to even, add the zero point, saturate.
               const auto qMin = static_cast<std::int64_t>(fOutputGrid->codeMin);
               const auto qMax = static_cast<std::int64_t>(fOutputGrid->codeMax);
               op += I4 + "double q = nearbyint(static_cast<double>(v) / " +
                     ExactDoubleLiteral(fOutputGrid->scale) + ") + " +
                     std::to_string(fOutputGrid->zeroPoint) + ";\n";
               op += I4 + "q = (q < " + std::to_string(qMin) + ") ? " + std::to_string(qMin) +
                     " : ((q > " + std::to_string(qMax) + ") ? " + std::to_string(qMax) + " : q);\n";
               op += I4 + "Y[base + k] = static_cast<TOut>(q);\n";
            } else {
               op += I4 + "Y[base + k] = v;\n";
            }
            op += I3 + "}\n";
         } else {
            op += I3 + "// Pass 1: block-wide max. Threads with no element contribute the\n";
            op += I3 + "// identity, which is what makes axisLen < " + bt + " safe.\n";
            op += I3 + "T vmax = std::numeric_limits<T>::lowest();\n";
            op += I3 + "for (std::size_t k = tid; k < axisLen; k += " + bt +
                  ") { T v = X[base + k]; if (v > vmax) vmax = v; }\n";
            op += I3 + "shmem[tid] = vmax;\n";
            op += I3 + "alpaka::syncBlockThreads(acc);\n";
            op += SoftmaxBlockReduce(I3, "if (@B > @A) @A = @B;");
            op += I3 + "vmax = shmem[0];\n";
            // Re-sync before reusing shmem, or a fast thread's pass-2 write races the slow
            // threads still reading shmem[0].
            op += I3 + "alpaka::syncBlockThreads(acc);\n\n";

            op += I3 + "// Pass 2: exp onto the shifted grid, store, block-wide sum.\n";
            op += I3 + "T sum = static_cast<T>(0);\n";
            op += I3 + "for (std::size_t k = tid; k < axisLen; k += " + bt +
                  ") { T e = static_cast<T>(exp(X[base + k] - vmax)); Y[base + k] = e; sum += e; }\n";
            op += I3 + "shmem[tid] = sum;\n";
            op += I3 + "alpaka::syncBlockThreads(acc);\n";
            op += SoftmaxBlockReduce(I3, "@A += @B;");
            op += I3 + "const T inv = static_cast<T>(1) / shmem[0];\n";
            op += I3 + "alpaka::syncBlockThreads(acc);\n\n";

            op += I3 + "// Pass 3: normalise.\n";
            op += I3 + "for (std::size_t k = tid; k < axisLen; k += " + bt + ") {\n";
            op += I4 + "Y[base + k] *= inv;\n";
            if (fLogSoftmax)
               op += I4 + "Y[base + k] = static_cast<T>(log(Y[base + k]));\n";
            op += I3 + "}\n";
         }
      } else {
         // inner > 1: adjacent threads already read adjacent addresses, so the serial
         // per-row scan is coalesced as it stands.
         op += SP + SP + SP + "const auto row = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + SP + "if (row >= numRows) return;\n";
         op += SP + SP + SP + "const std::size_t o = row / inner;\n";
         op += SP + SP + SP + "const std::size_t i = row % inner;\n";
         op += SP + SP + SP + "const std::size_t base = o * axisLen * inner + i;\n";
         op += SP + SP + SP + "T vmax = X[base];\n";
         op += SP + SP + SP + "for (std::size_t k = 1; k < axisLen; ++k) { T v = X[base + k * inner]; if (v > vmax) vmax = v; }\n";
         op += SP + SP + SP + "T sum = static_cast<T>(0);\n";
         op += SP + SP + SP + "for (std::size_t k = 0; k < axisLen; ++k) { T e = static_cast<T>(exp(X[base + k * inner] - vmax)); Y[base + k * inner] = e; sum += e; }\n";
         op += SP + SP + SP + "const T inv = static_cast<T>(1) / sum;\n";
         op += SP + SP + SP + "for (std::size_t k = 0; k < axisLen; ++k) {\n";
         op += SP + SP + SP + SP + "Y[base + k * inner] *= inv;\n";
         if (fLogSoftmax)
            op += SP + SP + SP + SP + "Y[base + k * inner] = static_cast<T>(log(Y[base + k * inner]));\n";
         op += SP + SP + SP + "}\n";
      }
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      return SP + "SoftmaxKernel_" + opName + " softmaxKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (fShape.empty())
         throw std::runtime_error("SOFIE Operator Softmax called to Generate without being initialized first");
      opName = "op_" + opName;
      const std::size_t size = fShape.size();
      const std::size_t axis = SoftmaxAxis();
      const std::string axisLen = "(" + fShape[axis].GetVal() + ")";
      const std::string inner = SoftmaxDimProduct(axis + 1, size);
      const std::string outer = SoftmaxDimProduct(0, axis);
      const std::string numRows = "(" + outer + ")*(" + inner + ")";

      std::stringstream out;
      out << "\n//------ SOFTMAX_GPU_ALPAKA\n";
      if (SoftmaxRowIsContiguous()) {
         // One block per row; the kernel strides the axis inside the block, so the grid
         // does not depend on axisLen.
         out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNY << "(\n";
         out << SP << SP << "Vec::all(static_cast<Idx>(" << numRows << ")),\n";
         out << SP << SP << "Vec::all(Idx{" << kSoftmaxBlockThreads << "u}),\n";
         out << SP << SP << "Vec::all(Idx{1u}));\n";
      } else {
         out << SP << "auto const elementsPerThread_" << fNY << " = Vec::all(static_cast<Idx>(1));\n";
         out << SP << "auto const elementsPerGrid_" << fNY << " = Vec::all(Idx{" << numRows << "});\n";
         out << SP << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      }
      // The fused form writes the QuantizeLinear carrier; the bypassed float intermediate is
      // never allocated.
      const std::string outBuf = fFusedOutputTensor.empty() ? fNY : fFusedOutputTensor;
      const std::string inArg = "alpaka::getPtrNative(deviceBuf_" + fNX + ")";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
          << ", softmaxKernel_" << opName << ", " << inArg
          << ", alpaka::getPtrNative(deviceBuf_" << outBuf << "), static_cast<Idx>(" << numRows
          << "), static_cast<Idx>(" << axisLen << "), static_cast<Idx>(" << inner << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   // <limits> for the numeric_limits identity the block-wide max reduction seeds with.
   std::vector<std::string> GetStdLibs() override
   {
      return { std::string("cmath"), std::string("limits") };
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_Softmax
