#ifndef SOFIE_ROPERATOR_QUANTBOUNDARY
#define SOFIE_ROPERATOR_QUANTBOUNDARY

// Defines the canonical quantization boundaries that every frontend translates into:
//   QuantEncode   float -> codes on a QuantSpec
//   QuantDecode   codes -> float
// ONNX QuantizeLinear/DequantizeLinear map one to one; a QONNX Quant maps to Decode(Encode(x)).
// Both resolve their QuantSpec in Initialize, fold when their input is an initializer, and
// generate Alpaka code through QuantCodec, so a boundary no handler absorbs runs on every backend.

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/quantization/spec/RQuantization_Spec.hxx"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace SOFIE {

// Grid description of a boundary: initializer names plus the attributes a frontend reads.
// ResolveQuantSpec turns it into a QuantSpec once shapes are known.
struct QuantBoundaryParameters {
   std::string scaleTensor;
   std::optional<double> scaleValue;   // Literal per-tensor scale when no initializer holds it.
   std::string zeroPointTensor;   // Zero-point initializer; empty means zero point 0.
   std::string bitWidthTensor;    // QONNX scalar initializer holding the bit width.
   EQuantEncoding encoding = EQuantEncoding::Integer;
   unsigned bitWidth = 8;         // Bit width when bitWidthTensor is empty.
   bool isSigned = true;
   bool narrow = false;
   EQuantRounding rounding = EQuantRounding::NearestEven;
   std::optional<int> axis;       // Explicit per-channel axis; inferred from the shape when empty.
   std::string origin;            // Originating frontend node, for diagnostics.
};

// Resolves the parameters' initializers into a QuantSpec for a tensor of `shape`.
// Defined in the frontend library.
QuantSpec ResolveQuantSpec(RModel &model, const QuantBoundaryParameters &params,
                           const std::vector<std::size_t> &shape);

// Returns the stride for the per-element channel of a per-channel spec: (index / stride) % channels.
std::size_t QuantSpecChannelStride(const QuantSpec &spec, const std::vector<std::size_t> &shape);

class ROperator_QuantBoundary : public ROperator {
protected:
   std::string fNX;
   std::string fNY;
   QuantBoundaryParameters fParams;
   QuantSpec fSpec;
   std::vector<std::size_t> fShape;
   // Tensor this boundary repeats exactly, set by canonicalization; the boundary then emits a
   // view of that tensor instead of a kernel.
   std::string fViewOf;

   ROperator_QuantBoundary(std::string nameX, std::string nameY, QuantBoundaryParameters params)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)), fParams(std::move(params))
   {
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }

   void ResolveSpec(RModel &model)
   {
      fShape = model.GetTensorShape(fNX);
      fSpec = ResolveQuantSpec(model, fParams, fShape);
      model.AddNeededCustomHeader("SOFIE/quantization/spec/RQuantization_Codec.hxx");
      model.AddNeededStdLib("cstdint");
   }

   // Return the per-element scale and zero-point expressions for generated code.
   std::string ScaleExpr(const std::string &channel) const
   {
      return fSpec.IsPerTensor() ? ExactDoubleLiteral(fSpec.Scale()) : "kScales[" + channel + "]";
   }
   std::string ZeroPointExpr(const std::string &channel) const
   {
      return fSpec.IsPerTensor() ? std::to_string(fSpec.ZeroPoint()) + ".0" : "kZeroPoints[" + channel + "]";
   }
   // Returns the per-channel parameter tables, emitted as kernel members. Kernel arguments are
   // size-limited, so tables wider than 256 channels throw.
   std::string ChannelTables(const std::string &indent) const
   {
      if (fSpec.IsPerTensor())
         return "";
      const auto channels = fSpec.ChannelCount();
      if (channels > 256)
         throw std::runtime_error("SOFIE quantization boundary " + fNY + " has " + std::to_string(channels) +
                                  " channels; per-channel boundaries wider than 256 must be absorbed by a handler");
      std::ostringstream out;
      out << indent << "double kScales[" << channels << "] = {";
      for (std::size_t c = 0; c < channels; ++c)
         out << (c ? ", " : "") << ExactDoubleLiteral(fSpec.Scale(c));
      out << "};\n" << indent << "double kZeroPoints[" << channels << "] = {";
      for (std::size_t c = 0; c < channels; ++c)
         out << (c ? ", " : "") << fSpec.ZeroPoint(c) << ".0";
      out << "};\n";
      return out.str();
   }
   std::string ChannelIndexStatement(const std::string &indent, const std::string &index) const
   {
      if (fSpec.IsPerTensor())
         return "";
      return indent + "const std::size_t channel = (" + index + " / " +
             std::to_string(QuantSpecChannelStride(fSpec, fShape)) + ") % " +
             std::to_string(fSpec.ChannelCount()) + ";\n";
   }

   // Returns the Alpaka launch, one thread per element from fNX to fNY, or the view declaration.
   std::string LaunchAlpaka(const std::string &opName, const std::string &kernelInstance) const
   {
      const auto length = ConvertShapeToLength(fShape);
      std::stringstream out;
      if (!fViewOf.empty()) {
         out << "\n//------ " << KernelTag() << " " << opName << " (view over " << fViewOf << ")\n";
         out << SP << "auto deviceBuf_" << fNY << " = alpaka::createView(devAcc, alpaka::getPtrNative(deviceBuf_"
             << fViewOf << "), static_cast<Idx>(" << length << "));\n";
         return out.str();
      }
      out << "\n//------ " << KernelTag() << " " << opName << " [" << ToString(fSpec) << "]\n";
      out << SP << "auto const elementsPerGrid_" << fNY << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << SP << "alpaka::enqueue(queue, alpaka::createTaskKernel<Acc>(workDiv_" << fNY << ", " << kernelInstance
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), "
          << length << "));\n";
      return out.str();
   }

   virtual const char *KernelTag() const = 0;

   int RoundingCode() const { return static_cast<int>(fSpec.rounding); }

public:
   bool IsQuantizationBoundary() const override { return true; }
   std::string GetQuantizationSourceTensor() const override { return fNX; }

   const std::string &GetInputTensor() const { return fNX; }
   const std::string &GetOutputTensor() const { return fNY; }
   const QuantSpec &GetSpec() const { return fSpec; }
   const std::vector<std::size_t> &GetShape() const { return fShape; }

   // Marks this boundary's output as equal to `tensor`, which another operator computes.
   void MarkViewOf(const std::string &tensor)
   {
      fViewOf = tensor;
      fInputTensorNames = {tensor};
   }
   bool IsView() const { return !fViewOf.empty(); }

   // Returns true for a per-tensor boundary that runs as a kernel, so it joins elementwise fusion
   // chains (e.g. one kernel for Clip -> encode). Per-channel boundaries need the channel index,
   // which a fused kernel lacks, so they keep their own kernel.
   bool IsElementwise() const override { return fSpec.IsPerTensor() && !fIsOutputConstant && !IsView(); }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      if (input.empty())
         return {};
      return {input.front()};
   }
};

class ROperator_QuantEncode final : public ROperator_QuantBoundary {
   const char *KernelTag() const override { return "QUANT_ENCODE"; }

   std::string EncodeExpr(const std::string &value, const std::string &channel) const
   {
      if (fSpec.IsFloatingPoint())
         return "SOFIE::QuantCodec::EncodeE4M3Scaled(static_cast<double>(" + value + "), " + ScaleExpr(channel) + ")";
      return "SOFIE::QuantCodec::EncodeInteger(static_cast<double>(" + value + "), " + ScaleExpr(channel) + ", " +
             ZeroPointExpr(channel) + ", " + ExactDoubleLiteral(fSpec.codeMin) + ", " +
             ExactDoubleLiteral(fSpec.codeMax) + ", " + std::to_string(RoundingCode()) + ")";
   }

   void FoldConstant(RModel &model);

public:
   ROperator_QuantEncode(std::string nameX, std::string nameY, QuantBoundaryParameters params)
      : ROperator_QuantBoundary(std::move(nameX), std::move(nameY), std::move(params))
   {
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType>) override { return {fSpec.CodeType()}; }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE quantization encode input " + fNX + " is not found in the model");
      ResolveSpec(model);
      if (model.IsInitializedTensor(fNX) && model.GetTensorType(fNX) == ETensorType::FLOAT) {
         FoldConstant(model);
         return;
      }
      model.AddIntermediateTensor(fNY, fSpec.CodeType(), fShape);
   }

   std::string Generate(std::string) override
   {
      throw std::runtime_error("SOFIE quantization boundary " + fName +
                               " generates Alpaka code only; quantized models need RModel::GenerateGPU_ALPAKA");
   }

   // Returns the codes as the codec produces them: a whole-number double (integer grids) or the
   // E4M3 byte, both exact. The fused kernel's store converts them to the output type.
   std::string GetElementwiseExpr(const std::string &v) const override { return EncodeExpr(v, "0"); }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override
   {
      if (fIsOutputConstant || IsView())
         return "";
      opName = "op_" + opName;
      std::string op = "\n//------ QUANT_ENCODE_KERNEL_ALPAKA " + opName + "\n";
      op += SP + "struct QuantEncodeKernel_" + opName + " {\n";
      op += ChannelTables(SP + SP);
      op += SP + SP + "template<typename TAcc, typename TIn, typename TOut>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TIn const * x, TOut * y, "
                      "std::size_t const length) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= length) return;\n";
      op += ChannelIndexStatement(SP + SP + SP, "idx");
      op += SP + SP + SP + "y[idx] = static_cast<TOut>(" + EncodeExpr("x[idx]", "channel") + ");\n";
      op += SP + SP + "}\n" + SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      if (fIsOutputConstant || IsView())
         return "";
      opName = "op_" + opName;
      return SP + "QuantEncodeKernel_" + opName + " quantEncodeKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (fIsOutputConstant)
         return "";
      opName = "op_" + opName;
      return LaunchAlpaka(opName, "quantEncodeKernel_" + opName);
   }
};

class ROperator_QuantDecode final : public ROperator_QuantBoundary {
   const char *KernelTag() const override { return "QUANT_DECODE"; }

   std::string DecodeExpr(const std::string &code, const std::string &channel) const
   {
      if (fSpec.IsFloatingPoint())
         return "SOFIE::QuantCodec::DecodeE4M3(static_cast<std::uint8_t>(" + code + ")) * " + ScaleExpr(channel);
      return "SOFIE::QuantCodec::DecodeInteger(static_cast<double>(" + code + "), " + ScaleExpr(channel) + ", " +
             ZeroPointExpr(channel) + ")";
   }

   void FoldConstant(RModel &model);

public:
   ROperator_QuantDecode(std::string nameX, std::string nameY, QuantBoundaryParameters params)
      : ROperator_QuantBoundary(std::move(nameX), std::move(nameY), std::move(params))
   {
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType>) override { return {ETensorType::FLOAT}; }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE quantization decode input " + fNX + " is not found in the model");
      ResolveSpec(model);
      // A constant carrier decodes at build time. Its codes remain this boundary's input, which
      // handlers that consume codes read.
      if (model.IsInitializedTensor(fNX)) {
         FoldConstant(model);
         return;
      }
      model.AddIntermediateTensor(fNY, ETensorType::FLOAT, fShape);
   }

   std::string Generate(std::string) override
   {
      throw std::runtime_error("SOFIE quantization boundary " + fName +
                               " generates Alpaka code only; quantized models need RModel::GenerateGPU_ALPAKA");
   }

   // Returns the decode rounded to float, as the unfused kernel stores it, so a fused chain
   // produces the same bits.
   std::string GetElementwiseExpr(const std::string &v) const override
   {
      return "static_cast<float>(" + DecodeExpr(v, "0") + ")";
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override
   {
      if (fIsOutputConstant || IsView())
         return "";
      opName = "op_" + opName;
      std::string op = "\n//------ QUANT_DECODE_KERNEL_ALPAKA " + opName + "\n";
      op += SP + "struct QuantDecodeKernel_" + opName + " {\n";
      op += ChannelTables(SP + SP);
      op += SP + SP + "template<typename TAcc, typename TIn, typename TOut>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TIn const * x, TOut * y, "
                      "std::size_t const length) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= length) return;\n";
      op += ChannelIndexStatement(SP + SP + SP, "idx");
      op += SP + SP + SP + "y[idx] = static_cast<TOut>(" + DecodeExpr("x[idx]", "channel") + ");\n";
      op += SP + SP + "}\n" + SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      if (fIsOutputConstant || IsView())
         return "";
      opName = "op_" + opName;
      return SP + "QuantDecodeKernel_" + opName + " quantDecodeKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (fIsOutputConstant)
         return "";
      opName = "op_" + opName;
      return LaunchAlpaka(opName, "quantDecodeKernel_" + opName);
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTBOUNDARY
