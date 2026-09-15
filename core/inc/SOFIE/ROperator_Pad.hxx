#ifndef SOFIE_ROPERATOR_Pad
#define SOFIE_ROPERATOR_Pad

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE{

template <typename T>
class ROperator_Pad final : public ROperator
{
public:
   enum EMode { kConstant, kReflect, kEdge, kWrap };
private:

   std::string fNX;
   std::string fNP;
   std::string fNCV;
   std::string fNAX;
   std::string fNY;
   T fConstantValue;
   EMode fMode;
   std::vector<Dim> fInputShape;
   std::vector<Dim> fOutputShape;
   std::vector<std::pair<int64_t, int64_t>> fPads;

public:

   ROperator_Pad(){}
   ROperator_Pad(const std::string & nameX, const std::string & nameP,  const std::string & nameCV,
                 const std::string & nameAX, const std::string & nameY, const std::string & mode) :
      fNX(UTILITY::Clean_name(nameX)), fNP(UTILITY::Clean_name(nameP)),
      fNCV(UTILITY::Clean_name(nameCV)), fNAX(UTILITY::Clean_name(nameAX)),
      fNY(UTILITY::Clean_name(nameY))
      {
         fMode = kConstant;
         if (mode == "constant")
            fMode = kConstant;
         else if (mode == "reflect")
            fMode = kReflect;
         else if (mode == "edge")
            fMode = kEdge;
         else if (mode == "wrap")
            fMode = kWrap;
         
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE Pad Op Input Tensor is not found in model");
      }

      fInputShape = model.GetDimTensorShape(fNX);

      if (fMode != EMode::kConstant) {
         throw std::runtime_error("SOFIE Pad Op supports now only Constant mode");
      }

      // get pads data
      int64_t * padsData = nullptr;
      if (model.IsInitializedTensor(fNP)) {
         padsData = static_cast<int64_t*>(model.GetInitializedTensorData(fNP).get());
      } else {
         throw std::runtime_error("SOFIE Pad Op supports now only initialized Pads data");
      }
      // get constant value
      fConstantValue = 0;
      if (!fNCV.empty()) {
         if (model.IsInitializedTensor(fNCV)) {
            T * cData = static_cast<T*>(model.GetInitializedTensorData(fNCV).get());
            fConstantValue = cData[0];
         } else {
            throw std::runtime_error("SOFIE Pad Op supports now only initialized Constant Value  data");
         }
      }
      std::vector<int64_t> axes;
      if (!fNAX.empty()) {
         if (model.IsInitializedTensor(fNAX)) {
            auto shape = model.GetTensorShape(fNAX);
            // it should be a 1D tensor
            size_t nax = shape[0];
            // switch types
            if (model.GetTensorType(fNAX) == ETensorType::INT64) {
               auto data = static_cast<int64_t*>(model.GetInitializedTensorData(fNAX).get());
               axes = std::vector<int64_t>(data, data + nax);
            } else if (model.GetTensorType(fNAX) == ETensorType::INT32) {
               auto data = static_cast<int32_t*>(model.GetInitializedTensorData(fNAX).get());
               axes.resize(nax);
               for (size_t i = 0; i < nax; i++)
                  axes[i] = data[i];
            }  else {
               throw std::runtime_error("SOFIE Pad Op invalid input Axes type");
            }
         } else {
            throw std::runtime_error("SOFIE Pad Op supports now only initialized Axes data");
         }
      }


      fOutputShape = fInputShape;
      size_t axesSize = axes.size();
      if (axesSize == 0) {
         for (size_t i = 0; i < fInputShape.size(); i++) {
            axes.push_back(i);
         }
         axesSize = fInputShape.size();
      }
      fPads.resize(fInputShape.size());
      for (size_t i = 0; i < fInputShape.size(); i++) {
         if (axes[i] < 0) axes[i] += fInputShape.size();
         if (axes[i] == int64_t(i)) {
            fPads[i].first = padsData[i];
            fPads[i].second = padsData[axesSize + i];
            int64_t padSum = fPads[i].first + fPads[i].second;
            if (!fInputShape[i].isParam) {
               int64_t outDim = static_cast<int64_t>(fInputShape[i].dim) + padSum;
               if (outDim < 0)
                  throw std::runtime_error("SOFIE Pad Op : invalid Pads values");
               fOutputShape[i] = Dim{static_cast<size_t>(outDim)};
            } else if (padSum != 0) {
               // dynamic dimension: build a symbolic expression for the padded size
               std::string expr = "(" + fInputShape[i].param + (padSum >= 0 ? " + " : " - ")
                                 + std::to_string(std::abs(padSum)) + ")";
               fOutputShape[i] = Dim{expr, size_t(-1)};
            }
            // else: dynamic dimension with no padding on this axis, output dim == input dim
         }
      }

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fOutputShape);

      if (model.Verbose()) {
         std::cout << "initializing Pad operator with pads ..  : ";
         for (auto & p : fPads)
            std::cout << "{ " << p.first << " , " << p.second << "} ";
         std::cout << std::endl;
         std::cout <<  "Pad: " << fNX << " " << ConvertDimShapeToString(fInputShape) << " -> " << fNY << " with shape " << ConvertDimShapeToString(fOutputShape)
                  << std::endl;
      }

   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fOutputShape.empty()){
         throw std::runtime_error("SOFIE Operator Pad called to Generate without being initialized first");
      }
      std::stringstream out;
      auto inputStride = UTILITY::ComputeStrideFromShape(fInputShape);
      auto outStride = UTILITY::ComputeStrideFromShape(fOutputShape);
      out << "\n//------ Pad\n";
      // fill first output tensor with the constant values
      std::string length = ConvertDimShapeToLength(fOutputShape);
      int dims = fOutputShape.size();
      out << "std::fill(tensor_" << fNY << ", tensor_" << fNY << " + " << length << ","
          << fConstantValue << ");\n";

      // copy now data from input tensor in output ones
      for (int i = 0; i < dims; i++) {
         for (int j = 1; j < i; j++) out << SP;
         out << "for (size_t id" << i << " = 0; id" << i << " < " << fInputShape[i].GetVal() << "; id"
             << i << "++) {\n";
      }
      // compute index from strides
      //linear_index = i_1 * stride[0] + i_2 * stride[1] + ... + i_N * stride[N-1]
      for (int j = 0; j < dims; j++) out << SP;
      out << "tensor_" << fNY << "[";
      for (int i = 0; i < dims; i++) {
         out << "(id" << i;
         if (fPads[i].first != 0) out << " + " << fPads[i].first;
         out << ")";
         if (i < dims-1) out << " * (" << outStride[i].GetVal() << ") + ";
      }
      out << "] =\n     tensor_" << fNX << "[";
      for (int i = 0; i < dims; i++) {
         out << "id" << i;
         if (i < dims-1) out << " * (" << inputStride[i].GetVal() << ") + ";
      }
      out << "];\n";
      for (int i = dims-1; i >= 0; i--) {
         for (int j = 1; j < i; j++) out << SP;
         out << "}\n";
      }

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      if (fOutputShape.empty())
         throw std::runtime_error("SOFIE Pad called to Generate_GPU_Kernel_ALPAKA without being initialized first");

      const size_t D = fOutputShape.size(); //dimensions

      auto inputStrides = UTILITY::ComputeStrideFromShape(fInputShape);
      auto outputStrides = UTILITY::ComputeStrideFromShape(fOutputShape);
      opName = "op_" + opName;
      std::string kname = "PadKernel_" + opName;

      std::stringstream cv;
      cv << fConstantValue;

      std::string op;
      op  = "\n//------ PAD_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ input,\n";
      op += SP + SP + SP + "T* __restrict__ output,\n";
      for (auto &p : dynParamNames)
         op += SP + SP + SP + "std::size_t const " + p + ",\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      EmitOutputCoordsFromThreadIdx(op, SP + SP + SP + SP, outputStrides, fOutputShape);
      op += "\n";

      op += SP + SP + SP + SP + "bool interior = true;\n";
      for (std::size_t d = 0; d < D; ++d) {
         std::string hi;
         if (fPads[d].first != 0)
            hi = "(" + std::to_string(fPads[d].first) + " + " + fInputShape[d].GetVal() + ")";
         else
            hi = fInputShape[d].GetVal();
         op += SP + SP + SP + SP + "interior = interior";
         if (fPads[d].first > 0)
            op += " && (out_" + std::to_string(d) + " >= " + std::to_string(fPads[d].first) + "u)";
         op += " && (out_" + std::to_string(d) + " < static_cast<std::size_t>(" + hi + "));\n";
      }
      op += "\n";

      op += SP + SP + SP + SP + "if (interior) {\n";
      op += SP + SP + SP + SP + SP + "std::size_t const input_idx =\n";
      for (std::size_t d = 0; d < D; ++d) {
         std::string lo = std::to_string(fPads[d].first);
         op += SP + SP + SP + SP + SP + SP
               + "(out_" + std::to_string(d) + " - " + lo + "u) * static_cast<std::size_t>("
               + inputStrides[d].GetVal() + ")";
         op += (d + 1 < D) ? " +\n" : ";\n";
      }
      op += SP + SP + SP + SP + SP + "output[elem_idx] = input[input_idx];\n";
      op += SP + SP + SP + SP + "} else {\n";
      op += SP + SP + SP + SP + SP + "output[elem_idx] = static_cast<T>(" + cv.str() + ");\n";
      op += SP + SP + SP + SP + "}\n";

      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "PadKernel_" + opName;
      return SP + kname + " padKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fInputShape.empty() || fOutputShape.empty())
         throw std::runtime_error("SOFIE Pad Op called to Generate without being initialized first");

      std::string totalElements = ConvertDimShapeToLength(fOutputShape);
      std::string kname = "padKernel_" + opName;

      std::stringstream out;
      out << "\n//------ PAD_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{static_cast<Idx>(" << totalElements << ")});\n";
      out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
         << ", " << kname
         << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")";
      for (auto &p : dynParamNames)
         out << ", static_cast<std::size_t>(" << p << ")";
      out << ", static_cast<Idx>(" << totalElements << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";

      return out.str();
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_Pad
