#ifndef SOFIE_ROPERATOR_LeakyRelu
#define SOFIE_ROPERATOR_LeakyRelu

#include "SOFIE/utils/SOFIE_common.hxx"
#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/model/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename T>
class ROperator_LeakyRelu final : public ROperator_ALPAKA
{

private:

   /* Attributes*/
   float falpha=0.01; //default value
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;
   std::string fType;

public:
   ROperator_LeakyRelu(){}
   ROperator_LeakyRelu(float alpha,std::string nameX, std::string nameY):
   falpha(alpha),fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {  
      fKind = OperatorKind::LEAKYRELU;
      if(std::is_same<T, float>::value){
         fType = "float";
      }
		else{
			throw
				std::runtime_error("SOFIE Encountered unsupported type parsing a Leaky Relu operator");
		}

      fInputTensorNames = { fNX };
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE Leaky Relu Op Input Tensor is not found in model");
      }
      fShape = model.GetDimTensorShape(fNX);
      fHasStridedInput = model.IsStridedInputTensor(fNX) && !fShape.empty();
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }

   bool SupportsStridedInput() const override { return true; }



   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator Leaky Relu called to Generate without being initialized first");
      }
      std::stringstream out;
      std::string length = ConvertDimShapeToLength(fShape);

      out << "constexpr float " << OpName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << falpha << ";\n";

      out << "\n//------ LEAKY RELU\n";
      if (fHasStridedInput) {
         out << GenerateStridedUnaryLoop(OpName, fNX, fNY, fShape, [&](const std::string &v) {
            return "((" + v + " >= 0 )? " + v + " : " + OpName + "_alpha * " + v + ")";
         });
         return out.str();
      }
      out << "for (int id = 0; id < " << length << " ; id++){\n";
      out << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] >= 0 )? tensor_" << fNX << "[id] : "<< OpName << "_alpha * tensor_"<< fNX<<"[id]);\n";
      out << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fHasStridedInput)
         return GenerateStridedUnaryKernel(
            "LeakyReluStridedKernel", "LEAKY_RELU",
            [](const std::string &v) { return v + " >= T(0) ? " + v + " : alpha * " + v; }, ", T alpha");
      std::string op;
      op = "\n//------ LEAKY_RELU_KERNEL_ALPAKA\n";
      op += "struct LeakyReluKernel {\n";
      op += "template<typename TAcc, typename T>\n";
      op += "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements, T alpha) const {\n";
      op += "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += "if (idx < numElements) {\n";
      op += "out[idx] = data[idx] >= T(0) ? data[idx] : alpha * data[idx];\n";
      op += "}\n";
      op += "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      if (fHasStridedInput)
         return "LeakyReluStridedKernel leakyReluStridedKernel;\n";
      return "LeakyReluKernel leakyReluKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator LeakyRelu called to Generate without being initialized first");
      }

      std::stringstream out;
      std::string length = ConvertDimShapeToLength(fShape);
      out << "\n//------ LEAKY_RELU_GPU_ALPAKA\n";
      out << "constexpr float " << OpName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << falpha << ";\n";
      if (fHasStridedInput) {
         out << GenerateStridedUnaryLaunch(OpName, "leakyReluStridedKernel", "LEAKY_RELU", fNX, fNY, fShape,
                                           ", " + OpName + "_alpha");
         return out.str();
      }
      out << "auto const elementsPerThread_"<<fNX<<" = Vec::all(static_cast<Idx>(1));\n";
      out << "auto const elementsPerGrid_"<<fNX<<" = Vec::all(Idx{"<< length << "});\n";
      out << "auto const workDiv_" << fNX << " = sofie_workdiv(elementsPerGrid_" << fNX << ");\n";
      out << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNX
         << ", leakyReluKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "), " << OpName << "_alpha);\n";
      out << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }

   /// Alpha accessor — used by the GEMM+LeakyReLU fusion pass.
   float GetAlpha() const { return falpha; }

   bool IsElementwise() const override { return !fHasStridedInput; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return "((" + v + " >= 0) ? " + v + " : " + std::to_string(falpha) + " * " + v + ")";
   }


   std::string GetFusableOutputTensorName() override {
      return fNY;
   }

   void UpdateFusableTensorName(std::string fusable_tensor_name, const std::function<void(const std::string&)>& removal_func){
      removal_func(fNX);
      removal_func(fNY);
      fNX = fusable_tensor_name;
      fNY = fusable_tensor_name;
      fInputTensorNames[0] =  fNX;
      fOutputTensorNames[0] = fNY;
   }

};

}//SOFIE

#endif //SOFIE_ROPERATOR_LeakyRelu
