#ifndef SOFIE_ROPERATOR_RWKV_WKV6
#define SOFIE_ROPERATOR_RWKV_WKV6

#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/model/RModel.hxx"
#include "SOFIE/utils/SOFIE_common.hxx"
#include <sstream>
#include <string>

namespace SOFIE {

template <typename T>
class ROperator_RWKV_WKV6 : public ROperator_ALPAKA {
private:
   std::string fNR, fNK, fNV, fNW, fNU, fNY, fNState;

   std::vector<Dim> fShapeR;
   std::string fB, fH, fT, fDh;
   std::string fType;
   std::vector<StridedInputInfo> fInputs; ///< r, k, v, w and u, read through their strides if they are graph inputs

public:
   ROperator_RWKV_WKV6() {}

   ROperator_RWKV_WKV6(const std::string &nameR,
                        const std::string &nameK,
                        const std::string &nameV,
                        const std::string &nameW,
                        const std::string &nameU,
                        const std::string &nameY)
      : fNR(UTILITY::Clean_name(nameR)),
        fNK(UTILITY::Clean_name(nameK)),
        fNV(UTILITY::Clean_name(nameV)),
        fNW(UTILITY::Clean_name(nameW)),
        fNU(UTILITY::Clean_name(nameU)),
        fNY(UTILITY::Clean_name(nameY)),
        fNState("state_wkv6_" + UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::RWKV_WKV6;
      fInputTensorNames  = { fNR, fNK, fNV, fNW, fNU };
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNR))
         throw std::runtime_error("SOFIE RWKV_WKV6: tensor " + fNR + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNU))
         throw std::runtime_error("SOFIE RWKV_WKV6: tensor " + fNU + " not found");

      fShapeR = model.GetDimTensorShape(fNR);
      if (fShapeR.size() != 4)
         throw std::runtime_error("SOFIE RWKV_WKV6: r must be rank-4 [B, H, T, Dh]");
      fInputs.clear();
      for (const auto &in : std::vector<std::pair<std::string, std::string>>{{fNR, "r"}, {fNK, "k"}, {fNV, "v"}, {fNW, "w"}, {fNU, "u"}}) {
         const bool strided = model.IsStridedInputTensor(in.first);
         fInputs.push_back({in.first, in.second, model.GetDimTensorShape(in.first), strided});
         fHasStridedInput |= strided;
      }

      fType = ConvertTypeToString(model.GetTensorType(fNR));
      fB  = fShapeR[0].GetVal();
      fH  = fShapeR[1].GetVal();
      fT  = fShapeR[2].GetVal();
      fDh = fShapeR[3].GetVal();

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNR), fShapeR);
      // State: s[B, H, Dh, Dh]
      model.AddIntermediateTensor(fNState, model.GetTensorType(fNR),
                                  { fShapeR[0], fShapeR[1], fShapeR[3], fShapeR[3] });
      model.AddNeededStdLib("cmath");
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      std::stringstream out;
      out << "\n//---- RWKV_WKV6 " << opName << "\n";
      const std::string B  = fB,  H  = fH,  Ts = fT,  Dh = fDh;
      const std::string X  = "tensor_" + fNR;
      const std::string K  = "tensor_" + fNK;
      const std::string V  = "tensor_" + fNV;
      const std::string W  = "tensor_" + fNW;
      const std::string U  = "tensor_" + fNU;
      const std::string Y  = "tensor_" + fNY;
      const std::string S  = "tensor_" + fNState;

      out << "for (size_t b = 0; b < " << B << "; ++b) {\n";
      out << "for (size_t h = 0; h < " << H << "; ++h) {\n";
      out << "for (size_t t = 0; t < " << Ts << "; ++t) {\n";
      out << "size_t rBase = b*" << H << "*" << Ts << "*" << Dh
          << " + h*" << Ts << "*" << Dh << " + t*" << Dh << ";\n";
      out << "size_t sBase = b*" << H << "*" << Dh << "*" << Dh
          << " + h*" << Dh << "*" << Dh << ";\n";
      out << "size_t uBase = h*" << Dh << ";\n";

      out << "for (size_t j = 0; j < " << Dh << "; ++j) {\n";
      out << fType << " y_j = 0;\n";
      out << "for (size_t i = 0; i < " << Dh << "; ++i) {\n";
      out << fType << " kv_ij = " << K << "[rBase+i] * " << V << "[rBase+j];\n";
      out << "y_j += " << X << "[rBase+i] * (" << S << "[sBase+i*" << Dh << "+j] + std::exp(" << U << "[uBase+i]) * kv_ij);\n";
      out << "}\n";
      out << Y << "[rBase+j] = y_j;\n";
      out << "}\n";
      // State update
      out << "for (size_t i = 0; i < " << Dh << "; ++i) {\n";
      out << fType << " decay_i = std::exp(-std::exp(" << W << "[rBase+i]));\n";
      out << "for (size_t j = 0; j < " << Dh << "; ++j)\n";
      out << S << "[sBase+i*" << Dh << "+j] = decay_i * " << S << "[sBase+i*" << Dh << "+j] + " << K << "[rBase+i] * " << V << "[rBase+j];\n";
      out << "}\n";
      out << "}\n" << "}\n" << "}\n";
      return RewriteCpuStridedReads(opName, fInputs, out.str());
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      return RewriteKernelStridedInputs(GenerateWkv6Kernel(std::move(opName)), fInputs, "std::size_t const Dh");
   }

   std::string GenerateWkv6Kernel(std::string opName) {
      opName = "op_" + opName;
      std::string kname = "WKV6Kernel_" + opName;
      std::string out;
      out  = "\n//------ RWKV_WKV6_KERNEL_ALPAKA\n";
      out += "struct " + kname + " {\n";
      out += "template<typename TAcc, typename T>\n";
      out += "ALPAKA_FN_ACC void operator()(\n";
      out += "TAcc const& acc,\n";
      out += "T const* __restrict__ r,\n";
      out += "T const* __restrict__ k,\n";
      out += "T const* __restrict__ v,\n";
      out += "T const* __restrict__ w,\n";
      out += "T const* __restrict__ u,\n";
      out += "T* __restrict__ y,\n";
      out += "T* __restrict__ state,\n";
      out += "std::size_t const B,\n";
      out += "std::size_t const H,\n";
      out += "std::size_t const T_len,\n";
      out += "std::size_t const Dh) const {\n\n";
      out += "auto const global_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      out += "if (global_idx >= B * H) return;\n";
      out += "std::size_t const b = global_idx / H;\n";
      out += "std::size_t const h = global_idx % H;\n\n";
      out += "for (std::size_t t = 0; t < T_len; ++t) {\n";
      out += "std::size_t const rBase = b*H*T_len*Dh + h*T_len*Dh + t*Dh;\n";
      out += "std::size_t const sBase = b*H*Dh*Dh + h*Dh*Dh;\n";
      out += "std::size_t const uBase = h*Dh;\n";
      // Compute output y
      out += "for (std::size_t j = 0; j < Dh; ++j) {\n";
      out += "T y_j = static_cast<T>(0);\n";
      out += "for (std::size_t i = 0; i < Dh; ++i) {\n";
      out += "T const kv_ij = k[rBase+i] * v[rBase+j];\n";
      out += "y_j += r[rBase+i] * (state[sBase+i*Dh+j] + alpaka::math::exp(acc, u[uBase+i]) * kv_ij);\n";
      out += "}\n";
      out += "y[rBase+j] = y_j;\n";
      out += "}\n";
      // Update state
      out += "for (std::size_t i = 0; i < Dh; ++i) {\n";
      out += "T const decay_i = alpaka::math::exp(acc, -alpaka::math::exp(acc, w[rBase+i]));\n";
      out += "for (std::size_t j = 0; j < Dh; ++j)\n";
      out += "state[sBase+i*Dh+j] = decay_i * state[sBase+i*Dh+j] + k[rBase+i] * v[rBase+j];\n";
      out += "}\n";
      out += "}\n";
      out += "}\n};\n";
      return out;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      return "WKV6Kernel_" + opName + " wkv6Kernel_" + opName + ";\n";
   }

   std::string GenerateInitCode_GPU_ALPAKA() override {
      return "alpaka::memset(queue, deviceBuf_" + fNState + ", 0);\n";
   }

   std::string GenerateResetStateCode_GPU_ALPAKA() override {
      return "alpaka::memset(queue, deviceBuf_" + fNState + ", 0);\n";
   }

   std::vector<std::string> GetPersistentTensorNames_GPU_ALPAKA() const override {
      return {fNState};
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::stringstream out;
      out << "\n//------ RWKV_WKV6_GPU_ALPAKA\n";
      out << "{\n";
      out << StridedLaunchLayouts(opName, fInputs);
      out << "auto const elementsPerGrid_" << opName
          << " = Vec::all(Idx{" << fB << " * " << fH << "});\n";
      out << "auto const workDiv_" << opName
          << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << "auto task_" << opName
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", wkv6Kernel_" << opName << ", "
          << "alpaka::getPtrNative(deviceBuf_" << fNR << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNK << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNV << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNW << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNU << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNState << "), "
          << "static_cast<Idx>(" << fB << "), "
          << "static_cast<Idx>(" << fH << "), "
          << "static_cast<Idx>(" << fT << "), "
          << "static_cast<Idx>(" << fDh << ")" << StridedLayoutArgs(opName, fInputs) << ");\n";
      out << "alpaka::enqueue(queue, task_" << opName << ");\n";
      out << "}\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_RWKV_WKV6
