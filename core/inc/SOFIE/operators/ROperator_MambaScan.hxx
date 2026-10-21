#ifndef SOFIE_ROPERATOR_MAMBASCAN
#define SOFIE_ROPERATOR_MAMBASCAN

#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/model/RModel.hxx"
#include "SOFIE/utils/SOFIE_common.hxx"
#include <sstream>
#include <string>

namespace SOFIE {

template <typename T>
class ROperator_MambaScan : public ROperator_ALPAKA {
private:
   std::string fNU, fNDelta, fNA, fNB, fNC, fNDbias, fNY, fNState;

   std::vector<Dim> fShapeU;
   std::string fB, fD, fL, fN;
   std::string fType;

   bool fNStatic = false;
   size_t fNValue = 0;

   bool fLStatic = false;
   size_t fLValue = 0;

   // u, delta, A, B, C and D_bias, read through their strides if they are graph inputs
   std::vector<StridedInputInfo> fInputs;

   bool UseParallelGPUScan() const
   {
      return fNStatic && fNValue > 0 && fNValue <= 256;
   }

   size_t GetGPUScanBlockSize() const
   {
      size_t threads = 32;
      while (threads < fNValue)
         threads *= 2;
      return threads;
   }

   bool UseSequenceParallelGPUScan() const
   {
      return fNStatic && fLStatic && fNValue > 0 && fNValue <= 256 && fLValue > 0;
   }

   size_t GetGPUSequenceScanThreadCount() const
   {
      if (fLValue <= 512)
         return 32;
      if (fLValue <= 1024)
         return 64;
      return 128;
   }

   size_t GetGPUSequenceScanItemsPerThread() const
   {
      if (fLValue <= 128)
         return 4;
      if (fLValue <= 256)
         return 8;
      return 16;
   }

public:
   ROperator_MambaScan() {}

   ROperator_MambaScan(const std::string &nameU,
                       const std::string &nameDelta,
                       const std::string &nameA,
                       const std::string &nameB,
                       const std::string &nameC,
                       const std::string &nameDbias,
                       const std::string &nameY)
      : fNU(UTILITY::Clean_name(nameU)),
        fNDelta(UTILITY::Clean_name(nameDelta)),
        fNA(UTILITY::Clean_name(nameA)),
        fNB(UTILITY::Clean_name(nameB)),
        fNC(UTILITY::Clean_name(nameC)),
        fNDbias(UTILITY::Clean_name(nameDbias)),
        fNY(UTILITY::Clean_name(nameY)),
        fNState("state_mamba_" + UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::MAMBA_SCAN;
      fInputTensorNames  = { fNU, fNDelta, fNA, fNB, fNC, fNDbias };
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNU))
         throw std::runtime_error("SOFIE MambaScan: tensor " + fNU + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNA))
         throw std::runtime_error("SOFIE MambaScan: tensor " + fNA + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNB))
         throw std::runtime_error("SOFIE MambaScan: tensor " + fNB + " not found");

      fShapeU = model.GetDimTensorShape(fNU);
      if (fShapeU.size() != 3)
         throw std::runtime_error("SOFIE MambaScan: u must be rank-3 [B, D, L]");
      fInputs.clear();
      for (const auto &in : std::vector<std::pair<std::string, std::string>>{
              {fNU, "u"}, {fNDelta, "delta"}, {fNA, "A"}, {fNB, "B"}, {fNC, "C"}, {fNDbias, "D_bias"}}) {
         const bool strided = model.IsStridedInputTensor(in.first);
         fInputs.push_back({in.first, in.second, model.GetDimTensorShape(in.first), strided});
         fHasStridedInput |= strided;
      }

      auto shapeA = model.GetDimTensorShape(fNA);
      if (shapeA.size() != 2)
         throw std::runtime_error("SOFIE MambaScan: A must be rank-2 [D, N]");

      fNStatic = !shapeA[1].isParam;
      if (fNStatic)
         fNValue = shapeA[1].dim;

      fLStatic = !fShapeU[2].isParam;
      if (fLStatic)
         fLValue = fShapeU[2].dim;

      fType = ConvertTypeToString(model.GetTensorType(fNU));
      fB = fShapeU[0].GetVal();
      fD = fShapeU[1].GetVal();
      fL = fShapeU[2].GetVal();
      fN = shapeA[1].GetVal();

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNU), fShapeU);
      model.AddIntermediateTensor(fNState, model.GetTensorType(fNU),
                                  { fShapeU[0], fShapeU[1], shapeA[1] });
      model.AddNeededStdLib("cmath");
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      std::stringstream out;
      out << "\n//---- MambaScan " << opName << "\n";
      out << "for (size_t b = 0; b < " << fB << "; ++b) {\n";
      out << "for (size_t d = 0; d < " << fD << "; ++d) {\n";
      out << "for (size_t t = 0; t < " << fL << "; ++t) {\n";
      out << fType << " dt  = tensor_" << fNDelta << "[b*" << fD << "*" << fL << " + d*" << fL << " + t];\n";
      out << fType << " u_val = tensor_" << fNU << "[b*" << fD << "*" << fL << " + d*" << fL << " + t];\n";
      out << "for (size_t n = 0; n < " << fN << "; ++n) {\n";
      out << fType << " dA = std::exp(dt * tensor_" << fNA << "[d*" << fN << " + n]);\n";
      out << fType << " dBu = dt * tensor_" << fNB << "[b*" << fN << "*" << fL << " + n*" << fL << " + t] * u_val;\n";
      out << "tensor_" << fNState << "[b*" << fD << "*" << fN << " + d*" << fN << " + n] = "
          << "dA * tensor_" << fNState << "[b*" << fD << "*" << fN << " + d*" << fN << " + n] + dBu;\n";
      out << "}\n";
      out << fType << " y_val = tensor_" << fNDbias << "[d] * u_val;\n";
      out << "for (size_t n = 0; n < " << fN << "; ++n)\n";
      out << "y_val += tensor_" << fNC << "[b*" << fN << "*" << fL << " + n*" << fL << " + t] * tensor_"
          << fNState << "[b*" << fD << "*" << fN << " + d*" << fN << " + n];\n";
      out << "tensor_" << fNY << "[b*" << fD << "*" << fL << " + d*" << fL << " + t] = y_val;\n";
      out << "}\n" << "}\n" << "}\n";
      return RewriteCpuStridedReads(opName, fInputs, out.str());
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      return RewriteKernelStridedInputs(GenerateMambaKernel(std::move(opName)), fInputs, "std::size_t const N");
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      // the layouts of the strided inputs are the last arguments of the kernel call
      std::string code = GenerateMambaLaunch(opName);
      const std::string last = "static_cast<Idx>(" + fN + "));";
      const std::string layoutArgs = StridedLayoutArgs("op_" + opName, fInputs);
      for (size_t pos = code.find(last); pos != std::string::npos; pos = code.find(last, pos + 1))
         code.replace(pos, last.size(), "static_cast<Idx>(" + fN + ")" + layoutArgs + ");");
      return StridedLaunchLayouts("op_" + opName, fInputs) + code;
   }

   std::string GenerateMambaKernel(std::string opName) {
      opName = "op_" + opName;

      if (UseSequenceParallelGPUScan()) {
         const size_t threadCount = GetGPUSequenceScanThreadCount();
         const size_t itemsPerThread = GetGPUSequenceScanItemsPerThread();
         const size_t chunkSize = threadCount * itemsPerThread;
         const std::string kname = "MambaScanSequenceKernel_" + opName;
         std::string out;

         out = "\n//------ MAMBA_SCAN_SEQUENCE_KERNEL_ALPAKA\n";
         out += "struct " + kname + " {\n";
         out += "template<typename TAcc, typename T>\n";
         out += "ALPAKA_FN_ACC void operator()(\n";
         out += "TAcc const& acc,\n";
         out += "T const* __restrict__ u,\n";
         out += "T const* __restrict__ delta,\n";
         out += "T const* __restrict__ A,\n";
         out += "T const* __restrict__ B,\n";
         out += "T const* __restrict__ C,\n";
         out += "T const* __restrict__ D_bias,\n";
         out += "T* __restrict__ y,\n";
         out += "T* __restrict__ state,\n";
         out += "std::size_t const B_dim,\n";
         out += "std::size_t const D_dim,\n";
         out += "std::size_t const L,\n";
         out += "std::size_t const N) const {\n\n";

         out += "constexpr std::size_t kThreads = " + std::to_string(threadCount) + "u;\n";
         out += "constexpr std::size_t kItems = " + std::to_string(itemsPerThread) + "u;\n";
         out += "constexpr std::size_t kChunkSize = " + std::to_string(chunkSize) + "u;\n\n";

         out += "auto const block_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
         out += "auto const thread_idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
         out += "if (block_idx >= B_dim * D_dim) return;\n\n";

         out += "std::size_t const b = block_idx / D_dim;\n";
         out += "std::size_t const d = block_idx % D_dim;\n\n";

         out += "auto& scanA = alpaka::declareSharedVar<T[" + std::to_string(2 * threadCount) + "], 0>(acc);\n";
         out += "auto& scanB = alpaka::declareSharedVar<T[" + std::to_string(2 * threadCount) + "], 1>(acc);\n\n";

         out += "T dt_vals[kItems];\n";
         out += "T delta_u_vals[kItems];\n";
         out += "T out_vals[kItems];\n\n";

         out += "for (std::size_t chunk_start = 0; chunk_start < L; chunk_start += kChunkSize) {\n";
         out += "for (std::size_t i = 0; i < kItems; ++i) {\n";
         out += "std::size_t const t = chunk_start + thread_idx * kItems + i;\n";
         out += "if (t < L) {\n";
         out += "T const dt = delta[b * D_dim * L + d * L + t];\n";
         out += "T const u_val = u[b * D_dim * L + d * L + t];\n";
         out += "dt_vals[i] = dt;\n";
         out += "delta_u_vals[i] = dt * u_val;\n";
         out += "out_vals[i] = D_bias[d] * u_val;\n";
         out += "} else {\n";
         out += "dt_vals[i] = static_cast<T>(0);\n";
         out += "delta_u_vals[i] = static_cast<T>(0);\n";
         out += "out_vals[i] = static_cast<T>(0);\n";
         out += "}\n";
         out += "}\n\n";

         out += "for (std::size_t n = 0; n < N; ++n) {\n";
         out += "T prefixA[kItems];\n";
         out += "T prefixB[kItems];\n";
         out += "T localA = static_cast<T>(1);\n";
         out += "T localB = static_cast<T>(0);\n";
         out += "T const A_val = A[d * N + n];\n";
         out += "std::size_t const state_idx = b * D_dim * N + d * N + n;\n";
         out += "T const initial_state = state[state_idx];\n\n";

         out += "for (std::size_t i = 0; i < kItems; ++i) {\n";
         out += "std::size_t const t = chunk_start + thread_idx * kItems + i;\n";
         out += "T stepA = static_cast<T>(1);\n";
         out += "T stepB = static_cast<T>(0);\n";
         out += "if (t < L) {\n";
         out += "stepA = alpaka::math::exp(acc, dt_vals[i] * A_val);\n";
         out += "stepB = B[b * N * L + n * L + t] * delta_u_vals[i];\n";
         out += "}\n";
         out += "localB = stepB + stepA * localB;\n";
         out += "localA = stepA * localA;\n";
         out += "prefixA[i] = localA;\n";
         out += "prefixB[i] = localB;\n";
         out += "}\n\n";

         out += "scanA[thread_idx] = localA;\n";
         out += "scanB[thread_idx] = localB;\n";
         out += "alpaka::syncBlockThreads(acc);\n\n";

         out += "std::size_t src = 0;\n";
         out += "std::size_t dst = kThreads;\n";
         out += "for (std::size_t offset = 1; offset < kThreads; offset <<= 1) {\n";
         out += "T const currentA = scanA[src + thread_idx];\n";
         out += "T const currentB = scanB[src + thread_idx];\n";
         out += "T nextA = currentA;\n";
         out += "T nextB = currentB;\n";
         out += "if (thread_idx >= offset) {\n";
         out += "T const previousA = scanA[src + thread_idx - offset];\n";
         out += "T const previousB = scanB[src + thread_idx - offset];\n";
         out += "nextA = currentA * previousA;\n";
         out += "nextB = currentB + currentA * previousB;\n";
         out += "}\n";
         out += "scanA[dst + thread_idx] = nextA;\n";
         out += "scanB[dst + thread_idx] = nextB;\n";
         out += "alpaka::syncBlockThreads(acc);\n";
         out += "std::size_t const tmp = src;\n";
         out += "src = dst;\n";
         out += "dst = tmp;\n";
         out += "}\n\n";

         out += "T thread_state = initial_state;\n";
         out += "if (thread_idx > 0) {\n";
         out += "T const previousA = scanA[src + thread_idx - 1];\n";
         out += "T const previousB = scanB[src + thread_idx - 1];\n";
         out += "thread_state = previousA * initial_state + previousB;\n";
         out += "}\n\n";

         out += "for (std::size_t i = 0; i < kItems; ++i) {\n";
         out += "std::size_t const t = chunk_start + thread_idx * kItems + i;\n";
         out += "if (t < L) {\n";
         out += "T const state_val = prefixA[i] * thread_state + prefixB[i];\n";
         out += "out_vals[i] += C[b * N * L + n * L + t] * state_val;\n";
         out += "}\n";
         out += "}\n\n";

         out += "if (thread_idx == 0) {\n";
         out += "T const totalA = scanA[src + kThreads - 1];\n";
         out += "T const totalB = scanB[src + kThreads - 1];\n";
         out += "state[state_idx] = totalA * initial_state + totalB;\n";
         out += "}\n";
         out += "alpaka::syncBlockThreads(acc);\n";
         out += "}\n\n";

         out += "for (std::size_t i = 0; i < kItems; ++i) {\n";
         out += "std::size_t const t = chunk_start + thread_idx * kItems + i;\n";
         out += "if (t < L)\n";
         out += "y[b * D_dim * L + d * L + t] = out_vals[i];\n";
         out += "}\n";
         out += "}\n";
         out += "}\n";
         out += "};\n";

         return out;
      }

      if (UseParallelGPUScan()) {
         const size_t blockSize = GetGPUScanBlockSize();
         const std::string kname = "MambaScanParallelKernel_" + opName;
         std::string out;

         out  = "\n//------ MAMBA_SCAN_PARALLEL_KERNEL_ALPAKA\n";
         out += "struct " + kname + " {\n";
         out += "template<typename TAcc, typename T>\n";
         out += "ALPAKA_FN_ACC void operator()(\n";
         out += "TAcc const& acc,\n";
         out += "T const* __restrict__ u,\n";
         out += "T const* __restrict__ delta,\n";
         out += "T const* __restrict__ A,\n";
         out += "T const* __restrict__ B,\n";
         out += "T const* __restrict__ C,\n";
         out += "T const* __restrict__ D_bias,\n";
         out += "T* __restrict__ y,\n";
         out += "T* __restrict__ state,\n";
         out += "std::size_t const B_dim,\n";
         out += "std::size_t const D_dim,\n";
         out += "std::size_t const L,\n";
         out += "std::size_t const N) const {\n\n";

         out += "auto const block_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
         out += "auto const thread_idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
         out += "if (block_idx >= B_dim * D_dim) return;\n\n";

         out += "std::size_t const b = block_idx / D_dim;\n";
         out += "std::size_t const d = block_idx % D_dim;\n";
         out += "bool const active = thread_idx < N;\n\n";

         out += "auto& reduction = alpaka::declareSharedVar<T[" + std::to_string(blockSize) + "], 0>(acc);\n";
         out += "std::size_t const state_idx = b * D_dim * N + d * N + thread_idx;\n";
         out += "T state_val = active ? state[state_idx] : static_cast<T>(0);\n\n";

         out += "for (std::size_t t = 0; t < L; ++t) {\n";
         out += "T const dt = delta[b * D_dim * L + d * L + t];\n";
         out += "T const u_val = u[b * D_dim * L + d * L + t];\n";
         out += "T contribution = static_cast<T>(0);\n\n";

         out += "if (active) {\n";
         out += "T const dA = alpaka::math::exp(acc, dt * A[d * N + thread_idx]);\n";
         out += "T const dBu = dt * B[b * N * L + thread_idx * L + t] * u_val;\n";
         out += "state_val = dA * state_val + dBu;\n";
         out += "contribution = C[b * N * L + thread_idx * L + t] * state_val;\n";
         out += "}\n\n";

         out += "reduction[thread_idx] = contribution;\n";
         out += "alpaka::syncBlockThreads(acc);\n\n";

         out += "for (std::size_t stride = " + std::to_string(blockSize / 2) + "u; stride > 0; stride >>= 1) {\n";
         out += "if (thread_idx < stride)\n";
         out += "reduction[thread_idx] += reduction[thread_idx + stride];\n";
         out += "alpaka::syncBlockThreads(acc);\n";
         out += "}\n\n";

         out += "if (thread_idx == 0)\n";
         out += "y[b * D_dim * L + d * L + t] = D_bias[d] * u_val + reduction[0];\n";

         out += "}\n\n";

         out += "if (active)\n";
         out += "state[state_idx] = state_val;\n";
         out += "}\n";
         out += "};\n";

         return out;
      }

      std::string kname = "MambaScanKernel_" + opName;
      std::string out;
      out  = "\n//------ MAMBA_SCAN_KERNEL_ALPAKA\n";
      out += "struct " + kname + " {\n";
      out += "template<typename TAcc, typename T>\n";
      out += "ALPAKA_FN_ACC void operator()(\n";
      out += "TAcc const& acc,\n";
      out += "T const* __restrict__ u,\n";
      out += "T const* __restrict__ delta,\n";
      out += "T const* __restrict__ A,\n";
      out += "T const* __restrict__ B,\n";
      out += "T const* __restrict__ C,\n";
      out += "T const* __restrict__ D_bias,\n";
      out += "T* __restrict__ y,\n";
      out += "T* __restrict__ state,\n";
      out += "std::size_t const B_dim,\n";
      out += "std::size_t const D_dim,\n";
      out += "std::size_t const L,\n";
      out += "std::size_t const N) const {\n\n";
      out += "auto const global_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      out += "if (global_idx >= B_dim * D_dim) return;\n";
      out += "std::size_t const b = global_idx / D_dim;\n";
      out += "std::size_t const d = global_idx % D_dim;\n\n";
      out += "for (std::size_t t = 0; t < L; ++t) {\n";
      out += "T const dt     = delta[b*D_dim*L + d*L + t];\n";
      out += "T const u_val  = u[b*D_dim*L + d*L + t];\n";
      out += "for (std::size_t n = 0; n < N; ++n) {\n";
      out += "T const dA  = alpaka::math::exp(acc, dt * A[d*N + n]);\n";
      out += "T const dBu = dt * B[b*N*L + n*L + t] * u_val;\n";
      out += "state[b*D_dim*N + d*N + n] = dA * state[b*D_dim*N + d*N + n] + dBu;\n";
      out += "}\n";
      out += "T y_val = D_bias[d] * u_val;\n";
      out += "for (std::size_t n = 0; n < N; ++n)\n";
      out += "y_val += C[b*N*L + n*L + t] * state[b*D_dim*N + d*N + n];\n";
      out += "y[b*D_dim*L + d*L + t] = y_val;\n";
      out += "}\n";
      out += "}\n};\n";
      return out;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;

      if (UseSequenceParallelGPUScan())
         return "MambaScanSequenceKernel_" + opName + " mambaSequenceKernel_" + opName + ";\n";

      if (UseParallelGPUScan())
         return "MambaScanParallelKernel_" + opName + " mambaParallelKernel_" + opName + ";\n";

      return "MambaScanKernel_" + opName + " mambaKernel_" + opName + ";\n";
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

   std::string GenerateMambaLaunch(std::string opName) {
      opName = "op_" + opName;

      if (UseSequenceParallelGPUScan()) {
         const size_t threadCount = GetGPUSequenceScanThreadCount();

         std::stringstream out;
         out << "\n//------ MAMBA_SCAN_SEQUENCE_GPU_ALPAKA\n";
         out << "{\n";
         out << "alpaka::memset(queue, deviceBuf_" << fNState << ", 0);\n";
         out << "auto const blocksPerGrid = Vec::all(static_cast<Idx>(" << fB << " * " << fD << "));\n";
         out << "auto const threadsPerBlock = Vec::all(static_cast<Idx>(" << threadCount << "));\n";
         out << "auto const elementsPerThread = Vec::all(static_cast<Idx>(1));\n";
         out << "auto const workDiv_" << opName << " = alpaka::WorkDivMembers<Dim, Idx>(blocksPerGrid, threadsPerBlock, elementsPerThread);\n";
         out << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
             << ", mambaSequenceKernel_" << opName << ", "
             << "alpaka::getPtrNative(deviceBuf_" << fNU << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNDelta << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNA << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNB << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNC << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNDbias << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNState << "), "
             << "static_cast<Idx>(" << fB << "), "
             << "static_cast<Idx>(" << fD << "), "
             << "static_cast<Idx>(" << fL << "), "
             << "static_cast<Idx>(" << fN << "));\n";
         out << "alpaka::enqueue(queue, task_" << opName << ");\n";
         out << "}\n";

         return out.str();
      }

      if (UseParallelGPUScan()) {
         const size_t blockSize = GetGPUScanBlockSize();

         std::stringstream out;
         out << "\n//------ MAMBA_SCAN_PARALLEL_GPU_ALPAKA\n";
         out << "{\n";
         out << "alpaka::memset(queue, deviceBuf_" << fNState << ", 0);\n";
         out << "auto const blocksPerGrid = Vec::all(static_cast<Idx>(" << fB << " * " << fD << "));\n";
         out << "auto const threadsPerBlock = Vec::all(static_cast<Idx>(" << blockSize << "));\n";
         out << "auto const elementsPerThread = Vec::all(static_cast<Idx>(1));\n";
         out << "auto const workDiv_" << opName << " = alpaka::WorkDivMembers<Dim, Idx>(blocksPerGrid, threadsPerBlock, elementsPerThread);\n";
         out << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
             << ", mambaParallelKernel_" << opName << ", "
             << "alpaka::getPtrNative(deviceBuf_" << fNU << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNDelta << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNA << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNB << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNC << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNDbias << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fNState << "), "
             << "static_cast<Idx>(" << fB << "), "
             << "static_cast<Idx>(" << fD << "), "
             << "static_cast<Idx>(" << fL << "), "
             << "static_cast<Idx>(" << fN << "));\n";
         out << "alpaka::enqueue(queue, task_" << opName << ");\n";
         out << "}\n";

         return out.str();
      }

      std::stringstream out;
      out << "\n//------ MAMBA_SCAN_GPU_ALPAKA\n";
      out << "{\n";
      out << "alpaka::memset(queue, deviceBuf_" << fNState << ", 0);\n";
      out << "auto const elementsPerGrid_" << opName
          << " = Vec::all(Idx{" << fB << " * " << fD << "});\n";
      out << "auto const workDiv_" << opName
          << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << "auto task_" << opName
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", mambaKernel_" << opName << ", "
          << "alpaka::getPtrNative(deviceBuf_" << fNU << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNDelta << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNA << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNB << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNC << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNDbias << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNState << "), "
          << "static_cast<Idx>(" << fB << "), "
          << "static_cast<Idx>(" << fD << "), "
          << "static_cast<Idx>(" << fL << "), "
          << "static_cast<Idx>(" << fN << "));\n";
      out << "alpaka::enqueue(queue, task_" << opName << ");\n";
      out << "}\n";
      return out.str();
   }

};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_MAMBASCAN
