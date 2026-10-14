#ifndef SOFIE_RGPUMODEL_HXX
#define SOFIE_RGPUMODEL_HXX

#include "SOFIE/RModel.hxx"
#include <cstddef>
#include <memory>
#include <string>

namespace SOFIE {

class RGPUModel {
public:
   enum class Backend { CUDA, HIP };
   enum class Precision { Float32, Float16 };
   struct CompileOptions {
      Backend backend = Backend::CUDA;
      std::string compiler;
      std::string architecture;
   };
   class Session;

   explicit RGPUModel(RModel model);
   std::size_t InputSize() const { return fInputSize; }
   std::size_t OutputSize() const { return fOutputSize; }
   Precision GetPrecision() const { return fPrecision; }
   static constexpr std::size_t WorkspaceAlignment() { return 256; }
   void Compile(const CompileOptions &options);
   std::unique_ptr<Session> CreateSession(void *stream, int device, std::size_t maxBatch) const;

private:
   struct Program;
   RModel fModel;
   Precision fPrecision = Precision::Float32;
   std::size_t fInputSize = 0, fOutputSize = 0, fFixedBatch = 0;
   std::string fBatchParam;
   std::shared_ptr<Program> fProgram;
};

class RGPUModel::Session {
public:
   Session(const Session &) = delete;
   Session &operator=(const Session &) = delete;
   ~Session();
   void SetWorkspace(void *workspace, std::size_t bytes);
   void Infer(const void *input, void *output, std::size_t batch) const;
   std::size_t WorkspaceSize() const { return fBytes; }

private:
   friend class RGPUModel;
   Session(std::shared_ptr<Program> program, void *stream, int device, std::size_t maxBatch,
           std::size_t fixedBatch);
   std::shared_ptr<Program> fProgram;
   void *fSession = nullptr;
   std::size_t fBytes = 0, fMaxBatch = 0, fFixedBatch = 0;
};

}
#endif
