#include "SOFIE/RGPUModel.hxx"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <dlfcn.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace SOFIE {
namespace {
void Check(const char *error)
{
   if (error)
      throw std::runtime_error(std::string("SOFIE GPU: ") + error);
}

std::size_t RowSize(const std::vector<Dim> &shape)
{
   if (shape.size() < 2)
      throw std::invalid_argument("SOFIE GPU requires a leading batch dimension and feature dimensions");
   std::size_t size = 1;
   for (std::size_t index = 1; index < shape.size(); ++index) {
      if (shape[index].isParam || !shape[index].dim ||
          size > std::numeric_limits<std::size_t>::max() / shape[index].dim)
         throw std::invalid_argument("SOFIE GPU requires fixed, nonzero feature dimensions without overflow");
      size *= shape[index].dim;
   }
   return size;
}

std::string IncludePath(const char *variable, const char *fallback)
{
   const auto *root = std::getenv(variable);
   const auto path = root && *root ? (std::filesystem::path(root) / "include").string() : std::string(fallback);
   if (path.empty() || !std::filesystem::is_directory(path))
      throw std::runtime_error(std::string("Set ") + variable + " to an installed runtime dependency");
   return "-I" + path;
}
}

struct RGPUModel::Program {
   using Create = const char *(*)(void *, int, std::size_t, void **, std::size_t *);
   using Destroy = void (*)(void *);
   using Bind = const char *(*)(void *, void *, std::size_t);
   using Infer = const char *(*)(void *, const void *, void *, std::size_t);
   void *library = nullptr;
   Create create = nullptr;
   Destroy destroy = nullptr;
   Bind bind = nullptr;
   Infer infer = nullptr;
   ~Program() { if (library) dlclose(library); }
};

RGPUModel::RGPUModel(RModel model) : fModel(std::move(model))
{
   if (fModel.GetInputTensorNames().size() != 1 || fModel.GetOutputTensorNames().size() != 1)
      throw std::invalid_argument("SOFIE GPU runtime requires one input and one output");
   fModel.Initialize();
   const auto &input = fModel.GetInputTensorNames().front();
   const auto &output = fModel.GetOutputTensorNames().front();
   const auto type = fModel.GetTensorType(input);
   if ((type != ETensorType::FLOAT && type != ETensorType::FLOAT16) || fModel.GetTensorType(output) != type)
      throw std::invalid_argument("SOFIE Alpaka runtime requires matching FP32 or FP16 input and output");
   fPrecision = type == ETensorType::FLOAT16 ? Precision::Float16 : Precision::Float32;
   if (fModel.ResolveAliasTensor(output) == input)
      throw std::invalid_argument("SOFIE GPU runtime requires a computed output tensor");
   const auto inputShape = fModel.GetDimTensorShape(input);
   const auto outputShape = fModel.GetDimTensorShape(output);
   fInputSize = RowSize(inputShape);
   fOutputSize = RowSize(outputShape);
   if (inputShape.front().isParam) {
      fBatchParam = inputShape.front().param;
      if (!outputShape.front().isParam || outputShape.front().param != fBatchParam)
         throw std::invalid_argument("SOFIE GPU output must preserve the input batch dimension");
   } else {
      fFixedBatch = inputShape.front().dim;
      if (!fFixedBatch || outputShape.front().isParam || outputShape.front().dim != fFixedBatch)
         throw std::invalid_argument("SOFIE GPU input and output batch dimensions must match");
   }
}

void RGPUModel::Compile(const CompileOptions &options)
{
   if (fProgram)
      throw std::logic_error("SOFIE GPU model is already compiled");
   if (options.backend != Backend::CUDA && options.backend != Backend::HIP)
      throw std::invalid_argument("Unsupported SOFIE GPU backend");
   if (options.architecture.empty() || options.architecture.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:+-") != std::string::npos)
      throw std::invalid_argument("SOFIE GPU requires a CUDA/HIP architecture");
   fModel.SetFilename("RuntimeModel");
   fModel.SetExternalWorkspace();
   fModel.GenerateGPU_ALPAKA(Options::kNoWeightFile);
   const bool hip = options.backend == Backend::HIP;
   const std::string api = hip ? "hip" : "cuda";
   const std::string valueType = fPrecision == Precision::Float16 ? "SOFIE::Float16" : "float";
   const std::string viewSuffix = fPrecision == Precision::Float16 ? "H1D" : "F1D";
   std::ostringstream source;
   source << fModel.ReturnGenerated();
   source << "\nusing Generated = SOFIE_RuntimeModel::Session<alpaka::" << (hip ? "TagGpuHipRt" : "TagGpuCudaRt") << ">;\n";
   source << "struct RuntimeSession {\nint device;\nGenerated::QueueAcc queue;\nGenerated model;\n"
          << "RuntimeSession(void* stream, int dev, size_t capacity) : device(dev), queue(static_cast<"
          << api << "Stream_t>(stream)), model(queue, \"\"" << (fBatchParam.empty() ? "" : ", capacity") << ") {}\n};\n";
   source << R"cpp(
template<class Action> const char* Guard(Action action) {
   static thread_local std::string error;
   try { action(); return nullptr; }
   catch (const std::exception& exception) { error = exception.what(); return error.c_str(); }
   catch (...) { error = "unknown generated model error"; return error.c_str(); }
}
)cpp";
   source << "void CheckDevice(int expected) { int device = -1; auto status = " << api
          << "GetDevice(&device); if (status != " << api << "Success) throw std::runtime_error(" << api
          << "GetErrorString(status)); if (device != expected) throw std::runtime_error(\"wrong active GPU device\"); }\n";
   source << "extern \"C\" const char* sofie_create(void* stream, int device, size_t capacity, void** result, size_t* bytes) {\n"
          << "return Guard([&] { CheckDevice(device); auto session = std::make_unique<RuntimeSession>(stream, device, capacity); "
          << "*bytes = session->model.GetWorkspaceSize(); *result = session.release(); }); }\n";
   source << "extern \"C\" void sofie_destroy(void* opaque) { delete static_cast<RuntimeSession*>(opaque); }\n";
   source << "extern \"C\" const char* sofie_bind(void* opaque, void* memory, size_t bytes) { return Guard([&] { "
          << "auto& session = *static_cast<RuntimeSession*>(opaque); CheckDevice(session.device); session.model.SetWorkspace(memory, bytes); }); }\n";
   source << "extern \"C\" const char* sofie_infer(void* opaque, const void* input, void* output, size_t batch) { return Guard([&] {\n"
          << "auto& session = *static_cast<RuntimeSession*>(opaque); CheckDevice(session.device);\n"
          << "Generated::ViewConst" << viewSuffix << " inputView{static_cast<const " << valueType << "*>(input), session.model.devAcc, Generated::Ext1D::all(batch * " << fInputSize << ")};\n"
          << "session.model._infer_impl(" << (fBatchParam.empty() ? "" : "batch, ") << "inputView);\n"
          << "Generated::View" << viewSuffix << " outputView{static_cast<" << valueType << "*>(output), session.model.devAcc, Generated::Ext1D::all(batch * " << fOutputSize << ")};\n"
          << "alpaka::memcpy(session.queue, outputView, session.model.deviceBuf_"
          << fModel.ResolveAliasTensor(fModel.GetOutputTensorNames().front())
          << ", Generated::Ext1D::all(batch * " << fOutputSize << "));\n"
          << "auto status = " << api << "GetLastError(); if (status != " << api << "Success) throw std::runtime_error(" << api << "GetErrorString(status));\n}); }\n";

   std::string pattern = (std::filesystem::temp_directory_path() / "sofie-alpaka-XXXXXX").string();
   if (!mkdtemp(pattern.data()))
      throw std::runtime_error("Cannot create SOFIE compilation directory");
   struct Directory {
      std::filesystem::path path;
      ~Directory() { std::error_code error; std::filesystem::remove_all(path, error); }
   } directory{pattern};
   const auto input = directory.path / "model.cu";
   const auto library = directory.path / "model.so";
   const auto log = directory.path / "compiler.log";
   {
      std::ofstream file(input);
      file << source.str();
      if (!file)
         throw std::runtime_error("Cannot write SOFIE generated source");
   }
   std::vector<std::string> arguments{options.compiler.empty() ? (hip ? "hipcc" : "nvcc") : options.compiler,
      "-std=c++20", "-O2", "-shared",
      IncludePath("SOFIE_ROOT", SOFIE_RUNTIME_INCLUDE_DIR),
      IncludePath("ALPAKA_ROOT", SOFIE_ALPAKA_INCLUDE_DIR),
      IncludePath("SOFIEBLAS_ROOT", SOFIE_BLAS_INCLUDE_DIR)};
   if (hip) {
      arguments.insert(arguments.end(), {"-x", "hip", "-fPIC", "--offload-arch=" + options.architecture,
         "-DALPAKA_ACC_GPU_HIP_ENABLED", "-lhipblaslt", "-lhipblas"});
   } else {
      arguments.insert(arguments.end(), {"-Xcompiler=-fPIC", "--expt-relaxed-constexpr", "--extended-lambda",
         "-arch=" + options.architecture, "-DALPAKA_ACC_GPU_CUDA_ENABLED", "-lcublas", "-lcublasLt"});
   }
   Dl_info coreInfo{};
   if (!dladdr(reinterpret_cast<void *>(&IncludePath), &coreInfo) || !coreInfo.dli_fname)
      throw std::runtime_error("Cannot locate the SOFIE core library for runtime compilation");
   const auto coreLibrary = std::filesystem::absolute(coreInfo.dli_fname);
   arguments.push_back(input.string());
   if (hip) arguments.insert(arguments.end(), {"-x", "none"});
   arguments.insert(arguments.end(), {coreLibrary.string(), "-o", library.string()});
   std::vector<char *> argv;
   for (auto &argument : arguments) argv.push_back(argument.data());
   argv.push_back(nullptr);
   posix_spawn_file_actions_t actions;
   if (posix_spawn_file_actions_init(&actions))
      throw std::runtime_error("Cannot initialize SOFIE compiler process");
   int error = posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
   if (!error) error = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
   pid_t process = 0;
   if (!error) error = posix_spawnp(&process, argv.front(), &actions, nullptr, argv.data(), environ);
   posix_spawn_file_actions_destroy(&actions);
   if (error)
      throw std::runtime_error("Cannot start SOFIE compiler " + arguments.front() + ": " + std::to_string(error));
   int status = 0;
   pid_t waited;
   do { waited = waitpid(process, &status, 0); } while (waited < 0 && errno == EINTR);
   if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status)) {
      std::ifstream file(log);
      std::ostringstream diagnostic;
      diagnostic << file.rdbuf();
      throw std::runtime_error("SOFIE Alpaka compilation failed:\n" + diagnostic.str());
   }
   auto program = std::make_shared<Program>();
   program->library = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
   if (!program->library)
      throw std::runtime_error(std::string("Cannot load SOFIE model: ") + dlerror());
   program->create = reinterpret_cast<Program::Create>(dlsym(program->library, "sofie_create"));
   program->destroy = reinterpret_cast<Program::Destroy>(dlsym(program->library, "sofie_destroy"));
   program->bind = reinterpret_cast<Program::Bind>(dlsym(program->library, "sofie_bind"));
   program->infer = reinterpret_cast<Program::Infer>(dlsym(program->library, "sofie_infer"));
   if (!program->create || !program->destroy || !program->bind || !program->infer)
      throw std::runtime_error("Missing SOFIE generated entry point");
   fProgram = std::move(program);
}

std::unique_ptr<RGPUModel::Session> RGPUModel::CreateSession(void *stream, int device, std::size_t maxBatch) const
{
   if (!fProgram)
      throw std::logic_error("Compile the SOFIE model before creating sessions");
   const auto maxSize = std::max(fInputSize, fOutputSize);
   if (device < 0 || !maxBatch || (fFixedBatch && maxBatch != fFixedBatch) ||
       maxBatch > std::numeric_limits<std::size_t>::max() / (fPrecision == Precision::Float16 ? 2 : 4) / maxSize)
      throw std::invalid_argument("Invalid SOFIE device or batch capacity");
   return std::unique_ptr<Session>(new Session(fProgram, stream, device, maxBatch, fFixedBatch));
}

RGPUModel::Session::Session(std::shared_ptr<Program> program, void *stream, int device,
                           std::size_t maxBatch, std::size_t fixedBatch)
   : fProgram(std::move(program)), fMaxBatch(maxBatch), fFixedBatch(fixedBatch)
{
   Check(fProgram->create(stream, device, maxBatch, &fSession, &fBytes));
}

RGPUModel::Session::~Session() { if (fSession) fProgram->destroy(fSession); }

void RGPUModel::Session::SetWorkspace(void *workspace, std::size_t bytes)
{
   Check(fProgram->bind(fSession, workspace, bytes));
}

void RGPUModel::Session::Infer(const void *input, void *output, std::size_t batch) const
{
   if (!batch) return;
   if (!input || !output || batch > fMaxBatch || (fFixedBatch && batch != fFixedBatch))
      throw std::invalid_argument("Invalid SOFIE inference buffers or batch size");
   Check(fProgram->infer(fSession, input, output, batch));
}
}
