#pragma once

#include "GPUMemoryMonitor.hxx"
#include "GPUProfiler.hxx"
#include "WarmupUtil.hxx"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <filesystem>
#include <iterator>

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {

        if (std::strstr(msg, "Make sure input") != nullptr)
            return;

        if (severity <= Severity::kWARNING)
            std::cerr << "[TensorRT] " << msg << "\n";
    }
};

inline std::size_t TRTTypeSize(nvinfer1::DataType type)
{
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 4;
    }
}

/// Resolve every dynamic dim of @p tensor to a concrete value: the caller's
/// override for that dim's symbolic name (via ITensor::getDimensionName,
/// set by the ONNX parser from the model's dim_param) if one was given,
/// else 1
inline nvinfer1::Dims TRTResolveDims(const nvinfer1::ITensor& tensor,
                                     const std::map<std::string, int64_t>& shape_overrides)
{
    nvinfer1::Dims dims = tensor.getDimensions();
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] > 0)
            continue;
        const char* dimName = tensor.getDimensionName(i);
        auto it = (dimName && dimName[0]) ? shape_overrides.find(dimName)
                                           : shape_overrides.end();
        dims.d[i] = (it != shape_overrides.end()) ? it->second : 1;
    }
    return dims;
}

inline std::size_t TRTDimsElements(const nvinfer1::Dims& dims)
{
    std::size_t n = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0)
            throw std::runtime_error("TensorRT unresolved dynamic dimension.");
        n *= static_cast<std::size_t>(dims.d[i]);
    }
    return n;
}

inline void BenchmarkTRT_GPU(const std::string& onnxPath,
                             const std::string& modelName,
                             int warmup,
                             int iterations,
                             bool profile,
                             const std::map<std::string, int64_t>& shape_overrides = {}){
    namespace fs = std::filesystem;

    TRTLogger logger;
    initLibNvInferPlugins(&logger, "");

    /*
     * Cache the TensorRT engine beside the ONNX model:
     *
     * model.onnx
     * model.trt.plan
     *
     * A normal --tensorrt run builds this file when missing.
     * A --profile run must load an existing plan and must never invoke
     * TensorRT's builder while Nsight Compute is attached.
     */
    fs::path onnxFile(onnxPath);
    fs::path planDir = fs::path("../../cache/tensorrt");
    fs::create_directories(planDir);

    fs::path planPath = planDir / (modelName + ".trt.plan");

    const bool profiling = std::getenv("SOFIE_TRT_PROFILE_ACTIVE") != nullptr;

    std::vector<char> serializedEngine;

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));

    if (!builder) {
        std::cerr << "[TensorRT] Failed to create builder for " << modelName << "\n";
        return;
    }

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));

    if (!network) {
        std::cerr << "[TensorRT] Failed to create network for " << modelName << "\n";
        return;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    if (!parser) {
        std::cerr << "[TensorRT] Failed to create ONNX parser for " << modelName << "\n";
        return;
    }

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "[TensorRT] Unsupported or failed ONNX parse: " << onnxPath << "\n";
        return;
    }

    std::map<std::string, nvinfer1::Dims> resolvedDimsByName;
    bool hasDynamicInput = false;

    for (int i = 0; i < network->getNbInputs(); ++i) {
        nvinfer1::ITensor* input = network->getInput(i);

        if (!input) {
            std::cerr << "[TensorRT] Invalid network input in " << modelName << "\n";
            return;
        }

        const nvinfer1::Dims originalDims = input->getDimensions();
        resolvedDimsByName[input->getName()] = TRTResolveDims(*input, shape_overrides);

        for (int d = 0; d < originalDims.nbDims; ++d) {
            if (originalDims.d[d] <= 0)
                hasDynamicInput = true;
        }
    }

    // Load an existing serialized engine.
    if (fs::exists(planPath)) {
        std::ifstream planFile(planPath, std::ios::binary | std::ios::ate);

        if (!planFile) {
            std::cerr << "[TensorRT] Failed to open engine plan: " << planPath << "\n";
            return;
        }

        const std::streamsize planSize = planFile.tellg();

        if (planSize <= 0) {
            std::cerr << "[TensorRT] Engine plan is empty: " << planPath << "\n";
            return;
        }

        serializedEngine.resize(static_cast<std::size_t>(planSize));

        planFile.seekg(0, std::ios::beg);

        if (!planFile.read(serializedEngine.data(), planSize)) {
            std::cerr << "[TensorRT] Failed to read engine plan: " << planPath << "\n";
            return;
        }
    }

    // Build and save the engine only outside Nsight Compute profiling.
    else {
        if (profiling) {
            std::cerr << "[TensorRT] No cached engine exists for " << modelName << ".\n"
                      << "[TensorRT] Build it before profiling by running:\n"
                      << "  ./sofie_benchmark --tensorrt\n"
                      << "[TensorRT] Expected engine file:\n"
                      << "  " << planPath << "\n";
            return;
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

        if (!config) {
            std::cerr << "[TensorRT] Failed to create builder config for " << modelName << "\n";
            return;
        }

        nvinfer1::IOptimizationProfile* optimizationProfile = builder->createOptimizationProfile();

        if (!optimizationProfile) {
            std::cerr << "[TensorRT] Failed to create optimization profile for " << modelName << "\n";
            return;
        }

        for (int i = 0; i < network->getNbInputs(); ++i) {
            nvinfer1::ITensor* input = network->getInput(i);
            const char* inputName = input->getName();
            const nvinfer1::Dims& resolvedDims = resolvedDimsByName.at(inputName);
            const nvinfer1::Dims originalDims = input->getDimensions();

            bool inputIsDynamic = false;
            for (int d = 0; d < originalDims.nbDims; ++d) {
                if (originalDims.d[d] <= 0)
                    inputIsDynamic = true;
            }

            if (inputIsDynamic) {
                const bool minOk = optimizationProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, resolvedDims);
                const bool optOk = optimizationProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, resolvedDims);
                const bool maxOk = optimizationProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, resolvedDims);

                if (!minOk || !optOk || !maxOk) {
                    std::cerr << "[TensorRT] Failed to set dimensions for input " << inputName << " in " << modelName << "\n";
                    return;
                }
            }
        }

        if (hasDynamicInput) {
            const int profileIndex = config->addOptimizationProfile(optimizationProfile);
            if (profileIndex < 0) {
                std::cerr << "[TensorRT] Failed to add optimization profile for " << modelName << "\n";
                return;
            }
        }

        auto builtPlan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

        if (!builtPlan) {
            std::cerr << "[TensorRT] Failed to build engine for " << modelName << "\n";
            return;
        }

        const auto* planBegin = static_cast<const char*>(builtPlan->data());

        serializedEngine.assign(planBegin, planBegin + builtPlan->size());

        std::ofstream planFile(planPath, std::ios::binary | std::ios::trunc);

        if (!planFile) {
            std::cerr << "[TensorRT] Failed to create engine plan: " << planPath << "\n";
            return;
        }

        planFile.write(serializedEngine.data(), static_cast<std::streamsize>(serializedEngine.size()));

        if (!planFile) {
            std::cerr << "[TensorRT] Failed to write engine plan: " << planPath << "\n";
            return;
        }

        std::cerr << "[TensorRT] Cached engine: " << planPath << "\n";
    }

    // Runtime inference: path used under Nsight Compute.
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    if (!runtime) {
        std::cerr << "[TensorRT] Failed to create runtime for " << modelName << "\n";
        return;
    }

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(serializedEngine.data(), serializedEngine.size()));

    if (!engine) {
        std::cerr << "[TensorRT] Failed to deserialize engine for " << modelName << ".\n"
                  << "[TensorRT] Delete and rebuild this plan if TensorRT, CUDA, the GPU, or the ONNX model changed:\n"
                  << "  " << planPath << "\n";
        return;
    }

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    if (!context) {
        std::cerr << "[TensorRT] Failed to create execution context for " << modelName << "\n";
        return;
    }

    cudaStream_t stream = nullptr;
    cudaError_t cudaStatus = cudaStreamCreate(&stream);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "[TensorRT] cudaStreamCreate failed for " << modelName << ": " << cudaGetErrorString(cudaStatus) << "\n";
        return;
    }

    const int tensorCount = engine->getNbIOTensors();

    // Resolve dynamic input dimensions before calculating buffer sizes.
    for (int i = 0; i < tensorCount; ++i) {
        const char* tensorName = engine->getIOTensorName(i);

        if (engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            const nvinfer1::Dims engineDims = engine->getTensorShape(tensorName);

            bool hasDynamicDimension = false;

            for (int d = 0; d < engineDims.nbDims; ++d) {
                if (engineDims.d[d] <= 0) {
                    hasDynamicDimension = true;
                    break;
                }
            }

            const nvinfer1::Dims& resolvedDims = resolvedDimsByName.at(tensorName);

            if (hasDynamicDimension && !context->setInputShape(tensorName, resolvedDims)) {
                std::cerr << "[TensorRT] Failed to set runtime shape for input " << tensorName << " in " << modelName << "\n";
                cudaStreamDestroy(stream);
                return;
            }
        }
    }

    std::vector<void*> buffers(static_cast<std::size_t>(tensorCount), nullptr);

    auto releaseResources = [&]() {
        for (void* pointer : buffers) {
            if (pointer) cudaFree(pointer);
        }
        if (stream) cudaStreamDestroy(stream);
    };

    for (int i = 0; i < tensorCount; ++i) {
        const char* tensorName = engine->getIOTensorName(i);
        const nvinfer1::Dims dimensions = context->getTensorShape(tensorName);
        const nvinfer1::DataType dataType = engine->getTensorDataType(tensorName);

        std::size_t bytes = 0;

        try {
            bytes = TRTDimsElements(dimensions) * TRTTypeSize(dataType);
        }
        catch (const std::exception& error) {
            std::cerr << "[TensorRT] Cannot allocate tensor " << tensorName << " for " << modelName << ": " << error.what() << "\n";
            releaseResources();
            return;
        }

        cudaStatus = cudaMalloc(&buffers[static_cast<std::size_t>(i)], bytes);

        if (cudaStatus != cudaSuccess) {
            std::cerr << "[TensorRT] cudaMalloc failed for tensor " << tensorName << " in " << modelName << ": " << cudaGetErrorString(cudaStatus) << "\n";
            releaseResources();
            return;
        }

        cudaStatus = cudaMemsetAsync(buffers[static_cast<std::size_t>(i)], 0, bytes, stream);

        if (cudaStatus != cudaSuccess) {
            std::cerr << "[TensorRT] cudaMemsetAsync failed for tensor " << tensorName << " in " << modelName << ": " << cudaGetErrorString(cudaStatus) << "\n";
            releaseResources();
            return;
        }

        if (!context->setTensorAddress(tensorName, buffers[static_cast<std::size_t>(i)])) {
            std::cerr << "[TensorRT] Failed to bind tensor " << tensorName << " in " << modelName << "\n";
            releaseResources();
            return;
        }
    }

    cudaStatus = cudaStreamSynchronize(stream);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "[TensorRT] Initial stream synchronization failed for " << modelName << ": " << cudaGetErrorString(cudaStatus) << "\n";
        releaseResources();
        return;
    }

    if (profile) {
        if (!context->enqueueV3(stream)) {
            std::cerr << "[TensorRT] Priming inference failed for "
                      << modelName << "\n";
            releaseResources();
            return;
        }

        cudaStatus = cudaStreamSynchronize(stream);

        if (cudaStatus != cudaSuccess) {
            std::cerr << "[TensorRT] Priming synchronization failed for "
                      << modelName << ": "
                      << cudaGetErrorString(cudaStatus) << "\n";
            releaseResources();
            return;
        }
    }
    GPUMemoryMonitor gpuMonitor;
    gpuMonitor.Start();

    bool warmupOk = sofie_bench::RunWarmup([&]() {
        if (!context->enqueueV3(stream)) {
            std::cerr << "[TensorRT] Warmup enqueue failed for " << modelName << "\n";
            return false;
        }
        cudaError_t syncStatus = cudaStreamSynchronize(stream);
        if (syncStatus != cudaSuccess) {
            std::cerr << "[TensorRT] Warmup synchronization failed for " << modelName << ": "
                       << cudaGetErrorString(syncStatus) << "\n";
            return false;
        }
        return true;
    }, warmup);

    if (!warmupOk) {
        releaseResources();
        return;
    }

    const int measuredIterations = profile ? 1 : iterations;

    if (profile)
        sofie_bench::StartGpuProfiler();

    const auto start = std::chrono::high_resolution_clock::now();

    bool inferenceSucceeded = true;

    for (int i = 0; i < measuredIterations; ++i) {
        if (!context->enqueueV3(stream)) {
            inferenceSucceeded = false;
            break;
        }
    }

    cudaStatus = cudaStreamSynchronize(stream);

    const auto stop = std::chrono::high_resolution_clock::now();

    if (profile)
        sofie_bench::StopGpuProfiler();

    gpuMonitor.Stop();

    if (!inferenceSucceeded || cudaStatus != cudaSuccess) {
        std::cerr << "[TensorRT] Inference failed for " << modelName;
        if (cudaStatus != cudaSuccess) {
            std::cerr << ": " << cudaGetErrorString(cudaStatus);
        }
        std::cerr << "\n";
        releaseResources();
        return;
    }

    const double averageMs = std::chrono::duration<double, std::milli>(stop - start).count() / static_cast<double>(measuredIterations);
    const double throughput = averageMs > 0.0 ? 1000.0 / averageMs : 0.0;
    const double peakGpuMB = gpuMonitor.PeakMB();

    std::printf(
        "%-50s %12.4f %14s %15s %16.1f "
        "%10s %10s %10s %10s %10s %12.2f\n",
        (modelName + " (TensorRT)").c_str(),
        averageMs,
        "-",
        "-",
        throughput,
        "-",
        "-",
        "-",
        "-",
        "-",
        peakGpuMB);

    const char* resultsPath = std::getenv("SOFIE_TRT_RESULTS");

    if (resultsPath) {
        std::ofstream results(resultsPath, std::ios::app);
        results << modelName << "," << averageMs << "," << throughput << "," << peakGpuMB << "\n";
    }

    releaseResources();
}