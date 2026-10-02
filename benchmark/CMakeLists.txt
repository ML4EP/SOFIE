cmake_minimum_required(VERSION 3.18)
include(FetchContent)

################################################################################
# SOFIE Alpaka Benchmark Toolkit
#
# Usage:
#   cmake -Bbuild -DSOFIE_BENCHMARK=ON .
#   cmake --build build --target sofie_benchmark
#   cd build && ./bin/sofie_benchmark [options]
#
# To also benchmark with ONNX Runtime GPU:
#   cmake -Bbuild -DSOFIE_BENCHMARK=ON -DSOFIE_BENCHMARK_ORT=ON \
#         [-DONNXRUNTIME_ROOT=/usr/local/onnxruntime] .
#   ./bin/sofie_benchmark --onnxruntime
#
# Place .onnx models in benchmark/models/ and re-run cmake to register them.
# sofie_benchmark lands in build/bin next to every other SOFIE binary (see
# CMAKE_RUNTIME_OUTPUT_DIRECTORY in the top-level CMakeLists.txt) — there is
# no separate benchmark/build tree to configure or hunt through.
################################################################################

option(SOFIE_BENCHMARK_ORT
    "Also benchmark ONNX Runtime GPU alongside SOFIE (requires ORT ≥ 1.18)"
    OFF)

option(SOFIE_BENCHMARK_TRT "Enable TensorRT benchmark comparison" OFF)

option(SOFIE_BENCHMARK_AOT "Enable PyTorch AOTInductor benchmark comparison" OFF)

option(SOFIE_BENCHMARK_MAMBA_AOT "Enable Mamba custom-op support for AOTInductor" OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH
        "Optional ONNX Runtime installation root")

set(TENSORRT_ROOT "" CACHE PATH
        "Optional TensorRT installation root")

set(MAMBA_SELECTIVE_SCAN_LIBRARY "" CACHE FILEPATH
        "Optional path to the selective_scan_cuda shared library used by Mamba AOT")

option(SOFIE_BENCHMARK_PROFILE
    "Enable per-operator GPU profiling (per-op timing + memory breakdown). \
Mutually exclusive with throughput benchmarking — rebuild without this flag \
to measure throughput."
    OFF)

option(SOFIE_BENCHMARK_LOWRANK
    "Emit all benchmark models with low rank factorization enabled \
(Options::kLowRankFactorize), so the same benchmark suite can be compared \
dense vs low-rank across two builds. Only affects the GPU codegen emitted \
for each model, not the sofie_benchmark runner itself."
    OFF)

set(SOFIE_BENCHMARK_LOWRANK_RATIO "0.5" CACHE STRING
    "Rank ratio used when SOFIE_BENCHMARK_LOWRANK is ON: rank = ratio * min(rows, cols) \
for each eligible Gemm weight matrix.")

################################################################################
# Backend / architecture selection
#
# SOFIE_BENCHMARK_BACKEND applies to both throughput benchmarking and profiling.
# Currently only CUDA is supported; setting any other value is a hard error.
################################################################################

set(SOFIE_BENCHMARK_BACKEND "CUDA" CACHE STRING
    "Target accelerator backend for SOFIE benchmark and profiling (currently only CUDA)")
set_property(CACHE SOFIE_BENCHMARK_BACKEND PROPERTY STRINGS CUDA)

string(TOUPPER "${SOFIE_BENCHMARK_BACKEND}" _bench_backend)

if(NOT _bench_backend STREQUAL "CUDA")
    message(FATAL_ERROR
        "SOFIE Benchmark: SOFIE_BENCHMARK_BACKEND='${SOFIE_BENCHMARK_BACKEND}' is not "
        "supported.  Only 'CUDA' is currently implemented.  "
        "AMD GPU (HIP/ROCm) support is planned for a future release.")
endif()

# Compile-time defines propagated to every benchmark translation unit.
# These drive BenchmarkBackend.hxx type aliases and SOFIE_BENCH_DEVICE_SYNC().
set(_SOFIE_BENCH_ALPAKA_DEFINE ALPAKA_ACC_GPU_CUDA_ENABLED)
set(_SOFIE_BENCH_BACKEND_DEFINE SOFIE_BACKEND_CUDA)

if(SOFIE_BENCHMARK_PROFILE)
    message(STATUS "SOFIE Benchmark: profiling ENABLED  "
                   "(backend = ${SOFIE_BENCHMARK_BACKEND}, throughput benchmarking disabled)")
else()
    message(STATUS "SOFIE Benchmark: backend = ${SOFIE_BENCHMARK_BACKEND}")
endif()

if(SOFIE_BENCHMARK_LOWRANK)
    message(STATUS "SOFIE Benchmark: low rank factorization ENABLED for all emitted models "
                   "(ratio = ${SOFIE_BENCHMARK_LOWRANK_RATIO}). Rebuild with -DSOFIE_BENCHMARK_LOWRANK=OFF "
                   "and re-run to compare against the dense baseline.")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../parsers/inc
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

################################################################################
# Discover models
################################################################################

set(SOFIE_BENCHMARK_MODEL_DIR "models" CACHE STRING
        "Directory containing ONNX benchmark models. Relative paths are resolved under benchmark/.")

if(IS_ABSOLUTE "${SOFIE_BENCHMARK_MODEL_DIR}")
    set(BENCHMARK_MODEL_DIR "${SOFIE_BENCHMARK_MODEL_DIR}")
else()
    set(BENCHMARK_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${SOFIE_BENCHMARK_MODEL_DIR}")
endif()

file(GLOB BENCHMARK_ONNX_MODELS "${BENCHMARK_MODEL_DIR}/*.onnx")

################################################################################
# Dynamic-shape ("parametric") SOFIE model families
#
# These base models keep symbolic ONNX input dimensions
# which SOFIE's GPU/Alpaka code generator supports end-to-end: the
# generated Session constructor and infer() take the shape parameters as
# runtime size_t arguments, so ONE generated/compiled model serves
# every input size instead of a separately-generated-and-compiled model per
# shape. sofie_benchmark_emitter therefore generates SOFIE code for each of
# these base models exactly once (see EnsureBaseFamilyGenerated() in
# src/BenchmarkEmitter.cxx.in) and every shape variant below reuses it,
# supplying only its own concrete shape values at construction/inference time.

set(SOFIE_BENCHMARK_DYNAMIC_FAMILY_STEMS
    gnn_large
    mlpf_fp32_fused
    mlpf_fp32_unfused
    gnn_h32_k2
    gnn_h64_k4
    transformer_d32_h2_L6_ff32
    transformer_d64_h4_L6_ff128
    punet_h32_k2_heads4_layers2
    punet_h64_k4_heads4_layers2
)

set(_dyn_family_regex "")
foreach(_stem ${SOFIE_BENCHMARK_DYNAMIC_FAMILY_STEMS})
    if(_dyn_family_regex)
        set(_dyn_family_regex "${_dyn_family_regex}|${_stem}")
    else()
        set(_dyn_family_regex "${_stem}")
    endif()
endforeach()

# Exclude the base/template models with symbolic dimensions from the model
# list itself
list(FILTER BENCHMARK_ONNX_MODELS EXCLUDE REGEX "/(${_dyn_family_regex})\\.onnx$")

################################################################################
# Shape sweep for the dynamic families 
# Each entry below is a string using that family's
# actual ONNX dim_param names: GNN and
# PUNet families use n_nodes/n_edges (edges = 5x nodes, matching how these
# graphs were generated); MLPF uses num_batch/num_elements (batch pinned to
# 1); the Transformers use n_nodes for the source sequence length and
# seq_length for the target.
################################################################################

function(sofie_benchmark_sweep_n_e OUT_VAR)
    set(_sweep "")
    foreach(_n 100 300 1000 3000 10000 30000 100000)
        math(EXPR _e "${_n} * 5")
        list(APPEND _sweep "n_nodes=${_n},n_edges=${_e}")
    endforeach()
    set(${OUT_VAR} "${_sweep}" PARENT_SCOPE)
endfunction()

function(sofie_benchmark_sweep_mlpf OUT_VAR)
    set(_sweep "")
    foreach(_n 128 256 512 1024 2048 4096)
        list(APPEND _sweep "num_batch=1,num_elements=${_n}")
    endforeach()
    set(${OUT_VAR} "${_sweep}" PARENT_SCOPE)
endfunction()

function(sofie_benchmark_sweep_transformer OUT_VAR SEQ_LENS)
    set(_sweep "")
    foreach(_s ${SEQ_LENS})
        list(APPEND _sweep "n_nodes=${_s},seq_length=${_s}")
    endforeach()
    set(${OUT_VAR} "${_sweep}" PARENT_SCOPE)
endfunction()

sofie_benchmark_sweep_n_e(_sweep_gnn_h32_k2)
sofie_benchmark_sweep_n_e(_sweep_gnn_h64_k4)
sofie_benchmark_sweep_n_e(_sweep_gnn_large)
sofie_benchmark_sweep_n_e(_sweep_punet_h32_k2_heads4_layers2)
sofie_benchmark_sweep_n_e(_sweep_punet_h64_k4_heads4_layers2)
sofie_benchmark_sweep_mlpf(_sweep_mlpf_fp32_fused)
sofie_benchmark_sweep_mlpf(_sweep_mlpf_fp32_unfused)
sofie_benchmark_sweep_transformer(_sweep_transformer_d32_h2_L6_ff32 "10;20;30;40;50;60")
sofie_benchmark_sweep_transformer(_sweep_transformer_d64_h4_L6_ff128 "10;20;30;40;50;60;64;128;256;512;1024;2048")

set(SOFIE_BENCHMARK_DEFAULT_RESULTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/results")

if(NOT BENCHMARK_ONNX_MODELS)
    message(STATUS
        "SOFIE Benchmark: No .onnx models found in ${BENCHMARK_MODEL_DIR}. "
        "Add ONNX models there and re-run cmake to enable benchmarking.")
    return()
endif()

list(LENGTH BENCHMARK_ONNX_MODELS N_MODELS)
message(STATUS "SOFIE Benchmark: Found ${N_MODELS} model(s) in ${BENCHMARK_MODEL_DIR}")

FetchContent_Declare(
    sofieBLAS
    GIT_REPOSITORY https://github.com/ML4EP/sofieBLAS
    GIT_TAG        dev
)
FetchContent_MakeAvailable(sofieBLAS)

FetchContent_Declare(
    alpaka
    GIT_REPOSITORY https://github.com/alpaka-group/alpaka
    GIT_TAG        2fa91a34ed11b2076e474c5507d920e85cf9b79d
)
FetchContent_MakeAvailable(alpaka)

################################################################################
# Hardware toolkit setup — CUDA
################################################################################

enable_language(CUDA)
find_library(NVML_LIBRARY NAMES nvidia-ml REQUIRED)
message(STATUS "SOFIE Benchmark: NVML = ${NVML_LIBRARY}")
find_package(CUDAToolkit REQUIRED)
message(STATUS "SOFIE Benchmark: CUDA toolkit ${CUDAToolkit_VERSION}")

################################################################################
# NVML — background GPU SM/memory occupancy sampling during the timed
# inference loop (see src/GpuOccupancy.hxx). Ships as a stub library
# alongside libcuda.so in the CUDA toolkit; the real libnvidia-ml.so.1 is
# resolved at runtime from the NVIDIA driver install, same pattern as the
# CUDA driver stub.
################################################################################

find_library(SOFIE_NVML_LIBRARY
    NAMES nvidia-ml
    PATHS "${CUDAToolkit_LIBRARY_DIR}/stubs" "${CUDAToolkit_LIBRARY_DIR}"
)

if(SOFIE_NVML_LIBRARY)
    message(STATUS "SOFIE Benchmark: NVML found — ${SOFIE_NVML_LIBRARY} (GPU occupancy sampling enabled)")
else()
    message(WARNING
        "SOFIE Benchmark: NVML (libnvidia-ml) not found — GPU occupancy "
        "sampling will be disabled; SM%/MemBw% columns will print N/A.")
endif()


set(SOFIE_ORT_FOUND FALSE)

if(SOFIE_BENCHMARK_ORT)

    set(_ort_search_roots "")
    if(ONNXRUNTIME_ROOT)
        list(APPEND _ort_search_roots "${ONNXRUNTIME_ROOT}")
    endif()
    list(APPEND _ort_search_roots
        /usr/local/onnxruntime /usr/local /usr /opt)

    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS ${_ort_search_roots}
        PATH_SUFFIXES include include/onnxruntime
        NO_DEFAULT_PATH)

    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS ${_ort_search_roots}
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH)

    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        set(SOFIE_ORT_FOUND TRUE)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION             "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
        set(SOFIE_ORT_TARGET onnxruntime::onnxruntime)
        message(STATUS "SOFIE Benchmark: ONNX Runtime found — ${ONNXRUNTIME_LIBRARY}")
        message(STATUS "SOFIE Benchmark: ORT headers  — ${ONNXRUNTIME_INCLUDE_DIR}")
    else()
        message(WARNING
            "SOFIE Benchmark: SOFIE_BENCHMARK_ORT=ON but ONNX Runtime "
            "not found.  Set -DONNXRUNTIME_ROOT=<path> or install ORT.  "
            "ORT benchmarking will be disabled.")
    endif()
endif()

if(SOFIE_BENCHMARK_ORT AND NOT SOFIE_ORT_FOUND)
    message(STATUS "SOFIE Benchmark: ORT benchmarking disabled (library not found)")
endif()

if(SOFIE_BENCHMARK_TRT)
    set(_trt_search_roots "")
    if(TENSORRT_ROOT)
        list(APPEND _trt_search_roots "${TENSORRT_ROOT}")
    endif()

    find_path(TENSORRT_INCLUDE_DIR
            NAMES NvInfer.h
            HINTS ${_trt_search_roots}
            PATH_SUFFIXES include include/x86_64-linux-gnu)

    find_library(NVINFER_LIBRARY
            NAMES nvinfer
            HINTS ${_trt_search_roots}
            PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

    find_library(NVONNXPARSER_LIBRARY
            NAMES nvonnxparser
            HINTS ${_trt_search_roots}
            PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

    find_library(NVINFER_PLUGIN_LIBRARY
            NAMES nvinfer_plugin
            HINTS ${_trt_search_roots}
            PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

    if(NOT TENSORRT_INCLUDE_DIR OR
            NOT NVINFER_LIBRARY OR
            NOT NVONNXPARSER_LIBRARY OR
            NOT NVINFER_PLUGIN_LIBRARY)
        message(FATAL_ERROR
                "SOFIE Benchmark: TensorRT was requested but could not be found. "
                "Install TensorRT or set -DTENSORRT_ROOT=<path>.")
    endif()

    get_filename_component(TENSORRT_LIBRARY_DIR "${NVINFER_LIBRARY}" DIRECTORY)

    message(STATUS "SOFIE Benchmark: TensorRT headers = ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "SOFIE Benchmark: TensorRT nvinfer = ${NVINFER_LIBRARY}")
endif()

if(SOFIE_BENCHMARK_AOT)
    find_package(Torch REQUIRED)
    message(STATUS "SOFIE Benchmark: PyTorch AOTInductor comparison enabled")

    ############################################################################
    # Real-architecture source checkouts for torch_models/*.py
    ############################################################################

    set(SOFIE_TICL_REPO "$ENV{HOME}/TICL-GNN-Trackster-Linking" CACHE PATH
        "Local checkout of TICL-GNN-Trackster-Linking (cloned automatically here if missing)")
    set(SOFIE_PARTICLEFLOW_REPO "$ENV{HOME}/.cache/sofie_benchmark/particleflow_src" CACHE PATH
        "Local checkout of particleflow (cloned automatically here if missing, \
pinned to a known-good commit; its training checkpoint under checkpoints/ is \
not part of the repo and must still be placed there by hand)")
    set(SOFIE_PARTICLEFLOW_GIT_TAG "8edc986f61004e8cfdf1a8a1ddfe79334a903ee4" CACHE STRING
        "particleflow commit to clone — must match the wiring the MLPF checkpoint/.onnx were validated against")

    if(NOT EXISTS "${SOFIE_TICL_REPO}/.git")
        message(STATUS "SOFIE Benchmark AOT: cloning TICL-GNN-Trackster-Linking into ${SOFIE_TICL_REPO} ...")
        FetchContent_Declare(ticl_gnn_trackster_linking
            GIT_REPOSITORY https://github.com/cms-patatrack/TICL-GNN-Trackster-Linking.git
            GIT_TAG        main
            SOURCE_DIR     "${SOFIE_TICL_REPO}"
        )
        FetchContent_Populate(ticl_gnn_trackster_linking)
    endif()

    if(NOT EXISTS "${SOFIE_PARTICLEFLOW_REPO}/.git")
        message(STATUS "SOFIE Benchmark AOT: cloning particleflow (${SOFIE_PARTICLEFLOW_GIT_TAG}) into ${SOFIE_PARTICLEFLOW_REPO} ...")
        FetchContent_Declare(particleflow
            GIT_REPOSITORY https://github.com/jpata/particleflow.git
            GIT_TAG        ${SOFIE_PARTICLEFLOW_GIT_TAG}
            SOURCE_DIR     "${SOFIE_PARTICLEFLOW_REPO}"
        )
        FetchContent_Populate(particleflow)
    endif()

    if(NOT EXISTS "${SOFIE_PARTICLEFLOW_REPO}/checkpoints/mlpf_model_kwargs.pkl")
        message(WARNING
            "SOFIE Benchmark AOT: ${SOFIE_PARTICLEFLOW_REPO}/checkpoints/ does "
            "not have the MLPF training checkpoint yet (mlpf_model_kwargs.pkl, "
            "mlpf_checkpoint-10-3.812332.pth). It isn't fetched automatically — "
            "place both files there by hand before running export_aot.py for "
            "mlpf_fp32_fused/mlpf_fp32_unfused.")
    endif()
endif()

################################################################################
# Build per-model strings for configure_file
################################################################################

set(_EMIT_BLOCK
"try {\n\
    EmitBenchmarkModel(\"@1@\", \"@2@\", outDir, \"@3@\", \"@4@\", @5@);\n\
} catch (const std::exception &e) {\n\
    std::cerr << \"[ERROR] @2@: \" << e.what() << \"\\n\";\n\
    ++failures;\n\
} catch (...) {\n\
    std::cerr << \"[ERROR] @2@: unknown exception\\n\";\n\
    ++failures;\n\
}\n\
")

set(BENCHMARK_EMIT_CAPTURES        "")
set(BENCHMARK_BENCH_HEADERS        "")
set(BENCHMARK_AOT_META_HEADERS     "")
set(BENCHMARK_FWD_DECLS            "")
set(BENCHMARK_SINGLE_MODEL_CASES   "")
set(BENCHMARK_SPAWN_CALLS          "")
set(GENERATED_HEADERS              "")
set(BENCHMARK_MODEL_CU_SRCS        "")
set(DYNAMIC_BASE_ONNX_FILES        "")

macro(sofie_benchmark_emit_entry _E_ONNX_FILE _E_MODEL_NAME _E_BASE_ONNX_FILE _E_BASE_MODEL_NAME _E_SHAPE_OVERRIDES_CPP)
    set(MODEL_AOT_ENABLED TRUE)
    if(_E_MODEL_NAME MATCHES "^mambav2_" AND NOT SOFIE_BENCHMARK_MAMBA_AOT)
        set(MODEL_AOT_ENABLED FALSE)
    endif()

    string(REGEX REPLACE "[^A-Za-z0-9]" "_" MODEL_CPPNAME "${_E_MODEL_NAME}")

    if(_E_BASE_ONNX_FILE)
        # Dynamic-shape variant: SOFIE code is shared, generated once from
        # _E_BASE_ONNX_FILE (see sofie_benchmark_emitter / EnsureBaseFamilyGenerated).
        set(GEN_HXX "${CMAKE_CURRENT_BINARY_DIR}/${_E_BASE_MODEL_NAME}_GPU_ALPAKA.hxx")
        list(FIND GENERATED_HEADERS "${GEN_HXX}" _gen_hxx_idx)
        if(_gen_hxx_idx EQUAL -1)
            list(APPEND GENERATED_HEADERS "${GEN_HXX}")
        endif()
        list(FIND DYNAMIC_BASE_ONNX_FILES "${_E_BASE_ONNX_FILE}" _base_onnx_idx)
        if(_base_onnx_idx EQUAL -1)
            list(APPEND DYNAMIC_BASE_ONNX_FILES "${_E_BASE_ONNX_FILE}")
        endif()
    else()
        set(GEN_HXX "${CMAKE_CURRENT_BINARY_DIR}/${_E_MODEL_NAME}_GPU_ALPAKA.hxx")
        list(APPEND GENERATED_HEADERS "${GEN_HXX}")
    endif()
    set(GEN_BENCH    "${CMAKE_CURRENT_BINARY_DIR}/${_E_MODEL_NAME}_bench.hxx")
    set(GEN_AOT_META "${CMAKE_CURRENT_BINARY_DIR}/${_E_MODEL_NAME}_aot_meta.hxx")
    list(APPEND GENERATED_HEADERS "${GEN_BENCH}" "${GEN_AOT_META}")

    string(REPLACE "@1@" "${_E_ONNX_FILE}"      _emit_cap "${_EMIT_BLOCK}")
    string(REPLACE "@2@" "${_E_MODEL_NAME}"     _emit_cap "${_emit_cap}")
    string(REPLACE "@3@" "${_E_BASE_ONNX_FILE}" _emit_cap "${_emit_cap}")
    string(REPLACE "@4@" "${_E_BASE_MODEL_NAME}" _emit_cap "${_emit_cap}")
    string(REPLACE "@5@" "${_E_SHAPE_OVERRIDES_CPP}" _emit_cap "${_emit_cap}")
    string(APPEND BENCHMARK_EMIT_CAPTURES "${_emit_cap}")

    string(APPEND BENCHMARK_BENCH_HEADERS
        "#include \"${_E_MODEL_NAME}_bench.hxx\"\n")

    string(APPEND BENCHMARK_AOT_META_HEADERS
            "#include \"${_E_MODEL_NAME}_aot_meta.hxx\"\n")

    # Forward declaration for the main TU (function defined in per-model .cu)
    string(APPEND BENCHMARK_FWD_DECLS
            "void Benchmark_${MODEL_CPPNAME}(int warmup, int iterations, const std::string& weightsDir, bool profile);\n")

    # Single-model dispatch: one if-branch per model (used in --single-model mode)
    string(APPEND BENCHMARK_SINGLE_MODEL_CASES
        "    if (model == \"${MODEL_CPPNAME}\") {\n"
        "        if (run_sofie)\n"
        "            Benchmark_${MODEL_CPPNAME}(warmup, iterations, weightsDir, profile);\n"
        "#ifdef SOFIE_BENCHMARK_ORT\n"
        "        if (run_ort)\n"
        "            BenchmarkORT_GPU(\"${_E_ONNX_FILE}\", \"${_E_MODEL_NAME}\", warmup, iterations, profile, ${_E_SHAPE_OVERRIDES_CPP});\n"
        "#endif\n"
        "#ifdef SOFIE_BENCHMARK_TRT\n"
        "        if (run_trt)\n"
        "            BenchmarkTRT_GPU(\"${_E_ONNX_FILE}\", \"${_E_MODEL_NAME}\", warmup, iterations, profile, ${_E_SHAPE_OVERRIDES_CPP});\n"
        "#endif\n"
    )

    if(MODEL_AOT_ENABLED)
        string(APPEND BENCHMARK_SINGLE_MODEL_CASES
                "#ifdef SOFIE_BENCHMARK_AOT\n"
                "        if (run_aot)\n"
                "            BenchmarkAOT_GPU(\"${BENCHMARK_MODEL_DIR}/aot_models/${_E_MODEL_NAME}.pt2\","
                " \"${_E_MODEL_NAME}\", GetAOTInputSpecs_${MODEL_CPPNAME}(), warmup, iterations, profile);\n"
                "#endif\n"
        )
    endif()

    string(APPEND BENCHMARK_SINGLE_MODEL_CASES
            "        return 0;\n"
            "    }\n"
    )

    # Subprocess spawn: run this binary as a child with --single-model <name>.
    # Each child gets a fresh CUDA context so GPU memory from the previous
    # model is completely freed before the next one allocates.
    string(APPEND BENCHMARK_SPAWN_CALLS
            "    {\n"

            "        auto RunBackend = [&](const std::string& flag,\n"
            "                              const std::string& suffix)\n"
            "        {\n"
            "            std::string cmd;\n"

            "            if (profile) {\n"
            "                const char* resultsDir = std::getenv(\"SOFIE_BENCHMARK_RESULTS_DIR\");\n"

            "                cmd = ncuPath;\n"
            "                cmd += \" --target-processes all\";\n"
            "                cmd += \" --profile-from-start off\";\n"
            "                cmd += \" --quiet\";\n"
            "                cmd += \" --csv\";\n"
            "                cmd += \" --force-overwrite\";\n"
            "                cmd += \" --export \";\n"
            "                cmd += std::string(resultsDir);\n"
            "                cmd += \"/\" + suffix + \"/${_E_MODEL_NAME}\";\n"
            "                cmd += \" --section LaunchStats\";\n"
            "                cmd += \" --section Occupancy\";\n"
            "                cmd += \" --section MemoryWorkloadAnalysis\";\n"
            "                cmd += \" --section SpeedOfLight\";\n"
            "                cmd += \" \";\n"
            "                cmd += std::string(argv[0]);\n"
            "            }\n"
            "            else {\n"
            "                cmd = std::string(argv[0]);\n"
            "            }\n"

            "            cmd += \" --single-model ${MODEL_CPPNAME}\";\n"
            "            cmd += \" \";\n"
            "            cmd += flag;\n"
            "            cmd += commonArgs;\n"

            "            int rc = std::system(cmd.c_str());\n"

            "            if (rc != 0) {\n"
            "                std::fprintf(stderr,\n"
            "                    \"[ERROR] ${_E_MODEL_NAME} (%s): subprocess exited %d — skipping, moving on.\\n\",\n"
            "                    suffix.c_str(), rc);\n"
            "                ++totalFailed;\n"
            "            }\n"
            "        };\n"

            "        RunBackend(\"--sofie-only\", \"sofie\");\n"

            "#ifdef SOFIE_BENCHMARK_ORT\n"
            "        if (run_ort)\n"
            "            RunBackend(\"--ort-only\", \"ort\");\n"
            "#endif\n"

            "#ifdef SOFIE_BENCHMARK_TRT\n"
            "        if (run_trt)\n"
            "            RunBackend(\"--trt-only\", \"tensorrt\");\n"
            "#endif\n"
    )

    if(MODEL_AOT_ENABLED)
        string(APPEND BENCHMARK_SPAWN_CALLS
                "#ifdef SOFIE_BENCHMARK_AOT\n"
                "        if (run_aot)\n"
                "            RunBackend(\"--aot-only\", \"pytorch_aot\");\n"
                "#endif\n"
        )
    endif()

    string(APPEND BENCHMARK_SPAWN_CALLS "    }\n")

    # Per-model compilation unit: include one bench.hxx → one .cu file.
    # ModelBench.cu.in's own placeholder is "@MODEL_NAME@" — this
    # configure_file(@ONLY) substitutes it from whatever variable is named
    # exactly MODEL_NAME in scope, not from this macro's own parameter
    set(MODEL_NAME "${_E_MODEL_NAME}")
    set(_model_cu "${CMAKE_CURRENT_BINARY_DIR}/${_E_MODEL_NAME}_bench.cu")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ModelBench.cu.in"
        "${_model_cu}"
        @ONLY
    )
    list(APPEND BENCHMARK_MODEL_CU_SRCS "${_model_cu}")
endmacro()

# static models discovered on disk
foreach(ONNX_FILE ${BENCHMARK_ONNX_MODELS})
    get_filename_component(MODEL_NAME "${ONNX_FILE}" NAME_WE)
    sofie_benchmark_emit_entry("${ONNX_FILE}" "${MODEL_NAME}" "" "" "{}")
endforeach()

# Dynamic families: one entry per swept shape, all sharing the family's one
# .onnx file
foreach(_stem ${SOFIE_BENCHMARK_DYNAMIC_FAMILY_STEMS})
    set(_base_onnx "${BENCHMARK_MODEL_DIR}/${_stem}.onnx")
    foreach(_point ${_sweep_${_stem}})
        string(REPLACE "," ";" _point_pairs "${_point}")
        set(_name_suffix "")
        set(_overrides_cpp "{")
        set(_first TRUE)
        foreach(_pair ${_point_pairs})
            string(REPLACE "=" ";" _kv "${_pair}")
            list(GET _kv 0 _k)
            list(GET _kv 1 _v)
            string(APPEND _name_suffix "_${_k}${_v}")
            if(NOT _first)
                string(APPEND _overrides_cpp ",")
            endif()
            string(APPEND _overrides_cpp "{\"${_k}\",${_v}}")
            set(_first FALSE)
        endforeach()
        string(APPEND _overrides_cpp "}")
        string(REGEX REPLACE "_n_nodes([0-9]+)_n_edges([0-9]+)" "_n\\1_e\\2" _name_suffix "${_name_suffix}")
        string(REGEX REPLACE "_num_batch([0-9]+)_num_elements([0-9]+)" "_b\\1_n\\2" _name_suffix "${_name_suffix}")
        string(REGEX REPLACE "_n_nodes([0-9]+)_seq_length([0-9]+)" "_n\\1_s\\2" _name_suffix "${_name_suffix}")

        sofie_benchmark_emit_entry("${_base_onnx}" "${_stem}${_name_suffix}" "${_base_onnx}" "${_stem}" "${_overrides_cpp}")
    endforeach()
endforeach()

list(LENGTH BENCHMARK_MODEL_CU_SRCS N_MODELS)

################################################################################
# Configure emitter and runner sources
################################################################################

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/BenchmarkEmitter.cxx.in"
    "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkEmitter_all.cxx"
    @ONLY
)

# Main runner: only main() + forward declarations (no model headers).
# Each model is compiled in its own _bench.cu TU
set(RUNNER_SRC "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkRunner_main.cpp")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/BenchmarkRunner.cxx.in"
    "${RUNNER_SRC}"
    @ONLY
)

################################################################################
# Emitter executable (plain C++, generates SOFIE headers at build time)
################################################################################

add_executable(sofie_benchmark_emitter
    "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkEmitter_all.cxx"
)

target_include_directories(sofie_benchmark_emitter PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../core/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../parsers/inc"
)

target_link_libraries(sofie_benchmark_emitter PRIVATE
    SOFIE_core
    SOFIE_parsers
)

target_compile_options(sofie_benchmark_emitter PRIVATE
    -Wno-unused-parameter
    -Wno-array-bounds
)

target_compile_definitions(sofie_benchmark_emitter PRIVATE
    $<$<BOOL:${SOFIE_BENCHMARK_PROFILE}>:SOFIE_BENCHMARK_PROFILE>
    $<$<BOOL:${SOFIE_BENCHMARK_LOWRANK}>:SOFIE_BENCHMARK_LOWRANK>
    $<$<BOOL:${SOFIE_BENCHMARK_LOWRANK}>:SOFIE_BENCHMARK_LOWRANK_RATIO=${SOFIE_BENCHMARK_LOWRANK_RATIO}>
)


add_custom_command(
    OUTPUT  ${GENERATED_HEADERS}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND "$<TARGET_FILE:sofie_benchmark_emitter>" "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS sofie_benchmark_emitter ${BENCHMARK_ONNX_MODELS} ${DYNAMIC_BASE_ONNX_FILES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "SOFIE Benchmark: generating headers for ${N_MODELS} model(s)..."
    VERBATIM
)

add_custom_target(sofie_benchmark_headers
    DEPENDS ${GENERATED_HEADERS}
)

set_source_files_properties(${BENCHMARK_MODEL_CU_SRCS} PROPERTIES LANGUAGE CUDA)

set_source_files_properties("${RUNNER_SRC}" PROPERTIES LANGUAGE CXX)

add_executable(sofie_benchmark "${RUNNER_SRC}" ${BENCHMARK_MODEL_CU_SRCS})

if(SOFIE_BENCHMARK_MAMBA_AOT)
    if(NOT SOFIE_BENCHMARK_AOT)
        message(FATAL_ERROR "SOFIE_BENCHMARK_MAMBA_AOT requires SOFIE_BENCHMARK_AOT=ON")
    endif()

    find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)

    target_sources(sofie_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/MambaAOTCustomOps.cxx")

    get_filename_component(_torch_cmake_dir "${Torch_DIR}" DIRECTORY)
    get_filename_component(_torch_share_dir "${_torch_cmake_dir}" DIRECTORY)
    get_filename_component(_torch_package_dir "${_torch_share_dir}" DIRECTORY)
    get_filename_component(_torch_site_packages "${_torch_package_dir}" DIRECTORY)

    if(MAMBA_SELECTIVE_SCAN_LIBRARY)
        if(NOT EXISTS "${MAMBA_SELECTIVE_SCAN_LIBRARY}")
            message(FATAL_ERROR
                    "SOFIE Benchmark: MAMBA_SELECTIVE_SCAN_LIBRARY does not exist: "
                    "${MAMBA_SELECTIVE_SCAN_LIBRARY}")
        endif()
    else()
        file(GLOB MAMBA_SELECTIVE_SCAN_LIBS
                "${_torch_site_packages}/selective_scan_cuda*${CMAKE_SHARED_LIBRARY_SUFFIX}")
        list(LENGTH MAMBA_SELECTIVE_SCAN_LIBS _mamba_selective_scan_count)

        if(NOT _mamba_selective_scan_count EQUAL 1)
            message(FATAL_ERROR
                    "SOFIE Benchmark: could not uniquely locate selective_scan_cuda next to PyTorch. "
                    "Set -DMAMBA_SELECTIVE_SCAN_LIBRARY=/path/to/selective_scan_cuda.so")
        endif()

        list(GET MAMBA_SELECTIVE_SCAN_LIBS 0 MAMBA_SELECTIVE_SCAN_LIBRARY)
    endif()

    get_filename_component(MAMBA_SELECTIVE_SCAN_DIR "${MAMBA_SELECTIVE_SCAN_LIBRARY}" DIRECTORY)
    get_filename_component(MAMBA_SELECTIVE_SCAN_NAME "${MAMBA_SELECTIVE_SCAN_LIBRARY}" NAME)

    target_link_directories(sofie_benchmark PRIVATE "${MAMBA_SELECTIVE_SCAN_DIR}")
    target_link_libraries(sofie_benchmark PRIVATE "-Wl,-l:${MAMBA_SELECTIVE_SCAN_NAME}")
    set_property(TARGET sofie_benchmark APPEND PROPERTY BUILD_RPATH
            "${MAMBA_SELECTIVE_SCAN_DIR};${_torch_package_dir}/lib")

    message(STATUS "SOFIE Benchmark: Mamba AOT custom-op support enabled")
    message(STATUS "SOFIE Benchmark: selective_scan_cuda = ${MAMBA_SELECTIVE_SCAN_LIBRARY}")
endif()

add_dependencies(sofie_benchmark sofie_benchmark_headers)

target_include_directories(sofie_benchmark PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${alpaka_SOURCE_DIR}/include"
    "${sofieblas_SOURCE_DIR}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

if(SOFIE_BENCHMARK_TRT)
    target_include_directories(sofie_benchmark PRIVATE "${TENSORRT_INCLUDE_DIR}")
    set_property(TARGET sofie_benchmark APPEND PROPERTY BUILD_RPATH "${TENSORRT_LIBRARY_DIR}")
endif()


if(NOT DEFINED SOFIE_BENCHMARK_CUDA_ARCH OR SOFIE_BENCHMARK_CUDA_ARCH STREQUAL "")
    if(CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(SOFIE_BENCHMARK_CUDA_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(SOFIE_BENCHMARK_CUDA_ARCH "75")   # RTX 2070 SUPER default
    endif()
endif()
message(STATUS "SOFIE Benchmark: CUDA architectures = ${SOFIE_BENCHMARK_CUDA_ARCH}")

set_target_properties(sofie_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES          "${SOFIE_BENCHMARK_CUDA_ARCH}"
    CUDA_STANDARD               20
    CUDA_STANDARD_REQUIRED      ON
)

target_compile_definitions(sofie_benchmark PRIVATE
    ${_SOFIE_BENCH_ALPAKA_DEFINE}
    ${_SOFIE_BENCH_BACKEND_DEFINE}
    ALPAKA_HAS_STD_ATOMIC_REF
    $<$<BOOL:${SOFIE_ORT_FOUND}>:SOFIE_BENCHMARK_ORT>
    $<$<BOOL:${SOFIE_BENCHMARK_TRT}>:SOFIE_BENCHMARK_TRT>
    $<$<BOOL:${SOFIE_BENCHMARK_AOT}>:SOFIE_BENCHMARK_AOT>
    $<$<BOOL:${SOFIE_BENCHMARK_PROFILE}>:SOFIE_BENCHMARK_PROFILE>
    $<$<BOOL:${SOFIE_NVML_LIBRARY}>:SOFIE_BENCH_HAVE_NVML>
)

target_compile_options(sofie_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --use_fast_math
        -O1
        -Wno-deprecated-gpu-targets
        -diag-suppress 177
        --maxrregcount=64
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O2
        -fPIC
    >
)

find_package(Threads REQUIRED)

target_link_libraries(sofie_benchmark PRIVATE
    SOFIE_core
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    Threads::Threads
    ${NVML_LIBRARY}
    $<$<BOOL:${SOFIE_ORT_FOUND}>:${SOFIE_ORT_TARGET}>
    $<$<BOOL:${SOFIE_BENCHMARK_TRT}>:${NVINFER_LIBRARY}>
    $<$<BOOL:${SOFIE_BENCHMARK_TRT}>:${NVONNXPARSER_LIBRARY}>
    $<$<BOOL:${SOFIE_BENCHMARK_TRT}>:${NVINFER_PLUGIN_LIBRARY}>
    $<$<BOOL:${SOFIE_BENCHMARK_AOT}>:${TORCH_LIBRARIES}>
    $<$<BOOL:${SOFIE_BENCHMARK_MAMBA_AOT}>:Python3::Python>
)
if(SOFIE_ORT_FOUND)
    message(STATUS "SOFIE Benchmark: target 'sofie_benchmark' configured "
                   "(${N_MODELS} model(s), CUDA backend + ORT-GPU)")
else()
    message(STATUS "SOFIE Benchmark: target 'sofie_benchmark' configured "
                   "(${N_MODELS} model(s), CUDA backend; "
                   "re-configure with -DSOFIE_BENCHMARK_ORT=ON for ORT comparison)")
endif()

if(testing)
    add_test(
        NAME    SofieBenchmark
        COMMAND sofie_benchmark --warmup 5 --iterations 20
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

################################################################################
# Large-input benchmark
#
# Usage:
#   cmake -Bbuild . -DSOFIE_BENCHMARK=ON -DSOFIE_BENCHMARK_LARGE=ON \
#         [-DSOFIE_BENCHMARK_LARGE_CUDA_ARCH=80]   # 80=A100, 90=H100
#   cmake --build build --target sofie_benchmark_large
################################################################################

option(SOFIE_BENCHMARK_LARGE
    "Build sofie_benchmark_large for cluster GPUs (A100/H100, ≥40 GB VRAM)"
    OFF)

if(SOFIE_BENCHMARK_LARGE)

    set(SOFIE_BENCHMARK_LARGE_CUDA_ARCH "80" CACHE STRING
        "CUDA SM architecture(s) for sofie_benchmark_large (e.g. 80=A100, 90=H100)")

    set(LARGE_MODEL_NAMES
        # ── Transformers (OOM on 8 GB) ──────────────────────────────────────
        "transformer_L100_B100"
        "transformer_L1000_B100"
        "transformer_L8000_B1"
    )

    set(_large_regex "")
    foreach(_m ${LARGE_MODEL_NAMES})
        if(_large_regex)
            set(_large_regex "${_large_regex}|${_m}")
        else()
            set(_large_regex "${_m}")
        endif()
    endforeach()
    set(_large_regex "(${_large_regex})\\.onnx$")

    file(GLOB _ALL_LARGE_ONNX "${CMAKE_CURRENT_SOURCE_DIR}/models/*.onnx")
    list(FILTER _ALL_LARGE_ONNX INCLUDE REGEX "${_large_regex}")

    if(NOT _ALL_LARGE_ONNX)
        message(WARNING
            "SOFIE_BENCHMARK_LARGE=ON but none of the expected large-input .onnx "
            "files were found in benchmark/models/.  Add them there, then "
            "re-run cmake.")
    else()
        list(LENGTH _ALL_LARGE_ONNX N_LARGE)
        message(STATUS
            "SOFIE Benchmark Large: ${N_LARGE} large-input model(s), "
            "CUDA arch = ${SOFIE_BENCHMARK_LARGE_CUDA_ARCH}")

        set(LARGE_EMIT_CAPTURES      "")
        set(LARGE_FWD_DECLS          "")
        set(LARGE_SINGLE_MODEL_CASES "")
        set(LARGE_SPAWN_CALLS        "")
        set(LARGE_GENERATED_HEADERS  "")
        set(LARGE_MODEL_CU_SRCS      "")

        # Every LARGE_MODEL_NAMES entry is a genuinely-static, standalone
        # model (no symbolic ONNX dims)
        foreach(ONNX_FILE ${_ALL_LARGE_ONNX})
            get_filename_component(MODEL_NAME "${ONNX_FILE}" NAME_WE)
            string(REGEX REPLACE "[^A-Za-z0-9]" "_" MODEL_CPPNAME "${MODEL_NAME}")

            set(GEN_HXX "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_GPU_ALPAKA.hxx")
            list(APPEND LARGE_GENERATED_HEADERS "${GEN_HXX}")
            set(GEN_BENCH    "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_bench.hxx")
            set(GEN_AOT_META "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_aot_meta.hxx")
            list(APPEND LARGE_GENERATED_HEADERS "${GEN_BENCH}" "${GEN_AOT_META}")

            string(REPLACE "@1@" "${ONNX_FILE}"  _ec "${_EMIT_BLOCK}")
            string(REPLACE "@2@" "${MODEL_NAME}" _ec "${_ec}")
            string(REPLACE "@3@" ""              _ec "${_ec}")
            string(REPLACE "@4@" ""              _ec "${_ec}")
            string(REPLACE "@5@" "{}"            _ec "${_ec}")
            string(APPEND LARGE_EMIT_CAPTURES "${_ec}")

            string(APPEND BENCHMARK_FWD_DECLS
                    "void Benchmark_${MODEL_CPPNAME}(int warmup, int iterations, const std::string& weightsDir, bool profile);\n")

            string(APPEND LARGE_SINGLE_MODEL_CASES
                "    if (model == \"${MODEL_CPPNAME}\") {\n"
                "        if (run_sofie)\n"
                "            Benchmark_${MODEL_CPPNAME}(warmup, iterations, weightsDir, profile);\n"
                "#ifdef SOFIE_BENCHMARK_ORT\n"
                "        if (run_ort) BenchmarkORT_GPU(\"${ONNX_FILE}\", \"${MODEL_NAME}\", warmup, iterations);\n"
                "#endif\n"
                "        return 0;\n"
                "    }\n"
            )

            string(APPEND LARGE_SPAWN_CALLS
                "    {\n"
                "        std::string cmd = std::string(argv[0])\n"
                "            + \" --single-model ${MODEL_CPPNAME}\"\n"
                "            + commonArgs;\n"
                "        int rc = std::system(cmd.c_str());\n"
                "        if (rc != 0) {\n"
                "            std::fprintf(stderr, \"[ERROR] ${MODEL_NAME}: subprocess exited %d\\n\", rc);\n"
                "            ++totalFailed;\n"
                "        }\n"
                "    }\n"
            )

            set(_model_cu "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_bench.cu")
            configure_file(
                "${CMAKE_CURRENT_SOURCE_DIR}/src/ModelBench.cu.in"
                "${_model_cu}"
                @ONLY
            )
            list(APPEND LARGE_MODEL_CU_SRCS "${_model_cu}")
        endforeach()

        set(BENCHMARK_EMIT_CAPTURES      "${LARGE_EMIT_CAPTURES}")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/BenchmarkEmitter.cxx.in"
            "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkEmitter_large.cxx"
            @ONLY
        )

        set(BENCHMARK_FWD_DECLS          "${LARGE_FWD_DECLS}")
        set(BENCHMARK_SINGLE_MODEL_CASES "${LARGE_SINGLE_MODEL_CASES}")
        set(BENCHMARK_SPAWN_CALLS        "${LARGE_SPAWN_CALLS}")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/BenchmarkRunner.cxx.in"
            "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkRunner_large.cpp"
            @ONLY
        )

        add_executable(sofie_benchmark_large_emitter
            "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkEmitter_large.cxx"
        )
        target_include_directories(sofie_benchmark_large_emitter PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../core/inc"
            "${CMAKE_CURRENT_SOURCE_DIR}/../parsers/inc"
        )
        target_link_libraries(sofie_benchmark_large_emitter PRIVATE
            SOFIE_core
            SOFIE_parsers
        )
        target_compile_options(sofie_benchmark_large_emitter PRIVATE
            -Wno-unused-parameter
            -Wno-array-bounds
        )
        target_compile_definitions(sofie_benchmark_large_emitter PRIVATE
            $<$<BOOL:${SOFIE_BENCHMARK_PROFILE}>:SOFIE_BENCHMARK_PROFILE>
            $<$<BOOL:${SOFIE_BENCHMARK_LOWRANK}>:SOFIE_BENCHMARK_LOWRANK>
            $<$<BOOL:${SOFIE_BENCHMARK_LOWRANK}>:SOFIE_BENCHMARK_LOWRANK_RATIO=${SOFIE_BENCHMARK_LOWRANK_RATIO}>
        )

        add_custom_command(
            OUTPUT  ${LARGE_GENERATED_HEADERS}
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
            COMMAND "$<TARGET_FILE:sofie_benchmark_large_emitter>" "${CMAKE_CURRENT_BINARY_DIR}"
            DEPENDS sofie_benchmark_large_emitter ${_ALL_LARGE_ONNX}
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "SOFIE Benchmark Large: generating headers for ${N_LARGE} large model(s)..."
            VERBATIM
        )

        add_custom_target(sofie_benchmark_large_headers
            DEPENDS ${LARGE_GENERATED_HEADERS}
        )

        set_source_files_properties(${LARGE_MODEL_CU_SRCS} PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(
            "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkRunner_large.cpp"
            PROPERTIES LANGUAGE CXX)

        add_executable(sofie_benchmark_large
            "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkRunner_large.cpp"
            ${LARGE_MODEL_CU_SRCS}
        )
        add_dependencies(sofie_benchmark_large sofie_benchmark_large_headers)

        target_include_directories(sofie_benchmark_large PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${alpaka_SOURCE_DIR}/include"
            "${sofieblas_SOURCE_DIR}/include"
            "${CUDAToolkit_INCLUDE_DIRS}"
        )

        set_target_properties(sofie_benchmark_large PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES          "${SOFIE_BENCHMARK_LARGE_CUDA_ARCH}"
            CUDA_STANDARD               20
            CUDA_STANDARD_REQUIRED      ON
        )

        target_compile_definitions(sofie_benchmark_large PRIVATE
            ${_SOFIE_BENCH_ALPAKA_DEFINE}
            ${_SOFIE_BENCH_BACKEND_DEFINE}
            ALPAKA_HAS_STD_ATOMIC_REF
            $<$<BOOL:${SOFIE_BENCHMARK_PROFILE}>:SOFIE_BENCHMARK_PROFILE>
            $<$<BOOL:${SOFIE_NVML_LIBRARY}>:SOFIE_BENCH_HAVE_NVML>
        )

        target_compile_options(sofie_benchmark_large PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --extended-lambda
                --expt-relaxed-constexpr
                --use_fast_math
                -O2
                -Wno-deprecated-gpu-targets
                -diag-suppress 177
                --maxrregcount=64
            >
            $<$<COMPILE_LANGUAGE:CXX>:
                -O2
                -fPIC
            >
        )

        target_link_libraries(sofie_benchmark_large PRIVATE
            SOFIE_core
            CUDA::cudart_static      # statically embed libcudart — no .so needed on the pod
            CUDA::cublas_static      # statically embed libcublas
            CUDA::cublasLt_static    # statically embed libcublasLt
            CUDA::culibos            # required companion for static cublas
            ${NVML_LIBRARY}
            # ORT intentionally excluded — no static libonnxruntime available
            Threads::Threads

        )

        message(STATUS
            "SOFIE Benchmark Large: target 'sofie_benchmark_large' configured "
            "(${N_LARGE} large model(s), sm${SOFIE_BENCHMARK_LARGE_CUDA_ARCH})")
    endif() # _ALL_LARGE_ONNX
endif() # SOFIE_BENCHMARK_LARGE
