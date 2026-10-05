cmake_minimum_required(VERSION 3.14)
include(FetchContent)

############################################################################
# Basic setup
############################################################################
include_directories(${CMAKE_SOURCE_DIR}/core/inc)
include_directories(${CMAKE_SOURCE_DIR}/parsers/inc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter QUIET)

function(sofie_find_python_module module)
  string(TOUPPER ${module} module_upper)
  set(found FALSE)
  set(version "")
  if (Python3_EXECUTABLE)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import ${module}; print(getattr(${module}, '__version__', ''))"
      RESULT_VARIABLE status
      OUTPUT_VARIABLE version
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (status EQUAL 0)
      set(found TRUE)
    endif()
  endif()
  set(SOFIE_${module_upper}_FOUND ${found} PARENT_SCOPE)
  set(SOFIE_${module_upper}_VERSION "${version}" PARENT_SCOPE)
endfunction()

sofie_find_python_module(torch)
sofie_find_python_module(onnx)
sofie_find_python_module(numpy)

if (SOFIE_ONNX_FOUND AND SOFIE_ONNX_VERSION VERSION_EQUAL "1.19.0")
  message(WARNING "Found broken onnx version ${SOFIE_ONNX_VERSION} (see https://github.com/onnx/onnx/issues/7249). Some SOFIE tests will be disabled.")
  set(broken_onnx TRUE)
endif()

set(SOFIE_ONNX_USABLE FALSE)
if (SOFIE_ONNX_FOUND AND SOFIE_NUMPY_FOUND AND NOT broken_onnx)
  set(SOFIE_ONNX_USABLE TRUE)
endif()

function(sofie_write_model_includes testsource outfile suffix)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/${testsource} _content)
  string(REGEX MATCHALL "ASSERT_RUN[A-Z_0-9]*\\([ \t\r\n]*[^,]+,[ \t\r\n]*[A-Za-z0-9_]+" _calls "${_content}")
  set(_names "")
  foreach(call ${_calls})
    string(REGEX REPLACE ".*,[ \t\r\n]*([A-Za-z0-9_]+)$" "\\1" name "${call}")
    list(APPEND _names ${name})
  endforeach()
  list(REMOVE_DUPLICATES _names)
  set(_inc "")
  foreach(name ${_names})
    string(APPEND _inc "#include \"${name}${suffix}.hxx\"\n")
  endforeach()
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${outfile} "${_inc}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${testsource})
endfunction()

function(sofie_set_driver_definitions target)
  string(REPLACE ";" " " _blas_libs "${BLAS_LIBRARIES}")
  target_compile_definitions(${target} PRIVATE
    SOFIE_TEST_CXX="${CMAKE_CXX_COMPILER}"
    SOFIE_TEST_CXX_FLAGS="-std=c++20 -O1 -I${CMAKE_SOURCE_DIR}/core/inc"
    SOFIE_TEST_LINK_FLAGS="${_blas_libs}"
    SOFIE_TEST_PYTHON="${Python3_EXECUTABLE}")
endfunction()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             ${CMAKE_CURRENT_SOURCE_DIR}/models/generators/generate_input_models.py)

set(USER_MODELS_DIR "")
if (ONNX_MODELS_DIR)
  get_filename_component(USER_MODELS_DIR ${ONNX_MODELS_DIR} ABSOLUTE)
  set(CHECKED_IN_MODELS_DIR ${USER_MODELS_DIR})
else()
  set(ONNX_MODELS_DIR models/onnx)
  set(CHECKED_IN_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/models/onnx)
endif()

option(ENABLE_ALPAKA_TESTS "Enable Alpaka-based SOFIE tests" OFF)

set(ALPAKA_BACKEND "cuda"
    CACHE STRING "Alpaka backend to test (cuda, cpu, hip, sycl)")
set_property(CACHE ALPAKA_BACKEND PROPERTY STRINGS cuda cpu hip sycl)

############################################################################
# Generate emitter sources
############################################################################
set(CAPTURE_STR
"try {\n\
    EmitModel(\"@1\", \"@2\");\n\
} catch (const std::exception& e) {\n\
    std::string msg = e.what();\n\
    if (msg.find(\"multiple output tensors are not supported\") != std::string::npos) {\n\
        std::cerr << \"[SKIP] Multiple outputs are not supported for @1\" << std::endl;\n\
    } else if (msg.find(\"is of a data type which is not yet supported\") != std::string::npos) {\n\
        std::cerr << \"[SKIP] Operator with unsupported data type in @1: \" << msg << std::endl;\n\
    } else {\n\
        std::cerr << \"[ERROR] Failed processing @1: \" << msg << std::endl;\n\
        failures++;\n\
    }\n\
} catch (...) {\n\
    std::cerr << \"[ERROR] Unknown failure processing @1\" << std::endl;\n\
    failures++;\n\
}\n\
")

file(GLOB ONNX_FILES "${CHECKED_IN_MODELS_DIR}/*.onnx")

set(ALL_CAPTURES "")
foreach(onnx_file ${ONNX_FILES})
    get_filename_component(fname ${onnx_file} NAME_WE)
    string(REPLACE "@1" "${onnx_file}" cap "${CAPTURE_STR}")
    string(REPLACE "@2" "${fname}" cap "${cap}")
    string(APPEND ALL_CAPTURES "${cap}")
endforeach()

# Alpaka tests use the same set of models as the CPU tests.
set(ALPAKA_CAPTURES "${ALL_CAPTURES}")

set(EMIT_CAPTURES "${ALPAKA_CAPTURES}")
configure_file(templates/EmitFromONNX_GPU_ALPAKA.cxx.in EmitFromONNX_GPU_ALPAKA_all.cxx @ONLY)

SOFIE_GENERATE_EXECUTABLE(
  emitLowRankModels
  models/LowRankModelGenerator.cxx
  LIBRARIES SOFIE_core
  FIXTURES_SETUP sofie-lowrank-emit-build
)

SOFIE_ADD_TEST(
  EmitLowRankModels
  COMMAND ./emitLowRankModels
  FIXTURES_REQUIRED sofie-lowrank-emit-build
  FIXTURES_SETUP sofie-lowrank-emit
)

SOFIE_GENERATE_EXECUTABLE(
  TestLowRankFactorization
  cpu/TestLowRankFactorization.cxx
  LIBRARIES SOFIE_core GTest::gtest GTest::gtest_main BLAS::BLAS
  FIXTURES_REQUIRED sofie-lowrank-emit
  FIXTURES_SETUP sofie-test-lowrank-build
)

target_include_directories(TestLowRankFactorization PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

SOFIE_ADD_TEST(
  TestLowRankFactorization
  EXEC ./TestLowRankFactorization
  FIXTURES_REQUIRED sofie-lowrank-emit;sofie-test-lowrank-build
)

SOFIE_GENERATE_EXECUTABLE(
  emitStridedInputModels
  models/StridedInputModelGenerator.cxx
  LIBRARIES SOFIE_core
  FIXTURES_SETUP sofie-strided-emit-build
)

SOFIE_ADD_TEST(
  EmitStridedInputModels
  COMMAND ./emitStridedInputModels
  FIXTURES_REQUIRED sofie-strided-emit-build
  FIXTURES_SETUP sofie-strided-emit
)

SOFIE_GENERATE_EXECUTABLE(
  TestStridedInput
  cpu/TestStridedInput.cxx
  LIBRARIES SOFIE_core GTest::gtest GTest::gtest_main BLAS::BLAS
  FIXTURES_REQUIRED sofie-strided-emit
  FIXTURES_SETUP sofie-test-strided-build
)

target_include_directories(TestStridedInput PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

SOFIE_ADD_TEST(
  TestStridedInput
  EXEC ./TestStridedInput
  FIXTURES_REQUIRED sofie-strided-emit;sofie-test-strided-build
)

############################################################################
# Alpaka tests
############################################################################
if (ENABLE_ALPAKA_TESTS)

  string(TOLOWER "${ALPAKA_BACKEND}" _alpaka_backend)
  if (NOT _alpaka_backend IN_LIST ALPAKA_BACKEND)
    message(FATAL_ERROR "Unsupported ALPAKA_BACKEND=${ALPAKA_BACKEND}")
  endif()

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

  ##########################################################################
  # Alpaka emitter
  ##########################################################################
SOFIE_GENERATE_EXECUTABLE(
      emitFromONNXAlpaka
      ${CMAKE_CURRENT_BINARY_DIR}/EmitFromONNX_GPU_ALPAKA_all.cxx
      LIBRARIES SOFIE_core SOFIE_parsers
      FIXTURES_SETUP sofie-compile-models-onnx-alpaka-build
)

  target_compile_options(emitFromONNXAlpaka PRIVATE
    -Wno-unused-parameter
    -Wno-array-bounds
  )

  SOFIE_ADD_TEST(
    SofieCompileModels_ONNX_Alpaka
    COMMAND ${CMAKE_COMMAND} -E env ./emitFromONNXAlpaka
    FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka-build
    FIXTURES_SETUP sofie-compile-models-onnx-alpaka
  )

  ##########################################################################
  # CUDA backend
  ##########################################################################
  if (_alpaka_backend STREQUAL "cuda")

    message(STATUS "Enabling Alpaka CUDA tests")

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(ALPAKA_CUDA_TEST_SOURCES
      alpaka/TestAlpakaLinear.cxx
      alpaka/TestAlpakaConv.cxx
      alpaka/TestAlpakaNormalization.cxx
      alpaka/TestAlpakaElementwiseUnary.cxx
      alpaka/TestAlpakaElementwiseBinaryLogic.cxx
      alpaka/TestAlpakaReduce.cxx
      alpaka/TestAlpakaShapeIndexing.cxx
      alpaka/TestAlpakaGNN.cxx
      alpaka/TestAlpakaPool.cxx
      alpaka/TestAlpakaLowRank.cxx
      alpaka/TestAlpakaKernelOnly.cxx
      alpaka/TestAlpakaStridedInput.cxx
      alpaka/TestAlpakaNormReduction.cxx
      alpaka/TestAlpakaSequenceOps.cxx
      alpaka/TestAlpakaSubGraph.cxx
    )

    set_source_files_properties(
      ${ALPAKA_CUDA_TEST_SOURCES}
      PROPERTIES LANGUAGE CUDA
    )

    SOFIE_GENERATE_EXECUTABLE(
      TestCustomModelsFromONNXForAlpakaCuda
      ${ALPAKA_CUDA_TEST_SOURCES}
      LIBRARIES SOFIE_core GTest::gtest GTest::gtest_main
      FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka;sofie-lowrank-emit;sofie-strided-emit
      FIXTURES_SETUP sofie-test-models-onnx-alpaka-build
    )

    target_include_directories(
      TestCustomModelsFromONNXForAlpakaCuda PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
      ${alpaka_SOURCE_DIR}/include
      ${sofieblas_SOURCE_DIR}/include
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}
    )

    if(CMAKE_CUDA_ARCHITECTURES)
        set(_cuda_archs "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(_cuda_archs "70;80;86")
    endif()

    set_target_properties(
            TestCustomModelsFromONNXForAlpakaCuda
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES "${_cuda_archs}"
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
    )

    target_compile_definitions(
      TestCustomModelsFromONNXForAlpakaCuda PRIVATE
      ALPAKA_ACC_GPU_CUDA_ENABLED
      ALPAKA_HAS_STD_ATOMIC_REF
    )

    target_compile_options(
      TestCustomModelsFromONNXForAlpakaCuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --generate-line-info
        --use_fast_math
        -Wno-deprecated-gpu-targets
        $<IF:$<BOOL:${SOFIE_ALPAKA_CUDA_DEBUG}>,-g;-G;-O0,-O3>
      >
      $<$<COMPILE_LANGUAGE:CXX>:
        -fPIC
        -pthread
        $<IF:$<BOOL:${SOFIE_ALPAKA_CUDA_DEBUG}>,-g;-O0,-O2>
      >
    )

    # ROOT-compatible: plain signature only
    target_link_libraries(
      TestCustomModelsFromONNXForAlpakaCuda
      CUDA::cudart
      CUDA::cublas
      CUDA::cublasLt
    )

    SOFIE_ADD_TEST(
      TestCustomModelsFromONNXForAlpakaCuda
      EXEC ./TestCustomModelsFromONNXForAlpakaCuda
      FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka;sofie-lowrank-emit;sofie-test-models-onnx-alpaka-build
      TIMEOUT 1800
    )

  elseif (_alpaka_backend STREQUAL "hip")
    message(FATAL_ERROR
      "ALPAKA_BACKEND=hip is not implemented yet (AMD GPU support is planned, "
      "see tests/alpaka/). Use ALPAKA_BACKEND=cuda for now.")
  elseif (_alpaka_backend STREQUAL "sycl")
    message(FATAL_ERROR "ALPAKA_BACKEND=sycl is not implemented yet.")
  elseif (_alpaka_backend STREQUAL "cpu")
    message(FATAL_ERROR "ALPAKA_BACKEND=cpu is not implemented yet.")
  endif() # cuda backend
endif()   # ENABLE_ALPAKA_TESTS

############################################################################
############################################################################
set(generate_onnx_models FALSE)
set(TEST_MODELS_DIR "")
set(ONNX_MODEL_NAMES "")
set(ONNX_MODEL_PATHS "")

if (USER_MODELS_DIR)
  set(TEST_MODELS_DIR ${USER_MODELS_DIR})
  file(GLOB _user_onnx_files "${USER_MODELS_DIR}/*.onnx")
  foreach(onnx_file ${_user_onnx_files})
    get_filename_component(fname ${onnx_file} NAME_WE)
    list(APPEND ONNX_MODEL_NAMES ${fname})
    list(APPEND ONNX_MODEL_PATHS ${onnx_file})
  endforeach()
elseif (SOFIE_ONNX_USABLE)
  set(generate_onnx_models TRUE)
  set(TEST_MODELS_DIR ${CMAKE_CURRENT_BINARY_DIR}/models/onnx)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/models/generators/generate_input_models.py --list
    OUTPUT_VARIABLE _onnx_model_names
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _onnx_list_status)
  if (NOT _onnx_list_status EQUAL 0)
    message(FATAL_ERROR "Failed to list the SOFIE ONNX test models with generate_input_models.py")
  endif()
  string(REPLACE "\n" ";" _generated_names "${_onnx_model_names}")
  foreach(fname ${_generated_names})
    list(APPEND ONNX_MODEL_NAMES ${fname})
    list(APPEND ONNX_MODEL_PATHS ${TEST_MODELS_DIR}/${fname}.onnx)
  endforeach()
  foreach(fname ScatterND_Ex1 ScatterND_Ex2 ScatterND_NegativeIndices ScatterND_2D)
    if (EXISTS ${CHECKED_IN_MODELS_DIR}/${fname}.onnx AND NOT fname IN_LIST ONNX_MODEL_NAMES)
      list(APPEND ONNX_MODEL_NAMES ${fname})
      list(APPEND ONNX_MODEL_PATHS ${CHECKED_IN_MODELS_DIR}/${fname}.onnx)
    endif()
  endforeach()
else()
  message(STATUS "The onnx and numpy Python modules are not usable: the generated ONNX models are not available and the ONNX tests based on them are disabled. Set ONNX_MODELS_DIR to use existing models.")
endif()

if (ONNX_MODEL_NAMES)
  set(EMIT_CAPTURES "")
  list(LENGTH ONNX_MODEL_NAMES _n_models)
  math(EXPR _last_model "${_n_models} - 1")
  foreach(idx RANGE 0 ${_last_model})
    list(GET ONNX_MODEL_NAMES ${idx} fname)
    list(GET ONNX_MODEL_PATHS ${idx} onnx_file)
    string(REPLACE "@1" "${onnx_file}" cap "${CAPTURE_STR}")
    string(REPLACE "@2" "${fname}" cap "${cap}")
    string(APPEND EMIT_CAPTURES "${cap}")
  endforeach()
  configure_file(templates/EmitFromONNX.cxx.in EmitFromONNX_all.cxx @ONLY)

  SOFIE_GENERATE_EXECUTABLE(emitFromONNX ${CMAKE_CURRENT_BINARY_DIR}/EmitFromONNX_all.cxx
                               LIBRARIES SOFIE_core SOFIE_parsers
                               FIXTURES_SETUP sofie-compile-models-onnx-build)

  set(compile_models_fixtures sofie-compile-models-onnx-build)
  if (generate_onnx_models)
    SOFIE_ADD_TEST(SofieGenerateModels_ONNX
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/models/generators/generate_input_models.py --outdir ${TEST_MODELS_DIR}
      FIXTURES_SETUP sofie-generate-models-onnx
    )
    list(APPEND compile_models_fixtures sofie-generate-models-onnx)
  endif()

  SOFIE_ADD_TEST(SofieCompileModels_ONNX
    COMMAND ${CMAKE_COMMAND} -E env ./emitFromONNX
    FIXTURES_REQUIRED ${compile_models_fixtures}
    FIXTURES_SETUP sofie-compile-models-onnx
  )

  set(alias_test_fixtures sofie-test-alias-build)
  if (generate_onnx_models)
    list(APPEND alias_test_fixtures sofie-generate-models-onnx)
  endif()
  SOFIE_GENERATE_EXECUTABLE(TestSofieAlias cpu/TestSofieAlias.cxx
    LIBRARIES SOFIE_core SOFIE_parsers GTest::gtest GTest::gtest_main
    FIXTURES_SETUP sofie-test-alias-build
  )
  target_compile_definitions(TestSofieAlias PRIVATE SOFIE_ONNX_MODELS_DIR="${TEST_MODELS_DIR}")
  SOFIE_ADD_TEST(TestSofieAlias
                    EXEC ./TestSofieAlias
                    FIXTURES_REQUIRED ${alias_test_fixtures})

  if (BLAS_FOUND)
    sofie_write_model_includes(cpu/TestCustomModelsFromONNX.cxx TestCustomModelsFromONNX_models.hxx _FromONNX)
    SOFIE_GENERATE_EXECUTABLE(TestCustomModelsFromONNX cpu/TestCustomModelsFromONNX.cxx
      LIBRARIES SOFIE_core GTest::gtest GTest::gtest_main BLAS::BLAS
      FIXTURES_REQUIRED sofie-compile-models-onnx
      FIXTURES_SETUP sofie-test-custom-models-onnx-build
    )
    target_include_directories(TestCustomModelsFromONNX PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    SOFIE_ADD_TEST(TestCustomModelsFromONNX
                   EXEC ./TestCustomModelsFromONNX
                   FIXTURES_REQUIRED sofie-compile-models-onnx;sofie-test-custom-models-onnx-build)
  endif()
endif()

SOFIE_GENERATE_EXECUTABLE(TestSofieParser cpu/TestSofieParser.cxx
  LIBRARIES SOFIE_core SOFIE_parsers GTest::gtest GTest::gtest_main
  FIXTURES_SETUP sofie-test-parser-build
)
SOFIE_ADD_TEST(TestSofieParser
                  EXEC ./TestSofieParser
                  FIXTURES_REQUIRED sofie-test-parser-build)

find_package(nlohmann_json QUIET)
if (NOT nlohmann_json_FOUND)
  find_path(SOFIE_NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
  if (SOFIE_NLOHMANN_JSON_INCLUDE_DIR)
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE ${SOFIE_NLOHMANN_JSON_INCLUDE_DIR})
    add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
    set(nlohmann_json_FOUND TRUE)
  endif()
endif()
if (nlohmann_json_FOUND AND BLAS_FOUND)
  SOFIE_GENERATE_EXECUTABLE(TestSafetensorsWeights cpu/TestSafetensorsWeights.cxx
    LIBRARIES SOFIE_core SOFIE_parsers GTest::gtest GTest::gtest_main BLAS::BLAS nlohmann_json::nlohmann_json
    FIXTURES_SETUP sofie-test-safetensors-build
  )
  target_include_directories(TestSafetensorsWeights PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  sofie_set_driver_definitions(TestSafetensorsWeights)
  SOFIE_ADD_TEST(TestSafetensorsWeights
                 EXEC ./TestSafetensorsWeights
                 FIXTURES_REQUIRED sofie-test-safetensors-build)
else()
  message(STATUS "nlohmann_json or BLAS not found: TestSafetensorsWeights is disabled")
endif()

if (SOFIE_TORCH_FOUND AND SOFIE_ONNX_USABLE AND BLAS_FOUND)
  configure_file(models/generators/ModelGeneratorUtils.py ModelGeneratorUtils.py COPYONLY)
  foreach(generator Conv1d Conv2d Conv3d ConvTrans2d Linear Recurrent)
    configure_file(models/generators/${generator}ModelGenerator.py ${generator}ModelGenerator.py COPYONLY)
  endforeach()

  SOFIE_GENERATE_EXECUTABLE(TestSofieModels cpu/TestSofieModels.cxx
    LIBRARIES SOFIE_core SOFIE_parsers GTest::gtest GTest::gtest_main BLAS::BLAS
    FIXTURES_SETUP sofie-test-models-build
  )
  target_include_directories(TestSofieModels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  sofie_set_driver_definitions(TestSofieModels)
  SOFIE_ADD_TEST(TestSofieModels
                 EXEC ./TestSofieModels
                 FIXTURES_REQUIRED sofie-test-models-build)
endif()
