cmake_minimum_required(VERSION 3.16)
project(Sofie
    VERSION 1.0.1
    DESCRIPTION "SOFIE"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(BLAS)
if(NOT BLAS_FOUND)
  message(WARNING "BLAS not found: sofie will not be fully tested")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(ccache)
  set(CMAKE_C_COMPILER_LAUNCHER ccache)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif()

option(testing "Build and run tests" OFF)
if(testing)
  find_package(GTest REQUIRED)
  enable_testing()
endif()

option(SOFIE_BENCHMARK "Build the SOFIE CUDA benchmark toolkit" OFF)

option(SOFIE_ALPAKA_CUDA_DEBUG "Build Alpaka CUDA tests with device debug info (-G, -O0) for use with cuda-gdb; slows GPU test execution, so OFF for normal test runs" OFF)

include(cmake/modules/SofieTestMacros.cmake)

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(parsers)

if(testing)
  add_subdirectory(test)
endif()

if(SOFIE_BENCHMARK)
  add_subdirectory(benchmark)
endif()

# ── Install cmake package config files ──────────────────────────────────────

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SOFIEConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SOFIEConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SOFIE
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/SOFIEConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(
  EXPORT SOFIETargets
  FILE SOFIETargets.cmake
  NAMESPACE SOFIE::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SOFIE
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/SOFIEConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/SOFIEConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SOFIE
)
