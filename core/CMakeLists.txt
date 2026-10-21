############################################################################
# CMakeLists.txt file for building SOFIE standalone package
############################################################################

add_library(SOFIE_core SHARED)

set(sources_headers
   SOFIE/operators/OperatorList.hxx
   SOFIE/model/RModel_Base.hxx
   SOFIE/model/RModel.hxx
   SOFIE/model/RModelFusion_ALPAKA.hxx
   SOFIE/model/RModelProfiler.hxx
   SOFIE/model/RModelProfilerGPU.hxx
   SOFIE/operators/ROperator.hxx
   SOFIE/operators/ROperator_BasicUnary.hxx
   SOFIE/operators/ROperator_BasicBinary.hxx
   SOFIE/operators/ROperator_BasicNary.hxx
   SOFIE/operators/ROperator_BatchNormalization.hxx
   SOFIE/operators/ROperator_Conv.hxx
   SOFIE/operators/ROperator_ConvTranspose.hxx
   SOFIE/operators/ROperator_Gemm.hxx
   SOFIE/operators/ROperator_Relu.hxx
   SOFIE/operators/ROperator_Tanh.hxx
   SOFIE/operators/ROperator_LeakyRelu.hxx
   SOFIE/operators/ROperator_Reshape.hxx
   SOFIE/operators/ROperator_Selu.hxx
   SOFIE/operators/ROperator_Sigmoid.hxx
   SOFIE/operators/ROperator_Slice.hxx
   SOFIE/operators/ROperator_Transpose.hxx
   SOFIE/operators/ROperator_Pool.hxx
   SOFIE/operators/ROperator_Concat.hxx
   SOFIE/operators/ROperator_Identity.hxx
   SOFIE/operators/ROperator_Constant.hxx
   SOFIE/operators/ROperator_Softmax.hxx
   SOFIE/operators/ROperator_Reduce.hxx
   SOFIE/operators/ROperator_Shape.hxx
   SOFIE/operators/ROperator_Cast.hxx
   SOFIE/operators/ROperator_Custom.hxx
   SOFIE/operators/ROperator_LayerNormalization.hxx
   SOFIE/operators/ROperator_L2Normalization.hxx
   SOFIE/operators/ROperator_Expand.hxx
   SOFIE/operators/ROperator_Erf.hxx
   SOFIE/operators/ROperator_Swish.hxx
   SOFIE/operators/ROperator_Elu.hxx
   SOFIE/operators/ROperator_Comparison.hxx
   SOFIE/operators/ROperator_EyeLike.hxx
   SOFIE/operators/ROperator_Range.hxx
   SOFIE/operators/ROperator_TopK.hxx
   SOFIE/operators/ROperator_Tile.hxx
   SOFIE/operators/ROperator_Split.hxx
   SOFIE/operators/ROperator_SubGraph.hxx
   SOFIE/operators/ROperator_Pad.hxx
   SOFIE/operators/ROperator_Where.hxx
   SOFIE/operators/ROperator_Einsum.hxx
   SOFIE/operators/ROperator_ScatterElements.hxx
   SOFIE/utils/RTensor.hxx
   SOFIE/utils/SOFIE_common.hxx
   SOFIE/utils/SOFIEHelpers.hxx
   SOFIE/utils/SOFIE_formatter.hxx

   SOFIE/functions/FunctionList.hxx
   SOFIE/model/RModel_GNN.hxx
   SOFIE/model/RModel_GraphIndependent.hxx
   SOFIE/functions/RFunction.hxx
   SOFIE/functions/RFunction_MLP.hxx
   SOFIE/functions/RFunction_Sum.hxx
   SOFIE/functions/RFunction_Mean.hxx
)

list(TRANSFORM sources_headers PREPEND "inc/")

set(sources_cxx
    src/model/RModel_Base.cxx
    src/utils/SOFIE_formatter.cxx
    src/model/RModel.cxx
    src/model/RModelProfiler.cxx
    src/model/RModelProfilerGPU.cxx
    src/model/RModel_ALPAKA.cxx
    src/model/RModel_Fusion_ALPAKA.cxx
    src/model/RModel_GNN.cxx
    src/model/RModel_GraphIndependent.cxx
    src/functions/RFunction.cxx
    src/functions/RFunction_MLP.cxx
    src/functions/RFunction_Mean.cxx
    src/functions/RFunction_Sum.cxx
    src/utils/SOFIE_common.cxx
    src/utils/SOFIE_common_helpers.cxx
)

# helper code shared by the generator and the generated code: written once in inc/SOFIE/utils/Shared/*.inc,
# embedded as a string for the generated code
set(shared_code_names ConvertShapeToLength ConvertShapeToString Broadcast BroadcastConvBias MemoryPlanner)
set(shared_code_dir "${CMAKE_CURRENT_SOURCE_DIR}/inc/SOFIE/utils/Shared")
set(shared_code_cxx "${CMAKE_CURRENT_BINARY_DIR}/SOFIE_shared_code.cxx")
set(shared_code_files)
foreach(name IN LISTS shared_code_names)
  list(APPEND shared_code_files "${shared_code_dir}/${name}.inc")
endforeach()
add_custom_command(
  OUTPUT "${shared_code_cxx}"
  COMMAND ${CMAKE_COMMAND} -DOUT=${shared_code_cxx} -DDIR=${shared_code_dir} "-DNAMES=${shared_code_names}"
          -P "${PROJECT_SOURCE_DIR}/cmake/EmbedSharedCode.cmake"
  DEPENDS ${shared_code_files} "${PROJECT_SOURCE_DIR}/cmake/EmbedSharedCode.cmake"
  COMMENT "Embedding shared helper code"
  VERBATIM)

target_sources(SOFIE_core PRIVATE ${sources_headers} ${sources_cxx} "${shared_code_cxx}")
target_include_directories(SOFIE_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS SOFIE_core
  EXPORT SOFIETargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/inc/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

