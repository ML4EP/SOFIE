# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building SOFIE standalone package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021

#sofie is built only if protobuf is found

add_library(SOFIE_core SHARED)

set(sources_headers
   SOFIE/OperatorList.hxx
   SOFIE/RModel_Base.hxx
   SOFIE/RModel.hxx
   SOFIE/RModelExtension.hxx
   SOFIE/RModelCodegenPass.hxx
   SOFIE/RModelProfiler.hxx
   SOFIE/RModelProfilerGPU.hxx
   SOFIE/RWeightFile.hxx
   SOFIE/ROperator.hxx
   SOFIE/ROperator_BasicUnary.hxx
   SOFIE/ROperator_BasicBinary.hxx
   SOFIE/ROperator_BasicNary.hxx
   SOFIE/ROperator_BatchNormalization.hxx
   SOFIE/ROperator_Conv.hxx
   SOFIE/ROperator_ConvTranspose.hxx
   SOFIE/ROperator_Gemm.hxx
   SOFIE/ROperator_Relu.hxx
   SOFIE/ROperator_Tanh.hxx
   SOFIE/ROperator_LeakyRelu.hxx
   SOFIE/ROperator_Reshape.hxx
   SOFIE/ROperator_Selu.hxx
   SOFIE/ROperator_Sigmoid.hxx
   SOFIE/ROperator_Slice.hxx
   SOFIE/ROperator_Transpose.hxx
   SOFIE/ROperator_Pool.hxx
   SOFIE/ROperator_Concat.hxx
   SOFIE/ROperator_Identity.hxx
   SOFIE/ROperator_Constant.hxx
   SOFIE/ROperator_Softmax.hxx
   SOFIE/ROperator_Reduce.hxx
   SOFIE/ROperator_Shape.hxx
   SOFIE/ROperator_Cast.hxx
   SOFIE/ROperator_Custom.hxx
   SOFIE/ROperator_LayerNormalization.hxx
   SOFIE/ROperator_Expand.hxx
   SOFIE/ROperator_Erf.hxx
   SOFIE/ROperator_Swish.hxx
   SOFIE/ROperator_Elu.hxx
   SOFIE/ROperator_Comparision.hxx
   SOFIE/ROperator_EyeLike.hxx
   SOFIE/ROperator_Range.hxx
   SOFIE/ROperator_TopK.hxx
   SOFIE/ROperator_Tile.hxx
   SOFIE/ROperator_Split.hxx
   SOFIE/ROperator_SubGraph.hxx
   SOFIE/ROperator_Pad.hxx
   SOFIE/ROperator_QONNXQuant.hxx
   SOFIE/ROperator_ONNXQuantizeLinear.hxx
   SOFIE/ROperator_Where.hxx
   SOFIE/ROperator_Einsum.hxx
   SOFIE/ROperator_Random.hxx
   SOFIE/ROperator_ScatterElements.hxx
   SOFIE/SOFIE_common.hxx
   SOFIE/SOFIEHelpers.hxx

   SOFIE/FunctionList.hxx
   SOFIE/RModel_GNN.hxx
   SOFIE/RModel_GraphIndependent.hxx
   SOFIE/RFunction.hxx
   SOFIE/RFunction_MLP.hxx
   SOFIE/RFunction_Sum.hxx
   SOFIE/RFunction_Mean.hxx
)

# The quantization library's headers, one directory per workflow module (see
# src/quantization/README.md).
# LinkDef names no quantization class, so keeping them out of the base list leaves the
# base dictionary complete.
set(sources_quantization_headers
   SOFIE/quantization/spec/RQuantization_Spec.hxx
   SOFIE/quantization/spec/RQuantization_Codec.hxx
   SOFIE/quantization/frontend/ROperator_QuantBoundary.hxx
   SOFIE/quantization/frontend/RQuantization_Frontend.hxx
   SOFIE/quantization/graph/RQuantization_Graph.hxx
   SOFIE/quantization/handlers/RQuantization_Handler.hxx
   SOFIE/quantization/handlers/RQuantization_Handlers.hxx
   SOFIE/quantization/handlers/RQuantization_HandlerSupport.hxx
   SOFIE/quantization/lowered/ROperator_QuantLowered.hxx
   SOFIE/quantization/lowered/ROperator_QuantizedDenseLinear.hxx
   SOFIE/quantization/lowered/ROperator_QuantizedConv.hxx
   SOFIE/quantization/lowered/ROperator_QuantizedElementwise.hxx
   SOFIE/quantization/lowered/ROperator_QuantizedGather.hxx
   SOFIE/quantization/passes/RQuantization_Passes.hxx
   SOFIE/quantization/diagnostics/RQuantization_Diagnostics.hxx
   SOFIE/quantization/integration/RQuantization_Workflow.hxx
   SOFIE/RQuantization.hxx
)

list(TRANSFORM sources_headers PREPEND "inc/")
list(TRANSFORM sources_quantization_headers PREPEND "inc/")

# The quantization library's translation units.
set(sources_quantization_cxx
    src/quantization/spec/RQuantization_Spec.cxx
    src/quantization/frontend/RQuantization_Frontend.cxx
    src/quantization/frontend/RQuantization_FrontendQDQ.cxx
    src/quantization/frontend/RQuantization_FrontendQONNX.cxx
    src/quantization/frontend/RQuantization_FrontendNative.cxx
    src/quantization/graph/RQuantization_Graph.cxx
    src/quantization/handlers/RQuantization_Handlers.cxx
    src/quantization/handlers/RQuantization_HandlerRegistry.cxx
    src/quantization/handlers/RQuantization_HandlerSupport.cxx
    src/quantization/handlers/RQuantization_HandlerDenseLinear.cxx
    src/quantization/handlers/RQuantization_HandlerConv.cxx
    src/quantization/handlers/RQuantization_HandlerElementwise.cxx
    src/quantization/handlers/RQuantization_HandlerGather.cxx
    src/quantization/handlers/RQuantization_HandlerFusible.cxx
    src/quantization/lowered/ROperator_QuantLowered.cxx
    src/quantization/passes/RQuantization_PassManager.cxx
    src/quantization/passes/RQuantization_PassCanonicalize.cxx
    src/quantization/passes/RQuantization_PassAnnotate.cxx
    src/quantization/passes/RQuantization_PassDecide.cxx
    src/quantization/passes/RQuantization_PassLower.cxx
    src/quantization/diagnostics/RQuantization_Diagnostics.cxx
    src/quantization/integration/RQuantization_Workflow.cxx
)

set(sources_cxx
    src/RModel_Base.cxx
    src/RModel.cxx
    src/RModelCodegenPass.cxx
    src/RWeightFile.cxx
    src/RModelProfiler.cxx
    src/RModelProfilerGPU.cxx
    src/RModel_ALPAKA.cxx
    src/RModel_GNN.cxx
    src/RModel_GraphIndependent.cxx
    src/RFunction.cxx
    src/RFunction_MLP.cxx
    src/RFunction_Mean.cxx
    src/RFunction_Sum.cxx
    src/SOFIE_common.cxx
)

target_sources(SOFIE_core PRIVATE ${sources_headers} ${sources_cxx})
target_include_directories(SOFIE_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(SOFIE_core PUBLIC utils)

if(SOFIE_WITH_ROOT AND ROOT_FOUND)
  target_compile_definitions(SOFIE_core PUBLIC SOFIE_SUPPORT_ROOT_BINARY)
  target_link_libraries(SOFIE_core PUBLIC Tree Core RIO)

  ROOT_GENERATE_DICTIONARY(G__SOFIE_core ${sources_headers}
      LINKDEF inc/LinkDef.h
      MODULE SOFIE_core
      OPTIONS --deep
  )

  # Install the dictionaries.
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libSOFIE_core_rdict.pcm
                ${CMAKE_CURRENT_BINARY_DIR}/libSOFIE_core.rootmap
          DESTINATION lib)
endif()

# The quantization pipeline (optional: SOFIE_WITH_QUANTIZATION). Depends on base SOFIE, which
# depends on nothing here, so the link below runs one way.
set(SOFIE_INSTALL_TARGETS SOFIE_core)
if(SOFIE_WITH_QUANTIZATION)
  add_library(SOFIE_quantization SHARED)
  target_sources(SOFIE_quantization PRIVATE ${sources_quantization_headers} ${sources_quantization_cxx})
  target_link_libraries(SOFIE_quantization PUBLIC SOFIE_core)
  list(APPEND SOFIE_INSTALL_TARGETS SOFIE_quantization)
endif()

install(TARGETS ${SOFIE_INSTALL_TARGETS}
  EXPORT SOFIETargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/inc/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
