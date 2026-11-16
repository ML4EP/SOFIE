# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021

add_library(SOFIE_parsers SHARED)

set(source_headers
   SOFIE/RModelParser_ONNX.hxx
)
list(TRANSFORM source_headers PREPEND "inc/")
target_include_directories(SOFIE_parsers
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set(sources_cxx
    src/RModelParser_ONNX.cxx
    src/ParseBasicBinary.cxx
    src/ParseBasicIs.cxx
    src/ParseBasicNary.cxx
    src/ParseBasicUnary.cxx
    src/ParseBatchNormalization.cxx
    src/ParseCast.cxx
    src/ParseClip.cxx
    src/ParseComparison.cxx
    src/ParseConcat.cxx
    src/ParseConstant.cxx
    src/ParseConv.cxx
    src/ParseConvTranspose.cxx
    src/ParseCumSum.cxx
    src/ParseEinsum.cxx
    src/ParseElu.cxx
    src/ParseErf.cxx
    src/ParseExpand.cxx
    src/ParseEyeLike.cxx
    src/ParseFuseBatchnormRelu.cxx
    src/ParseFuseConvAdd.cxx
    src/ParseFuseConvTransposeAdd.cxx
    src/ParseFuseGemmRelu.cxx
    src/ParseFuseL2Normalization.cxx
    src/ParseFuseMatMulAdd.cxx
    src/ParseGRU.cxx
    src/ParseGather.cxx
    src/ParseGatherND.cxx
    src/ParseGelu.cxx
    src/ParseGemm.cxx
    src/ParseGriffinRGLRU.cxx
    src/ParseGroupNorm.cxx
    src/ParseHardSigmoid.cxx
    src/ParseHardSwish.cxx
    src/ParseIdentity.cxx
    src/ParseIf.cxx
    src/ParseInstanceNormalization.cxx
    src/ParseLSTM.cxx
    src/ParseLayerNormalization.cxx
    src/ParseLeakyRelu.cxx
    src/ParseLogic.cxx
    src/ParseMambaScan.cxx
    src/ParseMatMul.cxx
    src/ParseNonZero.cxx
    src/ParseNot.cxx
    src/ParsePad.cxx
    src/ParsePool.cxx
    src/ParseRMSNorm.cxx
    src/ParseRNN.cxx
    src/ParseRWKVWKV6.cxx
    src/ParseRange.cxx
    src/ParseReduce.cxx
    src/ParseRelu.cxx
    src/ParseReshape.cxx
    src/ParseSDPA.cxx
    src/ParseScatterElements.cxx
    src/ParseScatterND.cxx
    src/ParseSelu.cxx
    src/ParseShape.cxx
    src/ParseSigmoid.cxx
    src/ParseSlice.cxx
    src/ParseSoftmax.cxx
    src/ParseSplit.cxx
    src/ParseSwish.cxx
    src/ParseTanh.cxx
    src/ParseTile.cxx
    src/ParseTopK.cxx
    src/ParseTranspose.cxx
    src/ParseTrilu.cxx
    src/ParseWhere.cxx
)
target_sources(SOFIE_parsers PRIVATE ${source_headers} ${sources_cxx})
target_include_directories(SOFIE_parsers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(SOFIE_parsers PUBLIC SOFIE_core)

set_target_properties(SOFIE_parsers PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)

install(TARGETS SOFIE_parsers
  EXPORT SOFIETargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/inc/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
