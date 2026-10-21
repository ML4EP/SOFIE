#include <limits>
#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>
#include <vector>
#include "SOFIE/model/RModel_Base.hxx"
#include "SOFIE/utils/SOFIE_formatter.hxx"


namespace SOFIE {

RModel_Base::RModel_Base(std::string name, std::string parsedtime):fFileName(name), fParseTime(parsedtime) {
    fName = fFileName.substr(0, fFileName.rfind("."));
    fName = UTILITY::Clean_name(fName);
}

void RModel_Base::GenerateHeaderInfo(std::string& hgname) {
    fGC += ("//Code generated automatically by SOFIE for Inference of Model file [" + fFileName + "] at [" + fParseTime.substr(0, fParseTime.length()-1) +"] \n");
    // add header guards
    hgname = fName;
    std::transform(hgname.begin(), hgname.end(), hgname.begin(), [](unsigned char c) {
                       return std::toupper(c);
                   } );
    hgname = "SOFIE_" + hgname + "_HXX";
    fGC += "\n#ifndef " + hgname + "\n";
    fGC += "#define " + hgname + "\n\n";
    for (const char *h : {"cstdint", "cstring", "string", "vector", "map", "memory", "sstream", "iostream", "iomanip",
                          "limits", "stdexcept", "algorithm", "cmath", "cassert"}) {
        fNeededStdLib.insert(h);
    }
    if (fUseWeightFile)
        fNeededStdLib.insert("fstream");
    if (fWeightFile == WeightFileType::Safetensors)
        fNeededStdLib.insert("iterator");
    // the standard headers (the model's and the ones of the helper functions) are merged, sorted and written when the
    // code is finalized, followed by the other includes
    for (auto &i : fCustomOpHeaders)
        fHeaderIncludesTail += "#include \"" + i + "\"\n";
    fGC += kHelperIncludesMarker;

    fGC += "\nnamespace SOFIE_" + fName + "{\n";
    if (!fNeededBlasRoutines.empty()) {
        fGC += ("namespace BLAS{\n");
        for (auto &routine : fNeededBlasRoutines) {
            if (routine == "Gemm") {
                fGC += ("\textern \"C\" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,\n"
                        "\t                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,\n"
                        "\t                       const float * beta, float * C, const int * ldc);\n");
                fBlasSgemmDeclared = true;
            } else if (routine == "Gemv") {
                fGC += ("\textern \"C\" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,\n"
                        "\t                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);\n");
            } else if (routine == "Axpy") {
                fGC += ("\textern \"C\" void saxpy_(const int * n, const float * alpha, const float * x,\n"
                        "\t                         const int * incx, float * y, const int * incy);\n");
            } else if (routine == "Copy") {
                fGC += ("\textern \"C\" void scopy_(const int *n, const float* x, const int *incx, float* y, const int* incy);\n");
            }
        }
        fGC += ("}//BLAS\n");
    }
    fGC += kHelperFunctionsMarker;
}

void RModel_Base::EmitHelperFunctionsCode()
{
    HelperFunctionsCode code =
        GenerateHelperFunctionsCode(fNeededHelperFunctions, "SOFIE_" + fName, fBlasSgemmDeclared);

    auto replaceMarker = [this](const std::string &marker, const std::string &replacement) {
        auto pos = fGC.find(marker);
        if (pos != std::string::npos) {
            fGC.replace(pos, marker.size(), replacement);
        }
    };

    // one block of standard headers without repetitions, sorted, then the other includes after a blank line
    std::set<std::string> stdHeaders(fNeededStdLib.begin(), fNeededStdLib.end());
    stdHeaders.insert(code.stdHeaders.begin(), code.stdHeaders.end());
    std::string includes;
    for (const auto &h : stdHeaders)
        includes += "#include <" + h + ">\n";
    if (!fHeaderIncludesTail.empty())
        includes += "\n" + fHeaderIncludesTail;
    replaceMarker(kHelperIncludesMarker, includes);
    replaceMarker(kHelperFunctionsMarker, code.definitions);

    if (!code.cladDefinitions.empty()) {
        auto pos = fGC.rfind("#endif");
        if (pos != std::string::npos) {
            fGC.insert(pos, code.cladDefinitions + "\n");
        } else {
            fGC += code.cladDefinitions;
        }
    }
}

void RModel_Base::FinalizeGeneratedCode()
{
    EmitHelperFunctionsCode();
    fGC = FormatGeneratedCode(fGC);
}

void RModel_Base::GenerateHeaderInfo_GPU_ALPAKA(std::string& hgname) {
    fGC += ("//Code generated automatically by SOFIE for GPU Inference using ALPAKA of Model file [" + fFileName + "] at [" + fParseTime.substr(0, fParseTime.length()-1) +"] \n");
    // add header guards
    hgname = fName;
    std::transform(hgname.begin(), hgname.end(), hgname.begin(), [](unsigned char c) {
                       return std::toupper(c);
                   } );
    hgname = "SOFIE_" + hgname + "_HXX";
    fGC += "\n#ifndef " + hgname + "\n";
    fGC += "#define " + hgname + "\n\n";
    // the generated code does not include SOFIE_common.hxx: it is self-contained, like the CPU one. The standard
    // headers it relies on are listed here and the helper functions it needs are emitted inline in the model namespace
    for (const char *h : {"cstdint", "cstring", "string", "vector", "memory", "sstream", "iostream", "limits",
                          "stdexcept", "algorithm", "cmath", "cassert", "cstddef", "type_traits"}) {
        fNeededStdLib.insert(h);
    }
    fNeededStdLib.insert("span");
    if (fUseWeightFile)
        fNeededStdLib.insert("fstream");
    // standard headers first (merged and sorted when the code is finalized), then alpaka and sofieBLAS, then the
    // custom operator headers
    fHeaderIncludesTail += "#include <alpaka/alpaka.hpp>\n";
    fHeaderIncludesTail += "#include <sofieBLAS/sofieBLAS.hpp>\n";
    for (auto &i : fCustomOpHeaders)
        fHeaderIncludesTail += "#include \"" + i + "\"\n";
    fGC += kHelperIncludesMarker;

    fGC += "\nnamespace SOFIE_" + fName + "{\n";
    fGC += kHelperFunctionsMarker;

    fGC += "\nusing Dim1D = alpaka::DimInt<1>;\n";
    fGC += "template<typename TDim, typename TIdx>\n";
    fGC += "inline alpaka::WorkDivMembers<TDim, TIdx> sofie_workdiv(\n";
    fGC += "alpaka::Vec<TDim, TIdx> const& numElems, TIdx blockSz = TIdx{256})\n{\n";

    fGC += "auto const numBlocks = alpaka::Vec<TDim, TIdx>::all(\n";
    fGC += "std::max(TIdx{1}, (numElems[0] + blockSz - TIdx{1}) / blockSz));\n";
    fGC += "return alpaka::WorkDivMembers<TDim, TIdx>(\n";
    fGC += "numBlocks,\n";
    fGC += "alpaka::Vec<TDim, TIdx>::all(blockSz),\n";
    fGC += "alpaka::Vec<TDim, TIdx>::all(TIdx{1}));\n";
    fGC += "}\n";

    if (fStridedInput) {
        // layout of an input tensor read through its strides: logical shape and strides (in elements)
        fGC += "template<std::size_t R>\n";
        fGC += "struct sofie_strided_layout {\n";
        fGC += "std::size_t shape[R];\n";
        fGC += "std::size_t stride[R];\n";
        fGC += "};\n";
        fGC += "// offset in the input buffer of the element with row-major (contiguous) logical index idx\n";
        fGC += "template<std::size_t R>\n";
        fGC += "ALPAKA_FN_HOST_ACC inline std::size_t sofie_strided_offset(sofie_strided_layout<R> const& layout, std::size_t idx)\n{\n";
        fGC += "std::size_t offset = 0;\n";
        fGC += "for (std::size_t d = R; d-- > 0;) {\n";
        fGC += "offset += (idx % layout.shape[d]) * layout.stride[d];\n";
        fGC += "idx /= layout.shape[d];\n";
        fGC += "}\n";
        fGC += "return offset;\n";
        fGC += "}\n";
    }
}

void RModel_Base::OutputGenerated(std::string filename, bool append) {
    // the model can be appended only if a file name is provided
    if (filename.empty()) {
        // if a file is pr
        filename = fName + ".hxx";
        append = false;
    }
    std::ofstream f;
    if (append)
        f.open(filename, std::ios_base::app);
    else
        f.open(filename);
    if (!f.is_open()) {
        throw std::runtime_error("sofie failed to open file for output generated inference code");
    }
    f << fGC;
    f.close();
}

}//SOFIE
