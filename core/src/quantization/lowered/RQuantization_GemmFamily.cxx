#include "SOFIE/quantization/lowered/RQuantization_GemmFamily.hxx"

#include <stdexcept>

namespace SOFIE {

namespace {

const QuantGemmTraits kInt8Traits{
   EQuantGemmFamily::Int8, "int8", ETensorType::INT8, ETensorType::INT32,
   /*kAlignment=*/4,       /*nAlignment=*/4,
   /*integerAccumulate=*/true, /*scalesInLibrary=*/false, /*fusedCodesOutput=*/true, /*requiresOcpFloat8=*/false};

const QuantGemmTraits kE4M3Traits{
   EQuantGemmFamily::E4M3, "e4m3", ETensorType::FLOAT8E4M3FN, ETensorType::FLOAT,
   /*kAlignment=*/16,      /*nAlignment=*/4,
   /*integerAccumulate=*/false, /*scalesInLibrary=*/true, /*fusedCodesOutput=*/false, /*requiresOcpFloat8=*/true};

// Narrower signed grids (int2..int7) also store int8 values and would multiply exactly on these
// kernels; they are not selected yet because no test covers them.
bool IsSignedInt8(const QuantSpec &spec)
{
   return spec.IsDefined() && spec.encoding == EQuantEncoding::Integer && spec.bitWidth == 8 && spec.isSigned;
}

bool IsPerTensorE4M3(const QuantSpec &spec)
{
   return spec.IsDefined() && spec.encoding == EQuantEncoding::FP8E4M3 && spec.IsPerTensor();
}

const std::string kBlasType = "typename std::remove_reference_t<decltype(blas)>::";

std::string Fp8(const std::string &ptr)
{
   return "reinterpret_cast<" + kBlasType + "Fp8E4M3Type const *>(" + ptr + ")";
}

} // namespace

const QuantGemmTraits &GemmTraits(EQuantGemmFamily family)
{
   switch (family) {
   case EQuantGemmFamily::Int8: return kInt8Traits;
   case EQuantGemmFamily::E4M3: return kE4M3Traits;
   }
   throw std::logic_error("SOFIE quantization: unknown GEMM family");
}

std::optional<EQuantGemmFamily> SelectGemmFamily(const QuantSpec &x, const QuantSpec &w)
{
   if (!x.IsPerTensor())
      return std::nullopt;
   if (IsPerTensorE4M3(x) && IsPerTensorE4M3(w))
      return EQuantGemmFamily::E4M3;
   if (IsSignedInt8(x) && IsSignedInt8(w))
      return EQuantGemmFamily::Int8;
   return std::nullopt;
}

std::string QuantGemmAccumulateCall(const QuantGemmTraits &traits, const QuantGemmCall &c)
{
   const std::string dims = "'t', 'n', " + c.m + ", " + c.n + ", " + c.k;
   const std::string batchTail = std::to_string(c.batch);
   switch (traits.family) {
   case EQuantGemmFamily::Int8:
      if (c.batched)
         return "blas.int8MatmulStridedBatched(" + dims + ", " + c.a + ", " + c.strideA + "LL, " + c.b + ", " +
                c.strideB + "LL, " + c.d + ", " + c.strideD + "LL, " + batchTail + ")";
      return "blas.int8Matmul(" + dims + ", " + c.a + ", " + c.b + ", " + c.d + ")";
   case EQuantGemmFamily::E4M3: {
      const std::string bf16Null = "static_cast<" + kBlasType + "Bf16Type const *>(nullptr)";
      if (c.batched)
         return "blas.matmulFp8Fp32StridedBatched(" + dims + ", 1.0f, " + Fp8(c.a) + ", " + c.scaleA + ", " +
                c.strideA + "LL, " + Fp8(c.b) + ", " + c.scaleB + ", " + c.strideB + "LL, " + bf16Null + ", " + c.d +
                ", " + c.strideD + "LL, " + batchTail + ", Epilogue::Default)";
      return "blas.matmulFp8Fp32(" + dims + ", 1.0f, " + Fp8(c.a) + ", " + c.scaleA + ", " + Fp8(c.b) + ", " +
             c.scaleB + ", " + bf16Null + ", " + c.d + ", Epilogue::Default)";
   }
   }
   throw std::logic_error("SOFIE quantization: unknown GEMM family");
}

std::string QuantGemmFusedCodesCall(const QuantGemmTraits &traits, const QuantGemmCall &c,
                                    const QuantGemmFusedOutput &f)
{
   const std::string dims = "'t', 'n', " + c.m + ", " + c.n + ", " + c.k;
   switch (traits.family) {
   case EQuantGemmFamily::Int8:
      if (c.batched)
         return "blas.int8MatmulStridedBatched(" + dims + ", " + f.alpha + ", " + c.a + ", " + c.strideA + "LL, " +
                c.b + ", " + c.strideB + "LL, " + f.bias + ", " + c.d + ", " + c.strideD + "LL, " +
                std::to_string(c.batch) + ", " + f.epilogue + ")";
      if (!f.alphaVector.empty())
         return "blas.int8MatmulVectorAlpha(" + dims + ", " + f.alphaVector + ", " + c.a + ", " + c.b + ", " + f.bias +
                ", " + c.d + ", " + f.epilogue + ")";
      return "blas.int8Matmul(" + dims + ", " + f.alpha + ", " + c.a + ", " + c.b + ", " + f.bias + ", " + c.d + ", " +
             f.epilogue + ")";
   case EQuantGemmFamily::E4M3: break;
   }
   throw std::logic_error(std::string("SOFIE quantization: the ") + traits.label +
                          " GEMM family has no codes-output epilogue");
}

} // namespace SOFIE
