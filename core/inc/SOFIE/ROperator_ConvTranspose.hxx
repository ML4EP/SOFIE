#ifndef SOFIE_ROPERATOR_CONVTRANSPOSE_HXX
#define SOFIE_ROPERATOR_CONVTRANSPOSE_HXX

#include <SOFIE/SOFIE_common.hxx>
#include <SOFIE/ROperator.hxx>
#include <SOFIE/RModel.hxx>

#include <memory>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace SOFIE {

/*! \brief Transposed Convolution operator
 *
 * Inference code generation for a transposed convolution layer.
 * See the <a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#convtranspose">ONNX documentation</a> for
 * details about the transposed conv layer.
 */
template <typename T>
class ROperator_ConvTranspose final : public ROperator {
private:
   std::string fAttrAutopad;
   std::vector<size_t> fAttrDilations;
   size_t fAttrGroup;
   std::vector<size_t> fAttrKernelShape;
   std::vector<size_t> fAttrOutputPadding;
   std::vector<size_t> fAttrOutputShape;
   std::vector<size_t> fAttrPads;
   std::vector<size_t> fAttrStrides;

   std::string fNX;
   std::string fNW;
   std::string fNB;
   std::string fNBroadcastedB;
   bool fUseSession = false;
   std::string fNY;

   std::string fConvK;
   std::string fImcol;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeW;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeY;

   std::string fType;

   size_t fDim; // dimension of the convolution

public:
   /*! Default constructor of ROperator_ConvTranspose */
   ROperator_ConvTranspose() {}

   /*! \brief Constructor of ROperator_ConvTranspose from the attributes
    *
    * \param autopad padding
    * \param dilations dilations of the kernel
    * \param group number of groups
    * \param kernelShape shape of the kernel
    * \param outputPadding padding of the output
    * \param outputShape shape of the output
    * \param pads padding of the input
    * \param strides strides
    * \param nameX name of the input
    * \param nameW name of the weight
    * \param nameB name of the bias
    * \param nameY name of the output
    */
   ROperator_ConvTranspose(std::string autopad, std::vector<size_t> dilations, size_t group,
                           std::vector<size_t> kernelShape, std::vector<size_t> outputPadding,
                           std::vector<size_t> outputShape, std::vector<size_t> pads, std::vector<size_t> strides,
                           std::string nameX, std::string nameW, std::string nameB, std::string nameY)
      : fAttrAutopad(autopad),
        fAttrDilations(dilations),
        fAttrGroup(group),
        fAttrKernelShape(kernelShape),
        fAttrOutputPadding(outputPadding),
        fAttrOutputShape(outputShape),
        fAttrPads(pads),
        fAttrStrides(strides),
        fNX(UTILITY::Clean_name(nameX)),
        fNW(UTILITY::Clean_name(nameW)),
        fNB(UTILITY::Clean_name(nameB)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = {fNX, fNW};
      fOutputTensorNames = {fNY};
      if (!fNB.empty()) {
         fInputTensorNames.emplace_back(fNB);
      }

      if (std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw std::runtime_error("SOFIE Encountered unsupported type parsing a Conv operator");
      }
   }

   /*! \brief Infers the shape of the input tensors
    * \param input shape of the input tensors
    */
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> /*input*/);

   /*! \brief Initialize the model
    * \param model Model
    */
   void Initialize(RModel &) override;

   /*! \brief Generate code for initializing the op
    */
   std::string GenerateInitCode() override;

   /*! \brief Generate the inference code
    * \param opName name of the operator
    */
   std::string Generate(std::string opName) override;

   /*! \brief Returns the blas routines needed to compile the generated code
    */
   std::vector<std::string> GetBlasRoutines() override { return {std::string("Gemm"), std::string("Axpy")}; }
};

template <typename T>
auto ROperator_ConvTranspose<T>::ShapeInference(std::vector<std::vector<size_t>> input)
   -> std::vector<std::vector<size_t>>
{
   const std::vector<size_t> &inputShape = input[0];
   const std::vector<size_t> &weightShape = input[1];
   size_t size = inputShape.size();
   fDim = size - 2;
   if (fAttrGroup == 0)
      fAttrGroup = 1;
   if (fAttrStrides.empty()) {
      fAttrStrides = std::vector<size_t>(fDim, 1);
   }
   if (fAttrDilations.empty()) {
      fAttrDilations = std::vector<size_t>(fDim, 1);
   }
   if (fAttrKernelShape.empty()) {
      fAttrKernelShape.resize(fDim);
      for (size_t i = 0; i < fDim; i++)
         fAttrKernelShape[i] = fShapeW[i + 2] + (fAttrDilations[i] - 1) * (fShapeW[i + 2] - 1);
   }
   if (fAttrOutputPadding.empty())
      fAttrOutputPadding = std::vector<size_t>(fDim, 0);

   std::vector<size_t> outShape(size);
   outShape[0] = inputShape[0];
   outShape[1] = weightShape[1] * fAttrGroup;

   if (fAttrPads.empty()) {
      fAttrPads = std::vector<size_t>(2 * fDim, 0);

      if (fAttrAutopad != "NOTSET") {
         throw std::runtime_error("ConvTranspose with padding SAME_UPPER or SMAE_LOWER not supported");
      }
   }
   if (fAttrOutputShape.empty()) {
      fAttrOutputShape.resize(fDim);
      for (size_t i = 0; i < fDim; i++) {
         size_t j = i + 2;
         fAttrOutputShape[i] = fAttrStrides[i] * (inputShape[j] - 1) + fAttrKernelShape[i] + fAttrOutputPadding[i] -
                               fAttrPads[i] - fAttrPads[fDim + i];
      }
   } else {
      fAttrPads = std::vector<size_t>(2 * fDim, 0);
      for (size_t i = 0; i < fDim; ++i) {
         size_t input_shape = inputShape[i + 2];
         size_t output_shape = fAttrOutputShape[i];
         size_t kernel_shape = weightShape[i + 2];

         size_t stride = fAttrStrides[i];
         size_t dilation = fAttrDilations[i];
         size_t output_padding = fAttrOutputPadding[i];

         size_t effective_kernel_shape = (kernel_shape - 1) * dilation + 1;
         size_t expected_shape_without_pad = (input_shape - 1) * stride + output_padding + effective_kernel_shape;

         if (expected_shape_without_pad < output_shape) {
            throw std::runtime_error("ConvTranspose: explicitly set output_shape is too large for "
                                     "the given input and kernel shapes.");
         }

         size_t total_padding = expected_shape_without_pad - output_shape;

         fAttrPads[i + fDim] = total_padding / 2;
         fAttrPads[i] = total_padding - fAttrPads[i + fDim];
      }
   }

   for (size_t i = 0; i < fDim; i++)
      outShape[i + 2] = fAttrOutputShape[i];
   std::vector<std::vector<size_t>> ret({outShape});
   return ret;
}

template <typename T>
void ROperator_ConvTranspose<T>::Initialize(RModel &model)
{
   fUseSession = model.UseSession();

   if (!model.CheckIfTensorAlreadyExist(fNX)) {
      throw std::runtime_error("SOFIE Conv Transpose op Input Tensor " + fNX + " is not found in model");
   }
   fShapeX = model.GetTensorShape(fNX);
   if (fShapeX.size() < 3 || fShapeX.size() > 5) {
      std::cout << fNX << " : " << ConvertShapeToString(fShapeX) << std::endl;
      throw std::runtime_error("SOFIE Conv Transpose Op input data tensor" + fNX +
                               " is not of 3,4 or 5 dimensions");
   }
   fDim = fShapeX.size() - 2;
   if (!model.CheckIfTensorAlreadyExist(fNW)) {
      throw std::runtime_error("SOFIE Conv op Input weight Tensor " + fNW + " is not found in model");
   }
   fShapeW = model.GetTensorShape(fNW);
   if (fShapeW.size() < 3 || fShapeW.size() > 5) {
      std::cout << fNW << " : " << ConvertShapeToString(fShapeW) << std::endl;
      throw std::runtime_error("SOFIE Conv Transpose Op input weight tensor" + fNW +
                               " is not of 3,4 or 5 dimensions");
   }
   fShapeY = ShapeInference({fShapeX, fShapeW})[0];

   model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
   if (fNB != "") {
      if (!model.CheckIfTensorAlreadyExist(fNB)) {
         throw std::runtime_error("SOFIE ConvTrans op Input Tensor " + fNB + " is not found in model");
      }
      fShapeB = model.GetTensorShape(fNB);
      if (fShapeB.size() < 1)
         throw std::runtime_error("SOFIE ConvTrans op: Bias Tensor has empty shape");

      size_t bsize = ConvertShapeToLength(fShapeB);
      size_t ysize = ConvertShapeToLength(fShapeY);
      bool broadcast_needed = (bsize != ysize);
      if (broadcast_needed) {
         if (bsize != fShapeY[1])
            throw std::runtime_error("SOFIE ConvTrans op: Bias Tensor has wrong shape: " +
                                     ConvertShapeToString(fShapeB));

         if (fType != "float")
            throw std::runtime_error(
               "SOFIE ConvTrans op: Broadcasting for non-float type tensors is not supported");
         if (!fUseSession) {
            auto original_data = model.GetInitializedTensorData(fNB);
            std::shared_ptr<void> new_data_ptr(
               UTILITY::BroadcastConvBias<float>(static_cast<float *>(original_data.get()), bsize, fShapeY),
               std::default_delete<float[]>());

            model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), fShapeY, new_data_ptr);
            fShapeB = model.GetTensorShape(fNB);
            fNBroadcastedB = fNB;
         } else {
            fNBroadcastedB = "Broadcasted" + fNB;
            model.AddIntermediateTensor(fNBroadcastedB, model.GetTensorType(fNB), fShapeY);
         }
      } else {
         if (fShapeY != fShapeB)
            throw std::runtime_error("SOFIE ConvTrans op: Broadcasting is not needed but bias has wrong shape" +
                                     ConvertShapeToString(fShapeB));
         fNBroadcastedB = fNB;
      }
   }

   size_t kernelSize = 1;
   size_t inputSize = 1;
   for (size_t i = 0; i < fDim; i++) {
      inputSize *= fShapeX[2 + i];
      kernelSize *= fAttrKernelShape[i];
   }

   std::vector<size_t> shape1 = {fShapeW[0], fShapeW[1], kernelSize};
   std::vector<size_t> shape2 = {fShapeW[1], kernelSize, inputSize};
   model.AddIntermediateTensor(fNY + "_f", ConvertStringToType(fType), shape1);
   model.AddIntermediateTensor(fNY + "_xcol", ConvertStringToType(fType), shape2);
   fConvK = fNY + "_f";
   fImcol = fNY + "_xcol";
   fOutputTensorNames.emplace_back(fConvK);
   fOutputTensorNames.emplace_back(fImcol);

   model.AddNeededHelperFunction("col2im");
   if (!fNB.empty())
      model.AddNeededHelperFunction("BroadcastConvBias");
}

template <typename T>
std::string ROperator_ConvTranspose<T>::GenerateInitCode()
{
   std::stringstream out;
   size_t bsize = ConvertShapeToLength(fShapeB);
   size_t ysize = ConvertShapeToLength(fShapeY);
   if (fUseSession && bsize != ysize && !fNBroadcastedB.empty()) {
      out << SP << "{\n";
      out << SP << SP << "float * data = UTILITY::BroadcastConvBias<float>(tensor_" << fNB << ", " << bsize << ", "
          << ConvertShapeToString(fShapeY) << ");\n";
      out << SP << SP << "std::copy(data, data + " << ConvertShapeToLength(fShapeY) << ", tensor_" << fNBroadcastedB
          << ");\n";
      out << SP << SP << "delete[] data;\n";
      out << SP << "}\n";
   }
   return out.str();
}

template <typename T>
std::string ROperator_ConvTranspose<T>::Generate(std::string OpName)
{
   OpName = "op_" + OpName;

   if (fShapeX.empty() || fShapeW.empty() || (fNB != "" && fShapeB.empty()) || fShapeY.empty()) {
      throw std::runtime_error("SOFIE Conv Op called to Generate without being initialized first");
   }

   std::stringstream out;

   size_t bsize = fShapeX[0];
   size_t kDepth = (fDim > 2) ? fShapeW[2] : 1;
   size_t kHeight = (fDim > 1) ? fShapeW[fDim] : 1;
   size_t kWidth = fShapeW[fDim + 1];

   size_t iDepth = (fDim > 2) ? fShapeX[2] : 1;
   size_t iHeight = (fDim > 1) ? fShapeX[fDim] : 1;
   size_t iWidth = fShapeX[fDim + 1];

   size_t oDepth = (fDim > 2) ? fShapeY[2] : 1;
   size_t oHeight = (fDim > 1) ? fShapeY[fDim] : 1;
   size_t oWidth = fShapeY[fDim + 1];

   out << "\n//----  operator ConvTranspose " << OpName << "\n";

   size_t id = (fDim > 2) ? fDim - 3 : 2;
   size_t ih = (fDim > 1) ? fDim - 2 : 1;
   size_t iw = fDim - 1;
   size_t wstrideDil = fAttrDilations[iw];
   size_t hstride = kWidth;
   size_t hstrideDil = fAttrKernelShape[iw];
   if (fDim > 1)
      hstrideDil *= fAttrDilations[ih];
   size_t dstride = kHeight * kWidth;
   size_t dstrideDil = fAttrKernelShape[iw];
   if (fDim > 1)
      dstrideDil *= fAttrKernelShape[ih];
   if (fDim > 2)
      dstrideDil *= fAttrDilations[id];
   size_t icstride = kHeight * kWidth * kDepth;
   size_t icstrideDil = 1;
   for (size_t i = 0; i < fDim; i++)
      icstrideDil *= fAttrKernelShape[i];
   size_t ocstride = fShapeW[1] * icstride;
   size_t ocstrideDil = fShapeW[1] * icstrideDil;

   if (!fUseSession) {
      size_t kernelSize = fAttrKernelShape[0];
      if (fDim > 1)
         kernelSize *= fAttrKernelShape[1];
      out << SP << fType << " tensor_" << fConvK << "[" << fShapeW[0] * fShapeW[1] * kernelSize << "] = {0};\n";
   }

   out << SP << "for (std::size_t ic = 0; ic < " << fShapeW[0] << "; ic++) {\n";
   out << SP << SP << "for (std::size_t oc = 0; oc < " << fShapeW[1] << "; oc++) {\n";
   if (fDim > 2)
      out << SP << SP << SP << "for (std::size_t kd = 0; kd < " << kDepth << "; kd++) {\n";
   if (fDim > 1)
      out << SP << SP << SP << "for (std::size_t kh = 0; kh < " << kHeight << "; kh++) {\n";
   out << SP << SP << SP << SP << "for (std::size_t kw = 0; kw < " << kWidth << "; kw++) {\n";

   out << SP << SP << SP << SP << SP << "tensor_" << fConvK << "[ic * " << ocstrideDil << " + oc * " << icstrideDil;
   if (fDim > 2)
      out << " + kd * " << dstrideDil;
   if (fDim > 1)
      out << " + kh * " << hstrideDil;
   out << " + kw * " << wstrideDil << "  ] = tensor_" << fNW << "[ic * " << ocstride << " + oc * " << icstride;

   if (fDim > 2)
      out << " + kd * " << dstride;
   if (fDim > 1)
      out << " + kh * " << hstride;
   out << " + kw ];\n";


   out << SP << SP << SP << SP << "}\n";
   if (fDim > 1)
      out << SP << SP << SP << "}\n";
   if (fDim > 2)
      out << SP << SP << SP << "}\n";

   out << SP << SP << "}\n";
   out << SP << "}\n";

   out << SP << "char " << OpName << "_transA = 'N';\n";
   out << SP << "char " << OpName << "_transB = 'T';\n";
   out << SP << "int " << OpName << "_m = " << iHeight * iWidth * iDepth << ";\n";
   out << SP << "int " << OpName << "_n = " << icstrideDil * fShapeW[1] << ";\n";
   out << SP << "int " << OpName << "_k = " << fShapeW[0] << ";\n";
   out << SP << "float " << OpName << "_alpha = 1.0;\n";
   out << SP << "float " << OpName << "_beta = 0.0;\n";

   if (!fUseSession) {
      out << SP << fType << " tensor_" << fImcol << "[" << fShapeW[1] * icstrideDil * iDepth * iHeight * iWidth
          << "] = {0};\n";
   }

   out << SP << "for (size_t n = 0; n < " << bsize << "; n++) {\n";


   if (fAttrGroup == 1) {
      out << SP << SP << "size_t x_offset = n * " << fShapeX[1] * iDepth * iHeight * iWidth << ";\n";
      out << SP << SP << "size_t out_offset = n * " << fShapeY[1] * oDepth * oHeight * oWidth << ";\n";

      out << SP << SP << "BLAS::sgemm_(&" << OpName << "_transA, &" << OpName << "_transB, &" << OpName << "_m, &"
          << OpName << "_n, &" << OpName << "_k, &" << OpName << "_alpha, "
          << "tensor_" << fNX << " + x_offset, &" << OpName
          << "_m,\n";
      out << SP << SP << SP << "tensor_" << fConvK << ", &" << OpName << "_n, &" << OpName << "_beta, tensor_" << fImcol
          << ", &" << OpName << "_m);\n";

      if (fDim < 3) {
         out << SP << SP << "UTILITY::col2im<float>(tensor_" << fImcol
             << ","
             << fShapeY[1] << "," << oHeight << "," << oWidth << ",";
         if (fDim == 1)
            out << "1, " << fAttrKernelShape[0] << ",0,0," << fAttrPads[0] << "," << fAttrPads[1] << ",1,"
                << fAttrStrides[0] << ",1," << fAttrDilations[0];
         else
            out << fAttrKernelShape[0] << "," << fAttrKernelShape[1] << "," << fAttrPads[0] << "," << fAttrPads[2]
                << "," << fAttrPads[1] << "," << fAttrPads[3] << "," << fAttrStrides[0] << "," << fAttrStrides[1] << ","
                << fAttrDilations[0] << "," << fAttrDilations[1];
         out << ", tensor_" << fNY << " + out_offset);\n\n ";
      } else {
         throw std::runtime_error("SOFIE 3D Conv Transpose not yet supported");
         out << SP << SP << "UTILITY::Im2col_3d<float>(tensor_" << fNX
             << " + x_offset,"
             << fShapeX[1] << "," << oDepth << "," << oHeight << "," << oWidth << "," << fAttrKernelShape[0] << ","
             << fAttrKernelShape[1] << "," << fAttrKernelShape[2] << "," << fAttrPads[0] << "," << fAttrPads[3] << ","
             << fAttrPads[1] << "," << fAttrPads[4] << "," << fAttrPads[2] << "," << fAttrPads[5] << ","
             << fAttrStrides[0] << "," << fAttrStrides[1] << "," << fAttrStrides[2] << "," << fAttrDilations[0] << ","
             << fAttrDilations[1] << "," << fAttrDilations[2] << ",tensor_" << fImcol << ");\n\n ";
      }
   } else {
      out << SP << SP << "for (size_t g = 0; g < " << fAttrGroup << "; g++) {\n";
      out << SP << SP << "size_t x_offset = n * " << fShapeX[1] * iHeight * iWidth << " + g * "
          << fShapeX[1] * iHeight * iWidth / fAttrGroup << ";\n ";
      out << SP << SP << "size_t out_offset = n * " << fShapeY[1] * oHeight * oWidth << " + g * "
          << fShapeY[1] * oHeight * oWidth / fAttrGroup << ";\n ";

      out << SP << SP << "BLAS::sgemm_(&" << OpName << "_transA, &" << OpName << "_transB, &" << OpName << "_m, &"
          << OpName << "_n, &" << OpName << "_k, &" << OpName << "_alpha, "
          << "tensor_" << fNX << " + x_offset, &" << OpName
          << "_m,\n";
      out << SP << SP << SP << "tensor_" << fConvK << ", &" << OpName << "_n, &" << OpName << "_beta, tensor_" << fImcol
          << " , &" << OpName << "_m);\n";

      if (fDim < 3) {
         out << SP << SP << "UTILITY::col2im<float>(tensor_" << fImcol
             << ","
             << fShapeY[1] << "," << oHeight << "," << oWidth << ",";
         if (fDim == 1)
            out << "1, " << fAttrKernelShape[0] << ",0,0," << fAttrPads[0] << "," << fAttrPads[1] << ",1,"
                << fAttrStrides[0] << ",1," << fAttrDilations[0];
         else
            out << fAttrKernelShape[0] << "," << fAttrKernelShape[1] << "," << fAttrPads[0] << "," << fAttrPads[2]
                << "," << fAttrPads[1] << "," << fAttrPads[3] << "," << fAttrStrides[0] << "," << fAttrStrides[1] << ","
                << fAttrDilations[0] << "," << fAttrDilations[1];
         out << ", tensor_" << fNY << " + out_offset);\n\n ";
      } else {
         throw std::runtime_error("SOFIE 3D Conv Transpose not yet supported");

         out << SP << SP << "UTILITY::Im2col_3d<float>(tensor_" << fNX
             << " + x_offset,"
             << fShapeX[1] << "," << oDepth << "," << oHeight << "," << oWidth << "," << fAttrKernelShape[0] << ","
             << fAttrKernelShape[1] << "," << fAttrKernelShape[2] << "," << fAttrPads[0] << "," << fAttrPads[3] << ","
             << fAttrPads[1] << "," << fAttrPads[4] << "," << fAttrPads[2] << "," << fAttrPads[5] << ","
             << fAttrStrides[0] << "," << fAttrStrides[1] << "," << fAttrStrides[2] << "," << fAttrDilations[0] << ","
             << fAttrDilations[1] << "," << fAttrDilations[2] << "," << "tensor_" << fImcol << ");\n\n ";
      }


      out << SP << SP << "}\n";
   }

   out << SP << "}\n";

   if (fNBroadcastedB != "") {
      out << SP << "int " << OpName << "_size = " << fShapeY[0] * fShapeY[1] * oDepth * oHeight * oWidth << ";\n";
      out << SP << "float " << OpName << "_gamma = 1.0;\n";
      out << SP << "int " << OpName << "_incx = 1;\n";
      out << SP << "int " << OpName << "_incy = 1;\n";

      out << SP << "BLAS::saxpy_(&" << OpName << "_size, &" << OpName << "_gamma, tensor_" << fNBroadcastedB << ", &"
          << OpName << "_incx, tensor_" << fNY << ", &" << OpName << "_incy);\n";
   }

   return out.str();
}

} // namespace SOFIE

#endif
