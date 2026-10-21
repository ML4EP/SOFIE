#ifndef SOFIE_ROPERATOR_POOL
#define SOFIE_ROPERATOR_POOL

#include "SOFIE/utils/SOFIE_common.hxx"
#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/RModel.hxx"

#include <memory>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace SOFIE {

struct RAttributes_Pool {
   // structure that contains Pool attribute
   std::string auto_pad = "NOTSET";
   int ceil_mode = 0;
   int count_include_pad = 0;     // not for MaxPool
   int storage_order = 0;         // not for AveragePool
   std::vector<size_t> dilations; // not for AveragePool
   std::vector<size_t> kernel_shape;
   std::vector<size_t> pads;
   std::vector<size_t> strides;
};

enum PoolOpMode { InvalidPool, MaxPool, AveragePool, GlobalAveragePool };

template<typename T>
class ROperator_Pool final : public ROperator_ALPAKA
{

private:

   PoolOpMode fPoolMode;

   size_t fAttrCeilMode;
   size_t fAttrCountIncludePad;
   size_t fAttrStorageOrder;
   std::string fAttrAutopad;
   std::vector<size_t> fAttrDilations;
   std::vector<size_t> fAttrKernelShape;
   std::vector<size_t> fAttrPads;
   std::vector<size_t> fAttrStrides;

   std::string fNX;
   std::string fNY;

   std::vector<size_t> fShapeX;
   std::vector<Dim> fShapeXDim; ///< shape of the input with the (possibly symbolic) batch dimension
   std::vector<size_t> fShapeY;

   std::string fBatchExpr;
   std::vector<Dim> fShapeYDim;

   std::string fType;

   size_t fDim;   // dimension of the MaxPool

public:

   std::string Name() {
      if (fPoolMode == AveragePool)  return "AveragePool";
      if (fPoolMode == MaxPool)  return "MaxPool";
      return "Invalid";
   }

   ROperator_Pool() {}

   ROperator_Pool(PoolOpMode mode, RAttributes_Pool attr, std::string nameX, std::string nameY)
      : fPoolMode(mode), fAttrCeilMode(attr.ceil_mode), fAttrCountIncludePad(attr.count_include_pad),
        fAttrStorageOrder(attr.storage_order), fAttrAutopad(attr.auto_pad),
        fAttrDilations(attr.dilations), fAttrKernelShape(attr.kernel_shape), fAttrPads(attr.pads), fAttrStrides(attr.strides),
        fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      if(std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw
            std::runtime_error("SOFIE Encountered unsupported type parsing a Pool operator");
      }
      fInputTensorNames = { fNX };
      fOutputTensorNames = { fNY };
      fKind = OperatorKind::POOL;
   }

   // function returning output shape given input
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) {
      // shape of pooling input has to be (according to ONNX): NxCxHxW
      // Where N is batch size, C : input  channels, H : input height, W = input width
      // or it can be [N, C, F1,F2,....FN] . Minimum dimension is 3
      if (input.size() != 1 ) {
         throw std::runtime_error("SOFIE" + Name() + "Op Shape inference need 1 input tensor");
      }
      if (input[0].size() < 3) {
         throw std::runtime_error("SOFIE" + Name() + "Op Shape inference only accept tensor with at least 3 dimensions");
      }
      // support only input tensors with dim = 3,4,5
      if (input[0].size() < 3 || input[0].size() >  5) {
         throw std::runtime_error("SOFIE" + Name() + "Op : tensors with dimension " + std::to_string(input[0].size()) + " are not yet supported");
      }

      if (input[0].size() -2 != fDim) {
         throw
            std::runtime_error("SOFIE Pool Op Shape inference - invalid inputs ");
      }
       // kernel shape
      size_t k1 = ((fAttrKernelShape.empty())? input[0][2] : fAttrKernelShape[0]);
      size_t k2 = (fDim > 1) ? ((fAttrKernelShape.empty()) ? input[0][3] : fAttrKernelShape[1]) : 1;
      size_t k3 = (fDim > 2) ? ((fAttrKernelShape.empty()) ? input[0][4] : fAttrKernelShape[2]) : 1;


      size_t i1 = (fDim > 1) ? ((fDim > 2) ? 3 : 2) : 1;
      size_t i2 = (fDim > 2) ? 4 : 3;
      size_t i3 = 5;

      if (fAttrDilations.empty()) {
         fAttrDilations = {1, 1, 1};
      }
      fAttrDilations.resize(3);
      if (fDim < 3) {
         fAttrDilations.resize(3, 1);
      }
           // Shape of the kernel
      fAttrKernelShape = {k1 + (fAttrDilations[0] - 1) * (k1 - 1),
                          k2 + (fAttrDilations[1] - 1) * (k2 - 1),
                          k3 + (fAttrDilations[2] - 1) * (k3 - 1)};

      if (fAttrStrides.empty()) {
         fAttrStrides = {1, 1, 1};
      }
      if (fDim < 3)
         fAttrStrides.resize(3, 1);

      if (fAttrAutopad == "NOTSET") {
         // in auto_pad is NOTSET then fAttrPads should have been set or default zero is used
         if (fAttrPads.empty()) {
            fAttrPads = {0, 0, 0, 0, 0, 0};
         }
      } else if (fAttrAutopad == "SAME_UPPER" || fAttrAutopad == "SAME_LOWER") {
         fAttrPads.assign(6, 0);
         for (size_t d = 0; d < fDim; ++d) {
            size_t inSize = input[0][d + 2];
            size_t stride_d = fAttrStrides[d];
            size_t outSize = (inSize + stride_d - 1) / stride_d;
            int totalPad = std::max(0, (int)((outSize - 1) * stride_d + fAttrKernelShape[d]) - (int)inSize);
            if (fAttrAutopad == "SAME_UPPER") {
               fAttrPads[d] = (size_t)(totalPad / 2);
               fAttrPads[d + fDim] = (size_t)(totalPad - totalPad / 2);
            } else {
               fAttrPads[d] = (size_t)(totalPad - totalPad / 2);
               fAttrPads[d + fDim] = (size_t)(totalPad / 2);
            }
         }
      } else if (fAttrAutopad != "VALID") {
         throw
            std::runtime_error("SOFIE" + Name() + "Op invalid Autopad value : " + fAttrAutopad);
      }
      // to be sure pad is vector of size 6
      if (fDim < 3) fAttrPads.resize(6, 0);

      size_t input1 = input[0][2];
      size_t input2 = (fDim > 1) ? input[0][3] : 1;
      size_t input3 = (fDim > 2) ? input[0][4] : 1;

      auto poolOutDim = [this](size_t in, size_t padBegin, size_t padEnd, size_t kern, size_t stride) -> size_t {
         size_t n = in + padBegin + padEnd - kern;
         if (!fAttrCeilMode)
            return n / stride + 1;
         return std::min((n + stride - 1) / stride, (in - 1 + padBegin) / stride) + 1;
      };

      size_t output1 = poolOutDim(input1, fAttrPads[0], fAttrPads[i1], fAttrKernelShape[0], fAttrStrides[0]);

      size_t batch_size = input[0][0];        // first element in input tensor
      size_t output_channels = input[0][1];   // first element in output tensor

      std::vector<std::vector<size_t>> ret({{ batch_size, output_channels, output1 }});

      if (fDim == 1)
         return ret;

      size_t output2 = poolOutDim(input2, fAttrPads[1], fAttrPads[i2], fAttrKernelShape[1], fAttrStrides[1]);
      // output is N x C x OH x OW
      ret[0].push_back(output2);
      if (fDim == 2)
         return ret;

      size_t output3 = poolOutDim(input3, fAttrPads[2], fAttrPads[i3], fAttrKernelShape[2], fAttrStrides[2]);

      // output is N x C x OH x OW x OD
      ret[0].push_back(output3);
      return ret;
   }

   void Initialize(RModel& model) override {

      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw
            std::runtime_error("SOFIE Pool op Input Tensor " + fNX + " is not found in model");
      }
      auto dimShapeX = model.GetDimTensorShape(fNX);
      if (dimShapeX.size() < 3 || dimShapeX.size()  > 5) {
         std::cout << fNX << " : " << ConvertDimShapeToString(dimShapeX) << std::endl;
         throw
            std::runtime_error("SOFIE Pool Op input data tensor" + fNX + " is not of 3,4 or 5 dimensions");
      }
      // channel and spatial dims must be static (see fBatchExpr's comment); only the
      // batch dim may be dynamic.
      for (size_t i = 1; i < dimShapeX.size(); i++) {
         if (dimShapeX[i].isParam)
            throw std::runtime_error("SOFIE Pool Op: only the batch (first) dimension of "
               + fNX + " may be dynamic - channel and spatial dimensions must be static");
      }
      fBatchExpr = dimShapeX[0].GetVal();
      fShapeXDim = dimShapeX;
      fHasStridedInput = model.IsStridedInputTensor(fNX);
      fShapeX.resize(dimShapeX.size());
      fShapeX[0] = dimShapeX[0].isParam ? size_t(1) : dimShapeX[0].dim;
      for (size_t i = 1; i < dimShapeX.size(); i++)
         fShapeX[i] = dimShapeX[i].dim;
       fDim = fShapeX.size() - 2;
      // case of GlobalAveragePool. It is a pool case with kernel shape == image shape
      if (fPoolMode == GlobalAveragePool) {
         fPoolMode = AveragePool;
         fAttrKernelShape.resize(3);
         fAttrKernelShape[0] = fShapeX[2];
         if (fDim > 1)
            fAttrKernelShape[1] = fShapeX[3];
         if (fDim > 2)
            fAttrKernelShape[2] = fShapeX[4];
         fAttrAutopad = "VALID";
         fAttrPads = {0, 0, 0, 0, 0, 0 };
         assert(fAttrStrides.empty());
      }
      // find shape of Y and add it in the list of intermediate tensors
      fShapeY = ShapeInference({fShapeX})[0];
      fShapeYDim.assign(fShapeY.begin(), fShapeY.end());
      fShapeYDim[0] = dimShapeX[0];   // restore the true (possibly symbolic) batch dim
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeYDim);

      // need cmath for INFINITY when using MaxPool
      if (fPoolMode == MaxPool) model.AddNeededStdLib("cmath");

   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   bool SupportsStridedInput() const override { return true; }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;

      if (fShapeX.empty() || fShapeY.empty()) {
         throw std::runtime_error("SOFIE Pool Op called to Generate without being initialized first");
      }

      std::stringstream out;

      out << "\n//----  operator " << Name() << "  " << OpName << "\n";
      out << "{\n"; // create a new scope to avoid name clash
      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      if (fHasStridedInput)
         out << GenerateStridedOffsetLambda(OpName, fNX, fShapeXDim);

      assert(fShapeX[0] == fShapeY[0]);
      assert(fShapeX[1] == fShapeY[1]);
      assert(fAttrPads.size() == 6);
      assert(fAttrKernelShape.size() == 3);
      auto clipToInput = [this](int upper, size_t size) {
         return (fAttrCeilMode && upper > (int)size) ? (int)size : upper;
      };
      // find lower bounds of filtered area
      int hmin = - fAttrPads[0];   // minimum lower bound value of filter area
      int hmax = clipToInput(fShapeX[2] + fAttrPads[fDim] - fAttrKernelShape[0] + (fAttrCeilMode ? (int)fAttrStrides[0] : 1),
                             fShapeX[2]);
      int wmin,wmax,dmin,dmax;

      if(fDim >= 2){
         wmin = -fAttrPads[1]; // minimum lower bound value of filter area
         wmax = clipToInput(fShapeX[3] + fAttrPads[fDim + 1] - fAttrKernelShape[1] + (fAttrCeilMode ? (int)fAttrStrides[1] : 1),
                            fShapeX[3]);
      }
      else{
         wmin=1;
         wmax=1;
      }
      if(fDim == 3){
         dmin = -fAttrPads[2]; // minimum lower bound value of filter area
         dmax = clipToInput(fShapeX[4] + fAttrPads[fDim + 2] - fAttrKernelShape[2] + (fAttrCeilMode ? (int)fAttrStrides[2] : 1),
                            fShapeX[4]);
      }
      else{
         dmin=1;
         dmax=1;
      }
      out << "constexpr int hsize = " << fShapeX[2] << ";\n";
      out << "constexpr int hmin = " << hmin << ";\n";
      out << "constexpr int hmax = " << hmax << ";\n";
      out << "constexpr int kh = " << fAttrKernelShape[0] << ";\n";
      if (fDim > 1) {
         size_t wsize = fShapeX[3];
         out << "constexpr int wsize = " << wsize << ";\n";
         out << "constexpr int wmin = " << wmin << ";\n";
         out << "constexpr int wmax = " << wmax << ";\n";
         out << "constexpr int kw = " << fAttrKernelShape[1] << ";\n";
         if (fDim > 2) {
            size_t dsize = fShapeX[4];
            out << "constexpr int dsize = " << dsize << ";\n";
            out << "constexpr int dwsize = " << dsize*wsize << ";\n"; // hstride
            out << "constexpr int dmin = " << dmin << ";\n";
            out << "constexpr int dmax = " << dmax << ";\n";
            out << "constexpr int kd = " << fAttrKernelShape[2] << ";\n";
         }
      }


      bool doPadding = false;
      for ( auto & e : fAttrPads)
         doPadding |= (e > 0);

      bool dynamicDivisor = doPadding || fAttrCeilMode;
      auto windowExtent = [this](const std::string &var, const std::string &kern, size_t size, size_t padEnd) {
         std::string hi = std::to_string(fAttrCountIncludePad ? size + padEnd : size);
         std::string lo = fAttrCountIncludePad ? var : "(" + var + " > 0 ? " + var + " : 0)";
         return "((" + var + " + " + kern + " < " + hi + " ? " + var + " + " + kern + " : " + hi + ")"
                " - " + lo + ")";
      };


      if(fDim==1){
         // loop on batches and channels
         out << "size_t outIndex = 0;\n";
         out << "for (size_t n = 0; n < static_cast<size_t>(" << fBatchExpr << ") * " << fShapeX[1] << "; n++) {\n";
         out << "size_t inputOffset = n*" << fShapeX[2] << ";\n";
         out << "for (int i = hmin; i < hmax; i+=" << fAttrStrides[0] << ") {\n";
         // loop on elements of filter region to compute maximum
         if (fPoolMode == MaxPool)
            out << "float value = -INFINITY;\n";
         else if (fPoolMode == AveragePool) {
            out << "float value = 0;\n";
            if (dynamicDivisor)
               out << "const int nsum = "
                   << windowExtent("i", "kh", fShapeX[2], fAttrPads[fDim]) << ";\n";
            else
               out << "constexpr int nsum = kh;\n";
         }
         // loop on rows of filtered region
         out << "for (int l = i;  l < i + kh; l++) {\n";
         out << "if (l < 0 || l >= hsize) continue;\n";
         out << "int index = inputOffset + l;\n";
         if (fPoolMode == MaxPool) {
         out << "auto xval = tensor_" << fNX << "[index];\n";
         out << "if (xval > value) value = xval;\n";
         }
         else if (fPoolMode == AveragePool) {
            // compute sum of values
            out << "value += tensor_" << fNX << "[index];\n";
         }
          out << "}\n"; // end loop on region elements
         if (fPoolMode == AveragePool) {
         // compute average
         out << "value /= float(nsum);\n";
         }

         out << "tensor_" << fNY << "[outIndex++] = value;\n";

         out << "}\n";   // end loop on i (image rows)
         out << "}\n";  // end loop on c*b
      }
      else if(fDim==2){
         // loop on batches and channels
         out << "size_t outIndex = 0;\n";
         out << "for (size_t n = 0; n < static_cast<size_t>(" << fBatchExpr << ") * " << fShapeX[1] << "; n++) {\n";
         out << "size_t inputOffset = n*" << fShapeX[2]*fShapeX[3] << ";\n";
         out << "for (int i = hmin; i < hmax; i+=" << fAttrStrides[0] << ") {\n";
         out << "for (int j = wmin; j < wmax; j+=" << fAttrStrides[1] << ") {\n";
         // loop on elements of filter region to compute maximum
         if (fPoolMode == MaxPool)
            out << "float value = -INFINITY;\n";
         else if (fPoolMode == AveragePool) {
            out << "float value = 0;\n";
            if (dynamicDivisor)
               out << "const int nsum = "
                   << windowExtent("i", "kh", fShapeX[2], fAttrPads[fDim]) << " * "
                   << windowExtent("j", "kw", fShapeX[3], fAttrPads[fDim + 1]) << ";\n";
            else
               out << "constexpr int nsum = kw*kh;\n";
         }
         // loop on rows of filtered region
         out << "for (int l = i;  l < i + kh; l++) {\n";
         out << "if (l < 0 || l >= hsize) continue;\n";
         // loop on columns of filtered region
         out << "for (int m = j; m < j + kw; m++) {\n";
         out << "if (m < 0 || m >= wsize) continue;\n";
         out << "int index = inputOffset + l*wsize + m;\n";
         if (fPoolMode == MaxPool) {
         out << "auto xval = tensor_" << fNX << "[index];\n";
         out << "if (xval > value) value = xval;\n";
         }
         else if (fPoolMode == AveragePool) {
            // compute sum of values
            out << "value += tensor_" << fNX << "[index];\n";
         }
         out << "}\n";
         out << "}\n"; // end loop on region elements
         if (fPoolMode == AveragePool) {
            // compute average
            out << "value /= float(nsum);\n";
         }
         out << "tensor_" << fNY << "[outIndex++] = value;\n";
         out << "}\n";   // end loop on j (columns of image)
         out << "}\n";   // end loop on i (image rows)
         out << "}\n";  // end loop on c*b
      }
      else if(fDim==3){
         // loop on batches and channels
         out << "size_t outIndex = 0;\n";
         out << "for (size_t n = 0; n < static_cast<size_t>(" << fBatchExpr << ") * " << fShapeX[1] << "; n++) {\n";
         out << "size_t inputOffset = n*" << fShapeX[2]*fShapeX[3]*fShapeX[4] << ";\n";
         out << "for (int i = hmin; i < hmax; i+=" << fAttrStrides[0] << ") {\n";
         out << "for (int j = wmin; j < wmax; j+=" << fAttrStrides[1] << ") {\n";
         out << "for (int k = dmin; k < dmax; k+=" << fAttrStrides[2] << ") {\n";
         // loop on elements of filter region to compute maximum
         if (fPoolMode == MaxPool)
            out << "float value = -INFINITY;\n";
         else if (fPoolMode == AveragePool) {
            out << "float value = 0;\n";
            if (dynamicDivisor)
               out << "const int nsum = "
                   << windowExtent("i", "kh", fShapeX[2], fAttrPads[fDim]) << " * "
                   << windowExtent("j", "kw", fShapeX[3], fAttrPads[fDim + 1]) << " * "
                   << windowExtent("k", "kd", fShapeX[4], fAttrPads[fDim + 2]) << ";\n";
            else
               out << "constexpr int nsum = kw*kh*kd;\n";
         }
         // loop on rows of filtered region
         out << "for (int l = i;  l < i + kh; l++) {\n";
         out << "if (l < 0 || l >= hsize) continue;\n";
         // loop on columns of filtered region
         out << "for (int m = j; m < j + kw; m++) {\n";
         out << "if (m < 0 || m >= wsize) continue;\n";
         // loop on layers of filtered region
         out << "for (int p = k; p < k + kd; p++) {\n";
         out << "if (p < 0 || p >= dsize) continue;\n";
         out << "int index = inputOffset + l*dwsize + m*dsize + p;\n";

         if (fPoolMode == MaxPool) {
            out << "auto xval = tensor_" << fNX << "[index];\n";
            out << "if (xval > value) value = xval;\n";
         }
         else if (fPoolMode == AveragePool) {
            // compute sum of values
            out << "value += tensor_" << fNX << "[index];\n";
         }
         out << "}\n";
         out << "}\n";
         out << "}\n"; // end loop on region elements
         if (fPoolMode == AveragePool) {
            // compute average
            out << "value /= float(nsum);\n";
         }

         out << "tensor_" << fNY << "[outIndex++] = value;\n";
         out << "}\n" ; // end loop on k (layers of image)
         out << "}\n";   // end loop on j (columns of image)
         out << "}\n";   // end loop on i (image rows)
         out << "}\n";  // end loop on c*b
      }
      // end scope
      out << "}\n";

      std::string code = out.str();
      if (fHasStridedInput) {
         // all the reads of the input use the contiguous index of the element
         const std::string plain = "tensor_" + fNX + "[index]";
         const std::string strided = "tensor_" + fNX + "[xoff_" + OpName + "(index)]";
         for (size_t pos = code.find(plain); pos != std::string::npos; pos = code.find(plain, pos + strided.size()))
            code.replace(pos, plain.size(), strided);
      }
      return code;
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Pool called to Generate without being initialized first");
      if (fPoolMode != MaxPool && fPoolMode != AveragePool)
         return "";

      const bool isAvg = (fPoolMode == AveragePool);
      bool doPadding = false;
      for (auto & e : fAttrPads) doPadding |= (e > 0);
      // count_include_pad == 0 with padding: divide by the in-bounds cells counted
      // at run time; otherwise by the constant kernel area
      const bool runtimeCount = isAvg && fAttrCountIncludePad == 0 && doPadding;

      const std::string kname = (isAvg ? "AvgPoolKernel_" : "MaxPoolKernel_") + opName;

      // Mode dependent fragments, so the index math stays shared across 1D/2D/3D.
      auto emitInit = [&]() {
         std::string s;
         if (isAvg) {
            s += "T value = static_cast<T>(0);\n";
            if (runtimeCount) s += "int count = 0;\n";
         } else {
            s += "T value = static_cast<T>(-INFINITY);\n";
         }
         return s;
      };
      // a strided input is read through its strides, from the logical (contiguous) index of its elements
      auto xAt = [&](const std::string &xidx) {
         return fHasStridedInput ? StridedKernelRead("X", "layoutX", xidx) : "X[" + xidx + "]";
      };
      auto emitAccum = [&](const std::string & xidx) {
         std::string s;
         if (isAvg) {
            s += "value += " + xAt(xidx) + ";\n";
            if (runtimeCount) s += "++count;\n";
         } else {
            s += "T xv = " + xAt(xidx) + ";\n";
            s += "if (xv > value) value = xv;\n";
         }
         return s;
      };
      auto emitFinal = [&](const std::string & area) {
         std::string s;
         if (isAvg)
            s += "value /= static_cast<T>(" + (runtimeCount ? std::string("count") : area) + ");\n";
         return s;
      };

      std::stringstream op;
      op << "\n//------ " << (isAvg ? "AVGPOOL" : "MAXPOOL") << "_KERNEL_ALPAKA\n";
      op << "struct " << kname << " {\n";
      op << "template<typename TAcc, typename T>\n";
      op << "ALPAKA_FN_ACC void operator()(\n";
      op << "TAcc const& acc,\n";
      op << "T const* __restrict__ X,\n";
      op << "T* __restrict__ Y,\n";
      op << "std::size_t const totalOut";
      if (fHasStridedInput)
         op << ",\n" << "sofie_strided_layout<" << fShapeX.size() << "> const layoutX";
      op << ") const {\n\n";

      if (fDim == 1) {
         op << "constexpr int H        = " << fShapeX[2]          << ";\n";
         op << "constexpr int OH       = " << fShapeY[2]          << ";\n";
         op << "constexpr int kh       = " << fAttrKernelShape[0] << ";\n";
         op << "constexpr int sh       = " << fAttrStrides[0]     << ";\n";
         op << "constexpr int pad_top  = " << fAttrPads[0]        << ";\n\n";

         op << "auto const tid    = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op << "auto const stride = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

         op << "for (std::size_t idx = tid; idx < totalOut; idx += stride) {\n";
         op << "int oh = idx % OH;\n";
         op << "int nc = idx / OH;\n";
         op << "int i  = oh * sh - pad_top;\n";
         op << "std::size_t base = static_cast<std::size_t>(nc) * H;\n\n";
         op << emitInit();
         op << "for (int l = i; l < i + kh; ++l) {\n";
         op << "if (l < 0 || l >= H) continue;\n";
         op << emitAccum("base + l");
         op << "}\n";
         op << emitFinal("kh");
         op << "Y[idx] = value;\n";
         op << "}\n";
      }
      else if (fDim == 2) {
         op << "constexpr int H        = " << fShapeX[2]          << ";\n";
         op << "constexpr int W        = " << fShapeX[3]          << ";\n";
         op << "constexpr int OH       = " << fShapeY[2]          << ";\n";
         op << "constexpr int OW       = " << fShapeY[3]          << ";\n";
         op << "constexpr int kh       = " << fAttrKernelShape[0] << ";\n";
         op << "constexpr int kw       = " << fAttrKernelShape[1] << ";\n";
         op << "constexpr int sh       = " << fAttrStrides[0]     << ";\n";
         op << "constexpr int sw       = " << fAttrStrides[1]     << ";\n";
         op << "constexpr int pad_top  = " << fAttrPads[0]        << ";\n";
         op << "constexpr int pad_left = " << fAttrPads[2]        << ";\n\n";

         op << "auto const tid    = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op << "auto const stride = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

         op << "for (std::size_t idx = tid; idx < totalOut; idx += stride) {\n";
         op << "int ow = idx % OW;\n";
         op << "int oh = (idx / OW) % OH;\n";
         op << "int nc = idx / (OH * OW);\n";
         op << "int i  = oh * sh - pad_top;\n";
         op << "int j  = ow * sw - pad_left;\n";
         op << "std::size_t base = static_cast<std::size_t>(nc) * (H * W);\n\n";
         op << emitInit();
         op << "for (int l = i; l < i + kh; ++l) {\n";
         op << "if (l < 0 || l >= H) continue;\n";
         op << "for (int m = j; m < j + kw; ++m) {\n";
         op << "if (m < 0 || m >= W) continue;\n";
         op << emitAccum("base + l * W + m");
         op << "}\n";
         op << "}\n";
         op << emitFinal("kh * kw");
         op << "Y[idx] = value;\n";
         op << "}\n";
      }
      else if (fDim == 3) {
         op << "constexpr int H         = " << fShapeX[2]          << ";\n";
         op << "constexpr int W         = " << fShapeX[3]          << ";\n";
         op << "constexpr int D         = " << fShapeX[4]          << ";\n";
         op << "constexpr int OH        = " << fShapeY[2]          << ";\n";
         op << "constexpr int OW        = " << fShapeY[3]          << ";\n";
         op << "constexpr int OD        = " << fShapeY[4]          << ";\n";
         op << "constexpr int kh        = " << fAttrKernelShape[0] << ";\n";
         op << "constexpr int kw        = " << fAttrKernelShape[1] << ";\n";
         op << "constexpr int kd        = " << fAttrKernelShape[2] << ";\n";
         op << "constexpr int sh        = " << fAttrStrides[0]     << ";\n";
         op << "constexpr int sw        = " << fAttrStrides[1]     << ";\n";
         op << "constexpr int sd        = " << fAttrStrides[2]     << ";\n";
         op << "constexpr int pad_top   = " << fAttrPads[0]        << ";\n";
         op << "constexpr int pad_left  = " << fAttrPads[2]        << ";\n";
         op << "constexpr int pad_front = " << fAttrPads[4]        << ";\n\n";

         op << "auto const tid    = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op << "auto const stride = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

         op << "for (std::size_t idx = tid; idx < totalOut; idx += stride) {\n";
         op << "int od = idx % OD;\n";
         op << "int ow = (idx / OD) % OW;\n";
         op << "int oh = (idx / (OD * OW)) % OH;\n";
         op << "int nc = idx / (OD * OW * OH);\n";
         op << "int i  = oh * sh - pad_top;\n";
         op << "int j  = ow * sw - pad_left;\n";
         op << "int k  = od * sd - pad_front;\n";
         op << "std::size_t base = static_cast<std::size_t>(nc) * (H * W * D);\n\n";
         op << emitInit();
         op << "for (int l = i; l < i + kh; ++l) {\n";
         op << "if (l < 0 || l >= H) continue;\n";
         op << "for (int m = j; m < j + kw; ++m) {\n";
         op << "if (m < 0 || m >= W) continue;\n";
         op << "for (int p = k; p < k + kd; ++p) {\n";
         op << "if (p < 0 || p >= D) continue;\n";
         op << emitAccum("base + l * (W * D) + m * D + p");
         op << "}\n";
         op << "}\n";
         op << "}\n";
         op << emitFinal("kh * kw * kd");
         op << "Y[idx] = value;\n";
         op << "}\n";
      }
      else {
         return "";
      }

      op << "}\n";
      op << "};\n";

      return op.str();
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fPoolMode == MaxPool)
         return "MaxPoolKernel_" + opName + " maxPoolKernel_" + opName + ";\n";
      if (fPoolMode == AveragePool)
         return "AvgPoolKernel_" + opName + " avgPoolKernel_" + opName + ";\n";
      return "";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Pool called to Generate without being initialized first");
      if (fPoolMode != MaxPool && fPoolMode != AveragePool)
         return "";

      const bool isAvg = (fPoolMode == AveragePool);
      // Use fShapeYDim (the true, possibly-symbolic batch dim), not fShapeY (which
      // holds a dummy batch value of 1 when the real batch dim is dynamic).
      std::string totalOut = ConvertDimShapeToLength(fShapeYDim);
      std::string kname = (isAvg ? "avgPoolKernel_" : "maxPoolKernel_") + opName;

      std::stringstream out;
      out << "\n//------ " << (isAvg ? "AVGPOOL" : "MAXPOOL") << "_GPU_ALPAKA\n";
      if (fHasStridedInput)
         out << GenerateStridedBroadcastLayout(opName + "_X", fNX, fShapeXDim, fShapeXDim.size(), fShapeXDim);
      out << "auto const elementsPerThread_" << fNY << " = Vec::all(static_cast<Idx>(1));\n";
      out << "auto const elementsPerGrid_"   << fNY << " = Vec::all(Idx{static_cast<Idx>(" << totalOut << ")});\n";
      out << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";

      out << "auto task_" << fNY << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
          << ", " << kname
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", static_cast<Idx>(" << totalOut << ")"
          << (fHasStridedInput ? ", layout_" + opName + "_X" : std::string()) << ");\n";
      out << "alpaka::enqueue(queue, task_" << fNY << ");\n";

      return out.str();
   }
};

} // namespace SOFIE


#endif
