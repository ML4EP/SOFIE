#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator.hxx"
#include "onnx.hxx"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <cstring>
#include <memory>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include "SOFIE/SOFIE_common.hxx"


namespace SOFIE {

// Declaration of operators
// Unary operators
void RegisterBasicUnaryParsers(RModelParser_ONNX &parser);
// Binary operators
void RegisterBasicBinaryParsers(RModelParser_ONNX &parser);
// Nary operators
void RegisterBasicNaryParsers(RModelParser_ONNX &parser);
//Comparision Operators
void RegisterComparisionParsers(RModelParser_ONNX &parser);
//Is Operators
void RegisterBasicIsParsers(RModelParser_ONNX &parser);
extern ParserFuncSignature ParseNot;
// Reduce operators
void RegisterReduceParsers(RModelParser_ONNX &parser);
// Others
extern ParserFuncSignature ParseBatchNormalization;
extern ParserFuncSignature ParseConstant;
extern ParserFuncSignature ParseTranspose;
extern ParserFuncSignature ParseRelu;
extern ParserFuncSignature ParseTanh;
extern ParserFuncSignature ParseConv;
extern ParserFuncSignature ParseConvTranspose;
extern ParserFuncSignature ParseLeakyRelu;
extern ParserFuncSignature ParseGelu;
extern ParserFuncSignature ParseSelu;
extern ParserFuncSignature ParseSigmoid;
extern ParserFuncSignature ParseSwish;
extern ParserFuncSignature ParseGemm;
extern ParserFuncSignature ParseRNN;
extern ParserFuncSignature ParseLSTM;
extern ParserFuncSignature ParsePool;
extern ParserFuncSignature ParseReshape;
extern ParserFuncSignature ParseSlice;
extern ParserFuncSignature ParseGRU;
extern ParserFuncSignature ParseIdentity;
extern ParserFuncSignature ParseSoftmax;
extern ParserFuncSignature ParseConcat;
extern ParserFuncSignature ParseCast;
extern ParserFuncSignature ParseExpand;
extern ParserFuncSignature ParseShape;
extern ParserFuncSignature ParseMatMul;
extern ParserFuncSignature ParseLayerNormalization;
extern ParserFuncSignature ParseInstanceNormalization;
extern ParserFuncSignature ParseGather;
extern ParserFuncSignature ParseGatherND;
extern ParserFuncSignature ParseErf;
extern ParserFuncSignature ParseElu;
extern ParserFuncSignature ParseHardSigmoid;
extern ParserFuncSignature ParseHardSwish;
extern ParserFuncSignature ParseEyeLike;
extern ParserFuncSignature ParseRange;
extern ParserFuncSignature ParseTopK;
extern ParserFuncSignature ParseTile;
extern ParserFuncSignature ParseSplit;
extern ParserFuncSignature ParseIf;
extern ParserFuncSignature ParsePad;
extern ParserFuncSignature ParseWhere;
extern ParserFuncSignature ParseEinsum;
extern ParserFuncSignature ParseScatterElements;
extern ParserFuncSignature ParseTrilu;
extern ParserFuncSignature ParseAnd;
extern ParserFuncSignature ParseOr;
extern ParserFuncSignature ParseXor;
extern ParserFuncSignature ParseBitwiseAnd;
extern ParserFuncSignature ParseBitwiseOr;
extern ParserFuncSignature ParseBitwiseXor;
extern ParserFuncSignature ParseBitwiseNot;
extern ParserFuncSignature ParseNonZero;
extern ParserFuncSignature ParseScatterND;
extern ParserFuncSignature ParseRMSNorm;
extern ParserFuncSignature ParseGroupNorm;
extern ParserFuncSignature ParseCumSum;
extern ParserFuncSignature ParseSDPA;
extern ParserFuncSignature ParseMambaScan;
extern ParserFuncSignature ParseRWKVWKV6;
extern ParserFuncSignature ParseGriffinRGLRU;
extern ParserFuncSignature ParseClip;

// Declaration of fused operators
extern ParserFuseFuncSignature ParseFuseConvAdd;
extern ParserFuseFuncSignature ParseFuseGemmRelu;
extern ParserFuseFuncSignature ParseFuseBatchnormRelu;
extern ParserFuseFuncSignature ParseFuseConvTransposeAdd;
extern ParserFuseFuncSignature ParseFuseMatMulAdd;
extern std::unique_ptr<ROperator> ParseFuseL2Normalization(RModelParser_ONNX &parser, const onnx::NodeProto &reduceNode,
                         const onnx::NodeProto &divNode, float epsilon);

// Definition of  RModelParser_ONNX::OperatorsMap
struct RModelParser_ONNX::OperatorsMapImpl {
   // Registered operators
   std::unordered_map<std::string, ParserFuncSignature> fOperatorsMap;
};

// helper function to get initialized tensor data
template<typename T>
struct ExtractDataFromTP {
};
// trait function to extract data from TensorProto
template<>
struct ExtractDataFromTP<float> {
   static void Copy(onnx::TensorProto * tensor, void * data, int length) {
      if (tensor->float_data_size() != length)
         throw std::runtime_error("SOFIE - Failed to read float initialized tensor - actual size is " + std::to_string(tensor->float_data_size()));
      const auto &src = tensor->float_data();
      std::copy(src.begin(), src.end(), static_cast<float *>(data));
   }
};
template<>
struct ExtractDataFromTP<double> {
   static void Copy(onnx::TensorProto * tensor, void * data, int length) {
      if (tensor->double_data_size() != length)
         throw std::runtime_error("SOFIE - Failed to read double initialized tensor - actual size is " + std::to_string(tensor->double_data_size()));
      const auto &src = tensor->double_data();
      std::copy(src.begin(), src.end(), static_cast<double *>(data));
   }
};
template<>
struct ExtractDataFromTP<int32_t> {
   static void Copy(onnx::TensorProto * tensor, void * data, int length) {
      if (tensor->int32_data_size() != length)
         throw std::runtime_error("SOFIE - Failed to read int32 initialized tensor - actual size is " + std::to_string(tensor->int32_data_size()));
      const auto &src = tensor->int32_data();
      std::copy(src.begin(), src.end(), static_cast<int32_t *>(data));
   }
};
template<>
struct ExtractDataFromTP<int64_t> {
   static void Copy(onnx::TensorProto * tensor, void * data, int length) {
      if (tensor->int64_data_size() != length)
         throw std::runtime_error("SOFIE - Failed to read int64 initialized tensor - actual size is " + std::to_string(tensor->int64_data_size()));
      const auto &src = tensor->int64_data();
      std::copy(src.begin(), src.end(), static_cast<int64_t *>(data));
   }
};

namespace {

template <typename T>
T bswap_value(T value) noexcept
{
   static_assert(std::is_trivially_copyable_v<T>);
   std::array<char, sizeof(T)> bytes;
   std::memcpy(bytes.data(), &value, sizeof(T));
   std::reverse(bytes.begin(), bytes.end());
   T result;
   std::memcpy(&result, bytes.data(), sizeof(T));
   return result;
}

void CopyLEToHost(void *dest, const void *source, std::size_t nbytes, ETensorType tensor_type)
{
   if constexpr (std::endian::native == std::endian::little) {
      if (dest != source)
         std::memcpy(dest, source, nbytes);
   } else {
      const std::size_t size = GetTypeSize(tensor_type);
      if (size != 1 && size != 2 && size != 4 && size != 8)
         throw std::runtime_error("Data type " + ConvertTypeToString(tensor_type) + " in tensor is not supported!\n");
      if (dest != source)
         std::memcpy(dest, source, nbytes);
      auto bytes = static_cast<unsigned char *>(dest);
      for (std::size_t k = 0; k + size <= nbytes; k += size)
         std::reverse(bytes + k, bytes + k + size);
   }
}

}

std::shared_ptr<void> RModelParser_ONNX::GetInitializedTensorData(onnx::TensorProto *tensorproto, size_t tensor_size, ETensorType tensor_type)
{

   std::shared_ptr<void> data(malloc(tensor_size), free);

   if (tensorproto->data_location() != onnx::TensorProto::EXTERNAL) {
      if (tensorproto->raw_data().size() > 0) {
         if (tensorproto->raw_data().size() != tensor_size)
            throw std::runtime_error("SOFIE - Failed to read raw data of initialized tensor - actual raw size is " +
                                 std::to_string(tensorproto->raw_data().size()));

         CopyLEToHost(data.get(), tensorproto->raw_data().c_str(), tensor_size, tensor_type);
      } else {
         switch (tensor_type) {
            case ETensorType::FLOAT: {
               ExtractDataFromTP<float>::Copy(tensorproto, data.get(), tensor_size/ 4);
               break;
            }
            case ETensorType::DOUBLE: {
               ExtractDataFromTP<double>::Copy(tensorproto, data.get(), tensor_size/ 8);
               break;
            }
            case ETensorType::INT32: {
               ExtractDataFromTP<int32_t>::Copy(tensorproto, data.get(), tensor_size/ 4);
               break;
            }
            case ETensorType::INT64: {
               ExtractDataFromTP<int64_t>::Copy(tensorproto, data.get(), tensor_size/ 8);
               break;
            }
            case ETensorType::BOOL: {
               throw std::runtime_error("SOFIE - ExtractData from TP in BOOL not supported");
               break;
            }
            case ETensorType::UINT8: {
               throw std::runtime_error("SOFIE - ExtractData from TP in UINT8 not supported");
               break;
            }
            default:
               throw std::runtime_error("Data type " + ConvertTypeToString(tensor_type) + " in weight tensor is not supported!\n");
         }
      }

   }  else {

      std::string location;
      size_t offset = 0, buffer_size = 0;

      for (const auto &kv : tensorproto->external_data()) {
         if (kv.key() == "location")  location = kv.value();
         else if (kv.key() == "offset") offset = std::stoull(kv.value());
         else if (kv.key() == "length") buffer_size = std::stoull(kv.value());
      }

      std::string dataFileName = fDataFileName;
      if (dataFileName.empty())
         dataFileName = location.empty() ? fDefaultDataFileName : fModelDirectory + location;
      if (dataFileName.empty())
         throw std::runtime_error("SOFIE ONNX : tensor " + tensorproto->name() +
                                  " has external data but no data file location is available");

      if (fVerbose)
         std::cout << "Initialized data are stored externally in file " << dataFileName
                   << " at location " << location << " offset " << offset << " and with length " << buffer_size << std::endl;

      if (buffer_size != tensor_size)
         throw std::runtime_error("SOFIE ONNX : invalid stored data size vs tensor size");

      if (fDataFile.is_open() && fOpenedDataFileName != dataFileName)
         fDataFile.close();
      if (!fDataFile.is_open()) {
         fDataFile.open(dataFileName, std::ios::binary);
         if (!fDataFile.is_open())
            throw std::runtime_error("SOFIE ONNX:  error reading external weight ONNX data file " + dataFileName);
         fOpenedDataFileName = dataFileName;
      }

      fDataFile.seekg(offset);
      fDataFile.read(reinterpret_cast<char *>(data.get()), buffer_size);
      CopyLEToHost(data.get(), data.get(), buffer_size, tensor_type);
   }

   return data;
}


// Constructor of the parser
RModelParser_ONNX::RModelParser_ONNX() noexcept : fOperatorsMapImpl(std::make_unique<OperatorsMapImpl>()) {
   // Register operators
   // Unary operators
   RegisterBasicUnaryParsers(*this);
   // Binary operators
   RegisterBasicBinaryParsers(*this);
   // Nary operators
   RegisterBasicNaryParsers(*this);
   //Comparision Operators
   RegisterComparisionParsers(*this);
   RegisterBasicIsParsers(*this);
   RegisterOperator("Not", ParseNot);
   // Reduce operators
   RegisterReduceParsers(*this);
   // Others
   RegisterOperator("BatchNormalization", ParseBatchNormalization);
   RegisterOperator("Constant", ParseConstant);
   RegisterOperator("ConstantOfShape", ParseConstant);
   RegisterOperator("Cast", ParseCast);
   RegisterOperator("Concat", ParseConcat);
   RegisterOperator("Conv", ParseConv);
   RegisterOperator("ConvTranspose", ParseConvTranspose);
   RegisterOperator("Gemm", ParseGemm);
   RegisterOperator("GRU", ParseGRU);
   RegisterOperator("Identity", ParseIdentity);
   RegisterOperator("LeakyRelu", ParseLeakyRelu);
   RegisterOperator("LSTM", ParseLSTM);
   RegisterOperator("AveragePool", ParsePool);
   RegisterOperator("GlobalAveragePool", ParsePool);
   RegisterOperator("MaxPool", ParsePool);
   RegisterOperator("Relu", ParseRelu);
   RegisterOperator("Reshape", ParseReshape);
   RegisterOperator("Flatten", ParseReshape);
   RegisterOperator("Squeeze", ParseReshape);
   RegisterOperator("Unsqueeze", ParseReshape);
   RegisterOperator("RNN", ParseRNN);
   RegisterOperator("Gelu", ParseGelu);
   RegisterOperator("Selu", ParseSelu);
   RegisterOperator("Shape", ParseShape);
   RegisterOperator("Sigmoid", ParseSigmoid);
   RegisterOperator("Swish", ParseSwish);
   RegisterOperator("Slice", ParseSlice);
   RegisterOperator("Softmax", ParseSoftmax);
   RegisterOperator("LogSoftmax", ParseSoftmax);
   RegisterOperator("Tanh", ParseTanh);
   RegisterOperator("Transpose", ParseTranspose);
   RegisterOperator("MatMul", ParseMatMul);
   RegisterOperator("LayerNormalization", ParseLayerNormalization);
   RegisterOperator("Expand", ParseExpand);
   RegisterOperator("Gather", ParseGather);
   RegisterOperator("GatherND", ParseGatherND);
   RegisterOperator("Erf", ParseErf);
   RegisterOperator("Elu", ParseElu);
   RegisterOperator("HardSigmoid", ParseHardSigmoid);
   RegisterOperator("HardSwish", ParseHardSwish);
   RegisterOperator("EyeLike", ParseEyeLike);
   RegisterOperator("Range", ParseRange);
   RegisterOperator("TopK", ParseTopK);
   RegisterOperator("Tile", ParseTile);
   RegisterOperator("Split", ParseSplit);
   RegisterOperator("If", ParseIf);
   RegisterOperator("InstanceNormalization", ParseInstanceNormalization);
   RegisterOperator("Pad", ParsePad);
   RegisterOperator("Where", ParseWhere);
   RegisterOperator("Einsum", ParseEinsum);
   RegisterOperator("ScatterElements", ParseScatterElements);
   RegisterOperator("Trilu", ParseTrilu);
   RegisterOperator("NonZero", ParseNonZero);
   RegisterOperator("Clip", ParseClip);
   RegisterOperator("ScatterND", ParseScatterND);
   // Logical operators
   RegisterOperator("And", ParseAnd);
   RegisterOperator("Or", ParseOr);
   RegisterOperator("Xor", ParseXor);
   // Bitwise operators
   RegisterOperator("BitwiseAnd", ParseBitwiseAnd);
   RegisterOperator("BitwiseOr", ParseBitwiseOr);
   RegisterOperator("BitwiseXor", ParseBitwiseXor);
   RegisterOperator("BitwiseNot", ParseBitwiseNot);

   RegisterOperator("RMSNorm", ParseRMSNorm);
   RegisterOperator("GroupNormalization", ParseGroupNorm);
   RegisterOperator("CumSum", ParseCumSum);
   RegisterOperator("ScaledDotProductAttention", ParseSDPA);
   RegisterOperator("SDPA", ParseSDPA);
   RegisterOperator("MambaScan", ParseMambaScan);
   RegisterOperator("RWKV_WKV6", ParseRWKVWKV6);
   RegisterOperator("GriffinRGLRU", ParseGriffinRGLRU);
}

// Destructor of the parser
RModelParser_ONNX::~RModelParser_ONNX() = default;

void RModelParser_ONNX::RegisterOperator(const std::string &name, ParserFuncSignature func)
{
   fOperatorsMapImpl->fOperatorsMap[name] = func;
}

bool RModelParser_ONNX::IsRegisteredOperator(const std::string &name)
{
   return fOperatorsMapImpl->fOperatorsMap.find(name) != fOperatorsMapImpl->fOperatorsMap.end();
}

std::vector<std::string> RModelParser_ONNX::GetRegisteredOperators()
{
   std::vector<std::string> ops;
   ops.reserve(fOperatorsMapImpl->fOperatorsMap.size());
   for (auto &it : fOperatorsMapImpl->fOperatorsMap) {
      ops.emplace_back(it.first);
   }
   // return sorted list in alphabetical order
   std::sort(ops.begin(), ops.end());
   return ops;
}

void RModelParser_ONNX::RegisterTensorType(const std::string &name, ETensorType type)
{
   fTensorTypeMap[UTILITY::Clean_name(name)] = type;
}

bool RModelParser_ONNX::IsRegisteredTensorType(const std::string &name)
{
   return fTensorTypeMap.find(UTILITY::Clean_name(name)) != fTensorTypeMap.end();
}

void RModelParser_ONNX::RegisterFusedTransposeInput(const std::string &transposeOutput, const std::string &transposeInput)
{
   const std::string outputName = UTILITY::Clean_name(transposeOutput);
   const std::string inputName = UTILITY::Clean_name(transposeInput);
   const auto [it, inserted] = fFusedTransposeInputs.emplace(outputName, inputName);

   if (!inserted && it->second != inputName)
      throw std::runtime_error("SOFIE ONNX Parser found conflicting Transpose fusions for tensor " + outputName);
}

   RModelParser_ONNX::MatMulInputInfo
   RModelParser_ONNX::ConsumeFusedTransposeInput(const std::string &matmulInput)
{
   const std::string inputName = UTILITY::Clean_name(matmulInput);
   const auto transposeIt = fFusedTransposeInputs.find(inputName);

   if (transposeIt == fFusedTransposeInputs.end())
      return {inputName, 0};

   MatMulInputInfo inputInfo{transposeIt->second, 1};
   fFusedTransposeInputs.erase(transposeIt);
   return inputInfo;
}

ETensorType RModelParser_ONNX::GetTensorType(const std::string &name)
{
   return fTensorTypeMap[UTILITY::Clean_name(name)];
}

void RModelParser_ONNX::RegisterTensorAlias(const std::string &aliasOutput, const std::string &aliasInput)
{
   const std::string outputName = UTILITY::Clean_name(aliasOutput);
   const std::string inputName = UTILITY::Clean_name(ResolveTensorAlias(aliasInput));

   if (outputName.empty() || inputName.empty())
      throw std::runtime_error("SOFIE cannot register an empty tensor alias");

   if (outputName == inputName)
      return;

   const auto [it, inserted] = fTensorAliases.emplace(outputName, inputName);

   if (!inserted && it->second != inputName) {
      throw std::runtime_error("SOFIE found conflicting aliases for tensor " + outputName);
   }
}

   std::string RModelParser_ONNX::ResolveTensorAlias(const std::string &tensorName) const
{
   const std::string originalName = tensorName;
   std::string currentName = UTILITY::Clean_name(tensorName);
   std::unordered_set<std::string> visited;
   bool resolvedAlias = false;

   while (true) {
      if (!visited.insert(currentName).second)
         throw std::runtime_error("SOFIE found a cycle in tensor aliases");

      const auto aliasIt = fTensorAliases.find(currentName);

      if (aliasIt == fTensorAliases.end())
         return resolvedAlias ? currentName : originalName;

      currentName = aliasIt->second;
      resolvedAlias = true;
   }
}

namespace {

bool IsGraphOutputForSoftmaxRewrite(const onnx::GraphProto &graph, const std::string &tensorName)
{
   for (int i = 0; i < graph.output_size(); ++i) {
      if (graph.output(i).name() == tensorName)
         return true;
   }

   return false;
}

int FindUniqueConsumerForSoftmaxRewrite(const onnx::GraphProto &graph, const std::string &tensorName)
{
   int consumerIdx = -1;

   for (int i = 0; i < graph.node_size(); ++i) {
      const auto &node = graph.node(i);
      bool consumesTensor = false;

      for (int j = 0; j < node.input_size(); ++j) {
         if (node.input(j) == tensorName) {
            consumesTensor = true;
            break;
         }
      }

      if (!consumesTensor)
         continue;

      if (consumerIdx != -1)
         return -1;

      consumerIdx = i;
   }

   return consumerIdx;
}

bool ReadSingleInt64TensorForSoftmaxRewrite(const onnx::TensorProto &tensor, int64_t &value)
{
   if (tensor.data_type() != onnx::TensorProto::INT64)
      return false;

   size_t length = 1;

   for (int i = 0; i < tensor.dims_size(); ++i)
      length *= static_cast<size_t>(tensor.dims(i));

   if (length != 1)
      return false;

   if (tensor.int64_data_size() == 1) {
      value = tensor.int64_data(0);
      return true;
   }

   if (tensor.raw_data().size() == sizeof(int64_t)) {
      std::memcpy(&value, tensor.raw_data().data(), sizeof(int64_t));

      if constexpr (std::endian::native != std::endian::little)
         value = bswap_value(value);

      return true;
   }

   return false;
}

bool TryGetSingleInt64ConstantForSoftmaxRewrite(const onnx::GraphProto &graph,
                                                const std::string &tensorName,
                                                int64_t &value)
{
   for (int i = 0; i < graph.initializer_size(); ++i) {
      const auto &initializer = graph.initializer(i);

      if (initializer.name() == tensorName)
         return ReadSingleInt64TensorForSoftmaxRewrite(initializer, value);
   }

   for (int i = 0; i < graph.node_size(); ++i) {
      const auto &node = graph.node(i);

      if (node.op_type() != "Constant" || node.output_size() != 1 || node.output(0) != tensorName)
         continue;

      for (int j = 0; j < node.attribute_size(); ++j) {
         const auto &attribute = node.attribute(j);

         if (attribute.name() == "value_int") {
            value = attribute.i();
            return true;
         }

         if (attribute.name() == "value_ints" && attribute.ints_size() == 1) {
            value = attribute.ints(0);
            return true;
         }

         if (attribute.name() == "value" && attribute.has_t())
            return ReadSingleInt64TensorForSoftmaxRewrite(attribute.t(), value);
      }

      return false;
   }

   return false;
}

bool IsLastAxisReduceMaxForSoftmaxRewrite(const onnx::GraphProto &graph,
                                         const onnx::NodeProto &node)
{
   if (node.op_type() != "ReduceMax")
      return false;

   int64_t keepdims = 1;
   int64_t axis = 0;
   bool hasAxisAttribute = false;

   for (int i = 0; i < node.attribute_size(); ++i) {
      const auto &attribute = node.attribute(i);

      if (attribute.name() == "keepdims") {
         keepdims = attribute.i();
      } else if (attribute.name() == "axes") {
         if (attribute.ints_size() != 1)
            return false;

         axis = attribute.ints(0);
         hasAxisAttribute = true;
      }
   }

   if (keepdims != 1)
      return false;

   const bool hasAxisInput = node.input_size() > 1 && !node.input(1).empty();

   if (hasAxisAttribute && hasAxisInput)
      return false;

   if (hasAxisInput) {
      if (!TryGetSingleInt64ConstantForSoftmaxRewrite(graph, node.input(1), axis))
         return false;
   } else if (!hasAxisAttribute) {
      return false;
   }

   return axis == -1;
}

bool IsLastAxisSoftmaxForRewrite(const onnx::NodeProto &node)
{
   if (node.op_type() != "Softmax")
      return false;

   int64_t axis = -1;

   for (int i = 0; i < node.attribute_size(); ++i) {
      if (node.attribute(i).name() == "axis") {
         axis = node.attribute(i).i();
         break;
      }
   }

   return axis == -1;
}

bool TryMatchRedundantSoftmaxStabilization(const onnx::GraphProto &graph,
                                           int reduceMaxIdx,
                                           int &subIdx,
                                           int &softmaxIdx)
{
   if (reduceMaxIdx < 0 || reduceMaxIdx >= graph.node_size())
      return false;

   const auto &reduceMaxNode = graph.node(reduceMaxIdx);

   if (reduceMaxNode.input_size() < 1 || reduceMaxNode.output_size() != 1)
      return false;

   if (!IsLastAxisReduceMaxForSoftmaxRewrite(graph, reduceMaxNode))
      return false;

   const std::string &inputName = reduceMaxNode.input(0);
   const std::string &reduceMaxOutput = reduceMaxNode.output(0);

   if (IsGraphOutputForSoftmaxRewrite(graph, reduceMaxOutput))
      return false;

   subIdx = FindUniqueConsumerForSoftmaxRewrite(graph, reduceMaxOutput);

   if (subIdx < 0)
      return false;

   const auto &subNode = graph.node(subIdx);

   if (subNode.op_type() != "Sub" || subNode.input_size() != 2 || subNode.output_size() != 1)
      return false;

   if (subNode.input(0) != inputName || subNode.input(1) != reduceMaxOutput)
      return false;

   const std::string &subOutput = subNode.output(0);

   if (IsGraphOutputForSoftmaxRewrite(graph, subOutput))
      return false;

   softmaxIdx = FindUniqueConsumerForSoftmaxRewrite(graph, subOutput);

   if (softmaxIdx < 0)
      return false;

   const auto &softmaxNode = graph.node(softmaxIdx);

   if (softmaxNode.input_size() != 1 || softmaxNode.output_size() != 1)
      return false;

   if (softmaxNode.input(0) != subOutput)
      return false;

   return IsLastAxisSoftmaxForRewrite(softmaxNode);
}

} // namespace

namespace {

bool IsGraphOutput(const onnx::GraphProto &graph, const std::string &tensorName)
{
   for (const auto &output : graph.output()) {
      if (output.name() == tensorName)
         return true;
   }

   return false;
}

bool TryGetTensorRank(const onnx::GraphProto &graph, const std::string &tensorName, size_t &rank)
{
   for (const auto &initializer : graph.initializer()) {
      if (initializer.name() == tensorName) {
         rank = initializer.dims_size();
         return true;
      }
   }

   auto tryValueInfo = [&](const onnx::ValueInfoProto &valueInfo) {
      if (valueInfo.name() != tensorName)
         return false;

      if (!valueInfo.has_type() || !valueInfo.type().has_tensor_type())
         return false;

      const auto &tensorType = valueInfo.type().tensor_type();

      if (!tensorType.has_shape())
         return false;

      rank = tensorType.shape().dim_size();
      return true;
   };

   for (const auto &input : graph.input()) {
      if (tryValueInfo(input))
         return true;
   }

   for (const auto &valueInfo : graph.value_info()) {
      if (tryValueInfo(valueInfo))
         return true;
   }

   for (const auto &output : graph.output()) {
      if (tryValueInfo(output))
         return true;
   }

   return false;
}

bool IsLastTwoAxesTranspose(const onnx::NodeProto &node, const onnx::GraphProto &graph)
{
   std::vector<int64_t> permutation;
   bool hasPermutation = false;

   for (const auto &attribute : node.attribute()) {
      if (attribute.name() == "perm") {
         permutation.assign(attribute.ints().begin(), attribute.ints().end());
         hasPermutation = true;
         break;
      }
   }

   if (!hasPermutation) {
      size_t rank = 0;

      if (!TryGetTensorRank(graph, node.input(0), rank))
         return false;

      // The default ONNX permutation reverses every axis. That is equivalent
      // to a matrix transpose only for rank-two tensors.
      return rank == 2;
   }

   if (permutation.size() < 2)
      return false;

   const size_t rank = permutation.size();

   for (size_t axis = 0; axis + 2 < rank; ++axis) {
      if (permutation[axis] != static_cast<int64_t>(axis))
         return false;
   }

   return permutation[rank - 2] == static_cast<int64_t>(rank - 1) &&
          permutation[rank - 1] == static_cast<int64_t>(rank - 2);
}

int FindSingleConsumerNode(const onnx::GraphProto &graph, const std::string &tensorName)
{
   int consumerIdx = -1;

   for (int nodeIdx = 0; nodeIdx < graph.node_size(); ++nodeIdx) {
      bool consumesTensor = false;

      for (const auto &inputName : graph.node(nodeIdx).input()) {
         if (inputName == tensorName) {
            consumesTensor = true;
            break;
         }
      }

      if (!consumesTensor)
         continue;

      if (consumerIdx != -1)
         return -1;

      consumerIdx = nodeIdx;
   }

   return consumerIdx;
}

int FindProducerNode(const onnx::GraphProto &graph, const std::string &tensorName)
{
   for (int nodeIdx = 0; nodeIdx < graph.node_size(); ++nodeIdx) {
      for (const auto &outputName : graph.node(nodeIdx).output()) {
         if (outputName == tensorName)
            return nodeIdx;
      }
   }

   return -1;
}

bool TryGetConstantFloat(const onnx::GraphProto &graph, const std::string &tensorName, float &value)
{
   for (const auto &initializer : graph.initializer()) {
      if (initializer.name() != tensorName)
         continue;

      if (initializer.data_type() != onnx::TensorProto::FLOAT)
         return false;

      size_t length = 1;
      for (const auto dim : initializer.dims())
         length *= static_cast<size_t>(dim);

      if (length != 1)
         return false;

      if (initializer.float_data_size() == 1) {
         value = initializer.float_data(0);
         return true;
      }

      if (initializer.raw_data().size() == sizeof(float)) {
         std::memcpy(&value, initializer.raw_data().data(), sizeof(float));
         return true;
      }

      return false;
   }

   const int producerIdx = FindProducerNode(graph, tensorName);

   if (producerIdx < 0)
      return false;

   const auto &constantNode = graph.node(producerIdx);

   if (constantNode.op_type() != "Constant" || constantNode.attribute_size() != 1)
      return false;

   const auto &attribute = constantNode.attribute(0);

   if (attribute.name() == "value_float") {
      value = attribute.f();
      return true;
   }

   if (attribute.name() != "value" || !attribute.has_t())
      return false;

   const auto &tensor = attribute.t();

   if (tensor.data_type() != onnx::TensorProto::FLOAT)
      return false;

   size_t length = 1;
   for (const auto dim : tensor.dims())
      length *= static_cast<size_t>(dim);

   if (length != 1)
      return false;

   if (tensor.float_data_size() == 1) {
      value = tensor.float_data(0);
      return true;
   }

   if (tensor.raw_data().size() == sizeof(float)) {
      std::memcpy(&value, tensor.raw_data().data(), sizeof(float));
      return true;
   }

   return false;
}

bool IsConstantTensor(const onnx::GraphProto &graph, const std::string &tensorName)
{
   for (const auto &initializer : graph.initializer()) {
      if (initializer.name() == tensorName)
         return true;
   }

   const int producerIdx = FindProducerNode(graph, tensorName);
   return producerIdx >= 0 && graph.node(producerIdx).op_type() == "Constant";
}

bool HasLastAxisReduce(const onnx::NodeProto &reduceNode)
{
   int64_t keepdims = 1;
   std::vector<int64_t> axes;

   for (const auto &attribute : reduceNode.attribute()) {
      if (attribute.name() == "keepdims")
         keepdims = attribute.i();
      else if (attribute.name() == "axes")
         axes.assign(attribute.ints().begin(), attribute.ints().end());
   }

   return keepdims == 1 && axes.size() == 1 && axes[0] == -1;
}

bool TryMatchL2Normalization(const onnx::GraphProto &graph, int reduceIdx, int &clipIdx, int &expandIdx, int &divIdx, float &epsilon)
{
   const auto &reduceNode = graph.node(reduceIdx);

   if (reduceNode.op_type() != "ReduceL2" || reduceNode.input_size() < 1 ||
       reduceNode.output_size() != 1 || !HasLastAxisReduce(reduceNode))
      return false;

   const std::string inputName = reduceNode.input(0);
   const std::string reduceOutput = reduceNode.output(0);

   clipIdx = FindSingleConsumerNode(graph, reduceOutput);

   if (clipIdx < 0)
      return false;

   const auto &clipNode = graph.node(clipIdx);

   if (clipNode.op_type() != "Clip" || clipNode.input_size() < 2 ||
       clipNode.output_size() != 1 || clipNode.input(0) != reduceOutput ||
       clipNode.input(1).empty() || !TryGetConstantFloat(graph, clipNode.input(1), epsilon))
      return false;

   if (clipNode.input_size() > 2 && !clipNode.input(2).empty())
      return false;

   const std::string clipOutput = clipNode.output(0);
   expandIdx = FindSingleConsumerNode(graph, clipOutput);

   if (expandIdx < 0)
      return false;

   const auto &expandNode = graph.node(expandIdx);

   if (expandNode.op_type() != "Expand" || expandNode.input_size() != 2 ||
       expandNode.output_size() != 1 || expandNode.input(0) != clipOutput)
      return false;

   const int shapeIdx = FindProducerNode(graph, expandNode.input(1));

   if (shapeIdx < 0)
      return false;

   const auto &shapeNode = graph.node(shapeIdx);

   if (shapeNode.op_type() != "Shape" || shapeNode.input_size() != 1 ||
       shapeNode.input(0) != inputName)
      return false;

   const std::string expandOutput = expandNode.output(0);
   divIdx = FindSingleConsumerNode(graph, expandOutput);

   if (divIdx < 0)
      return false;

   const auto &divNode = graph.node(divIdx);

   if (divNode.op_type() != "Div" || divNode.input_size() != 2 ||
       divNode.output_size() != 1 || divNode.input(0) != inputName ||
       divNode.input(1) != expandOutput)
      return false;

   if (IsGraphOutput(graph, reduceOutput) || IsGraphOutput(graph, clipOutput) ||
       IsGraphOutput(graph, expandOutput))
      return false;

   return true;
}

bool TryGetCastTargetType(const onnx::NodeProto &castNode, int64_t &targetType)
{
   for (const auto &attribute : castNode.attribute()) {
      if (attribute.name() == "to") {
         targetType = attribute.i();
         return true;
      }
   }

   return false;
}

bool GraphReferencesTensor(const onnx::GraphProto &graph, const std::string &tensorName)
{
   for (const auto &node : graph.node()) {
      for (const auto &inputName : node.input()) {
         if (inputName == tensorName)
            return true;
      }

      for (const auto &attribute : node.attribute()) {
         if (attribute.has_g() && GraphReferencesTensor(attribute.g(), tensorName))
            return true;

         for (int k = 0; k < attribute.graphs_size(); ++k) {
            if (GraphReferencesTensor(attribute.graphs(k), tensorName))
               return true;
         }
      }
   }

   return false;
}

bool IsReferencedByNestedGraph(const onnx::GraphProto &graph, const std::string &tensorName)
{
   for (const auto &node : graph.node()) {
      for (const auto &attribute : node.attribute()) {
         if (attribute.has_g() && GraphReferencesTensor(attribute.g(), tensorName))
            return true;

         for (int k = 0; k < attribute.graphs_size(); ++k) {
            if (GraphReferencesTensor(attribute.graphs(k), tensorName))
               return true;
         }
      }
   }

   return false;
}

onnx::NodeProto ResolveNodeInputs(const RModelParser_ONNX &parser, const onnx::NodeProto &node)
{
   onnx::NodeProto resolvedNode = node;

   for (int inputIdx = 0; inputIdx < resolvedNode.input_size(); ++inputIdx) {
      if (!resolvedNode.input(inputIdx).empty())
         resolvedNode.set_input(inputIdx, parser.ResolveTensorAlias(resolvedNode.input(inputIdx)));
   }

   return resolvedNode;
}

} // anonymous namespace

namespace {

bool IsConvBiasAdd(const onnx::GraphProto &graph, const onnx::NodeProto &convnode, const onnx::NodeProto &addnode)
{
   if (convnode.input_size() > 2 || addnode.input_size() != 2)
      return false;
   const std::string &added = (addnode.input(0) == convnode.output(0)) ? addnode.input(1) : addnode.input(0);
   for (int i = 0; i < graph.initializer_size(); i++) {
      if (graph.initializer(i).name() == added)
         return graph.initializer(i).dims_size() == 1;
   }
   return false;
}

} // namespace

// Parse an operator
std::unique_ptr<ROperator>
RModelParser_ONNX::ParseOperator(const size_t i, const onnx::GraphProto &graphproto, const std::vector<size_t> &nodes, const std::vector<int> & children)
{
   if (i >= nodes.size())
      throw std::runtime_error("SOFIE - Error in parsing ordered operators " + std::to_string(i) + " is >=  " + std::to_string(nodes.size()));
   int idx = nodes[i];
   const auto &graphNode = graphproto.node(idx);
   onnx::NodeProto nodeproto = ResolveNodeInputs(*this, graphNode);
   const std::string op_type = nodeproto.op_type();
   if (fVerbose)
      std::cout << "Parsing operator " << op_type << std::endl;

   if (fFusedOperators.count(idx) == 1) {
      const auto fusion = fFusedOperators[idx];
      if (fusion.first == EFusedOp::kSkipped)
         return nullptr;
      const int idx1 = fusion.second;
      const onnx::NodeProto firstNode = ResolveNodeInputs(*this, graphproto.node(idx1));
      if (fVerbose) {
         std::cout << "\tFusing operators " << graphproto.node(idx1).name()
                   << " with  " <<  graphproto.node(idx).name() << std::endl;
      }
      if (fusion.first == EFusedOp::kMatMulAdd) {
         return ParseFuseMatMulAdd(*this, firstNode, nodeproto);
      } else if (fusion.first == EFusedOp::kConvAdd) {
         return ParseFuseConvAdd(*this, firstNode, nodeproto);
      } else if (fusion.first == EFusedOp::kConvTransAdd) {
         return ParseFuseConvTransposeAdd(*this, firstNode, nodeproto);
      } else if (fusion.first == EFusedOp::kGemmRelu) {
         return ParseFuseGemmRelu(*this, firstNode, nodeproto);
      } else if (fusion.first == EFusedOp::kBatchnormRelu) {
         return ParseFuseBatchnormRelu(*this, firstNode, nodeproto);
      }
   }

   // Eliminate operators proven to preserve both values and element type.
   if (op_type == "Cast" && nodeproto.input_size() == 1 && nodeproto.output_size() == 1) {
      const std::string sourceInput = nodeproto.input(0);
      const std::string aliasOutput = graphNode.output(0);
      const bool outputCanBeAliased = !sourceInput.empty() && !aliasOutput.empty() &&
         !IsGraphOutput(graphproto, aliasOutput)
         && !IsReferencedByNestedGraph(graphproto, aliasOutput) && IsRegisteredTensorType(sourceInput);

      if (outputCanBeAliased) {
         const ETensorType sourceType = GetTensorType(sourceInput);
         int64_t targetType = -1;
         const bool isTransparent =
            TryGetCastTargetType(graphNode, targetType) && targetType == static_cast<int64_t>(sourceType);

         if (isTransparent) {
            RegisterTensorAlias(aliasOutput, sourceInput);
            RegisterTensorType(aliasOutput, sourceType);

            if (fVerbose)
               std::cout << "\tAliased transparent " << op_type << " output "
               << aliasOutput << " to " << sourceInput << std::endl;

            return nullptr;
         }
      }
   }

   // Softmax already performs its own numerically stable maximum subtraction.
   // Remove an explicit ReduceMax(X, -1) -> Sub(X, max) stabilization prefix.
   if (op_type == "ReduceMax") {
      int subIdx = -1;
      int softmaxIdx = -1;

      if (TryMatchRedundantSoftmaxStabilization(graphproto, idx, subIdx, softmaxIdx)) {
         onnx::NodeProto rewrittenSoftmax = ResolveNodeInputs(*this, graphproto.node(softmaxIdx));
         rewrittenSoftmax.set_input(0, nodeproto.input(0));

         auto op = ParseSoftmax(*this, rewrittenSoftmax);

         fFusedOperators[subIdx] = {EFusedOp::kSkipped, idx};
         fFusedOperators[softmaxIdx] = {EFusedOp::kSkipped, idx};

         if (fVerbose) {
            std::cout << "\tRemoved redundant ReduceMax -> Sub stabilization before Softmax" << std::endl;
         }

         return op;
      }
   }

   if (op_type == "ReduceL2") {
      int clipIdx = -1;
      int expandIdx = -1;
      int divIdx = -1;
      float epsilon = 0.0f;

      if (TryMatchL2Normalization(graphproto, idx, clipIdx, expandIdx, divIdx, epsilon)) {
         auto op = ParseFuseL2Normalization(*this, nodeproto, graphproto.node(divIdx), epsilon);

         fFusedOperators[clipIdx] = {EFusedOp::kSkipped, idx};
         fFusedOperators[expandIdx] = {EFusedOp::kSkipped, idx};
         fFusedOperators[divIdx] = {EFusedOp::kSkipped, idx};

         return op;
      }
   }

   if (children.size() == 1) {
      const int idx2 = children.front();
      const onnx::NodeProto childNode = ResolveNodeInputs(*this, graphproto.node(idx2));

      if (op_type == "Transpose") {
         const bool validNodeShape = nodeproto.input_size() == 1 && nodeproto.output_size() == 1;
         const bool childIsMatMul = childNode.op_type() == "MatMul";
         const bool validMatMulInputs = childIsMatMul && childNode.input_size() == 2;
         const bool outputIsVisible = validNodeShape && IsGraphOutput(graphproto, nodeproto.output(0));
         const bool validPermutation = validNodeShape && IsLastTwoAxesTranspose(nodeproto, graphproto);
         const bool supportedType = validNodeShape && IsRegisteredTensorType(nodeproto.input(0)) && GetTensorType(nodeproto.input(0)) == ETensorType::FLOAT;

         if (validMatMulInputs && !outputIsVisible && validPermutation && supportedType) {
            RegisterFusedTransposeInput(nodeproto.output(0), nodeproto.input(0));
            return nullptr;
         }
      } else if (op_type == "MatMul") {
         if (childNode.op_type() == "Add" && childNode.input_size() == 2) {
            fFusedOperators[idx2] = {EFusedOp::kMatMulAdd, idx};
            return nullptr;
         }
      } else if (op_type == "Conv" || op_type == "ConvTranspose") {
         if (childNode.op_type() == "Add" && IsConvBiasAdd(graphproto, nodeproto, childNode)) {
            fFusedOperators[idx2] = {op_type == "Conv" ? EFusedOp::kConvAdd : EFusedOp::kConvTransAdd, idx};
            return nullptr;
         }
      } else if (op_type == "Gemm") {
         if (childNode.op_type() == "Relu") {
            fFusedOperators[idx2] = {EFusedOp::kGemmRelu, idx};
            return nullptr;
         }
      } else if (op_type == "BatchNormalization") {
         if (childNode.op_type() == "Relu") {
            fFusedOperators[idx2] = {EFusedOp::kBatchnormRelu, idx};
            return nullptr;
         }
      }
   }

   auto it = fOperatorsMapImpl->fOperatorsMap.find(op_type);
   if (it == fOperatorsMapImpl->fOperatorsMap.end()) {
      std::cout << "operator " << op_type << " is not supported" << std::endl;
      throw std::runtime_error("SOFIE Operator type " + op_type + " is not yet supported");
   }
   if (fVerbose) {
      std::cout << "\tCreating operator " << op_type << std::endl;
   }
   return it->second(*this, nodeproto);
}

// Parse a model
RModel RModelParser_ONNX::Parse(std::string const &filename, bool verbose)
{
   fVerbose = verbose;

   fTensorTypeMap.clear();
   fFusedTransposeInputs.clear();
   fTensorAliases.clear();

   auto model = LoadModel(filename);
   if (!model)
      throw std::runtime_error("SOFIE - Failed to load onnx file " + filename);

   const onnx::GraphProto &graph = model->graph(); // not a memory leak. model freed automatically at the end.


   std::time_t ttime = std::time(0);
   std::tm *gmt_time = std::gmtime(&ttime);
   std::string parsetime(std::asctime(gmt_time));

   // get name of model (filename without directory name)
   char sep = '/';
#ifdef _WIN32
   sep = '\\';
#endif
   size_t isep = filename.rfind(sep, filename.length());
   std::string filename_nodir = filename;
   if (isep != std::string::npos) {
      filename_nodir = (filename.substr(isep + 1, filename.length() - isep));
   }

   fModelDirectory = (isep != std::string::npos) ? filename.substr(0, isep + 1) : "";
   fDefaultDataFileName = filename + ".data";

   RModel rmodel(filename_nodir, parsetime);
   ParseONNXGraph(rmodel, graph, filename_nodir);
   ResetExternalDataState();
   return rmodel;
}

RModel RModelParser_ONNX::Parse(std::istream &input, std::string const &name, bool verbose)
{
   fVerbose = verbose;

   fTensorTypeMap.clear();

   auto model = LoadModel(input);
   if (!model)
      throw std::runtime_error("SOFIE - Failed to parse ONNX model from input stream");

   const onnx::GraphProto &graph = model->graph(); // not a memory leak. model freed automatically at the end.

   std::time_t ttime = std::time(0);
   std::tm *gmt_time = std::gmtime(&ttime);
   std::string parsetime(std::asctime(gmt_time));

   RModel rmodel(name, parsetime);
   ParseONNXGraph(rmodel, graph, name);
   ResetExternalDataState();
   return rmodel;
}

void RModelParser_ONNX::ResetExternalDataState()
{
   fDataFileName.clear();
   fModelDirectory.clear();
   fDefaultDataFileName.clear();
   fOpenedDataFileName.clear();
   if (fDataFile.is_open())
      fDataFile.close();
}

std::unique_ptr<onnx::ModelProto> RModelParser_ONNX::LoadModel(const std::string &filename) {
   std::fstream input(filename, std::ios::in | std::ios::binary);
   if (!input) {
      std::cerr << "SOFIE - Failed to open onnx file " << filename << std::endl;
      return {};
   }

   return LoadModel(input);
}

std::unique_ptr<onnx::ModelProto> RModelParser_ONNX::LoadModel(std::istream &input)
{
   auto model = std::make_unique<onnx::ModelProto>();

   if (!model->ParseFromIstream(&input)) {
      std::cerr << "SOFIE - Failed to parse ONNX model from input stream" << std::endl;
      return {};
   }

   // ONNX version is ir_version()  - model_version() returns 0
   if (fVerbose) {
      std::cout << "ONNX Version " << model->ir_version() << std::endl;
   }
   return model;
}

void RModelParser_ONNX::CheckGraph(const onnx::GraphProto & graph, int & level, std::map<std::string, int> & missingOperators) {
   if (fVerbose)
      std::cout << "\n" << graph.name() << " Graph operator list\n";
   for (int i = 0; i < graph.node_size(); i++) {
      const auto & node = graph.node(i);
      const std::string opType =  node.op_type();
      if (fVerbose) {
         std::cout << "\tOperator " << i << " : " << opType << " (" << node.name() << "), " << graph.node(i).input_size()
                      << " inputs : {";
            for (int j = 0; j < graph.node(i).input_size(); j++) {
               std::cout << graph.node(i).input(j);
               if (j < graph.node(i).input_size() - 1)
                  std::cout << ", ";
            }
         std::cout << " }" << std::endl;
      }
      // check if operator exists
      if (!IsRegisteredOperator(opType))
         missingOperators[opType] = level;
      // see if sub-graph exists as node attributes
      for (int j = 0; j < node.attribute_size(); j++) {
         const auto & attribute = node.attribute(j);
         if (attribute.has_g()) {
            const auto & subGraph = attribute.g();
            level += 1;
            CheckGraph(subGraph, level, missingOperators);
         }
      }
   }
}

bool RModelParser_ONNX::CheckModel(std::string filename, bool verbose) {

   fVerbose = verbose;
   auto model = LoadModel(filename);
   if (!model) return false;

   const onnx::GraphProto &graph = model->graph();
    // Initial operator order
   if (fVerbose)
      std::cout << "\nModel operator list " << model->producer_name() << "\n";

   std::map<std::string, int> missingOperators;
   int level = 1;
   CheckGraph(graph, level, missingOperators);

   if (!missingOperators.empty()) {
      std::cout << "List of missing operators for model loaded from file " << filename << std::endl;
      for (auto & op : missingOperators) {
         std::cout << op.first << "  " << op.second << std::endl;
      }
      return false;
   }
   std::cout << "All operators in the loaded model are supported!\n";
   return true;
}

void RModelParser_ONNX::ParseONNXGraph(RModel & rmodel, const onnx::GraphProto & graph, std::string  graphName)
{
   bool verbose = fVerbose;

   if (graphName.empty())
      graphName = graph.name();

   if (verbose)
      std::cout << "\nParsing Graph - " << graphName << std::endl;

   struct FusedOperatorsGuard {
      std::map<int, std::pair<EFusedOp, int>> &fMap;
      std::unordered_map<std::string, std::string> &fTransposes;
      std::unordered_map<std::string, std::string> &fAliases;
      std::map<int, std::pair<EFusedOp, int>> fSaved;
      std::unordered_map<std::string, std::string> fSavedTransposes;
      std::unordered_map<std::string, std::string> fSavedAliases;
      FusedOperatorsGuard(std::map<int, std::pair<EFusedOp, int>> &map,
                          std::unordered_map<std::string, std::string> &transposes,
                          std::unordered_map<std::string, std::string> &aliases)
         : fMap(map), fTransposes(transposes), fAliases(aliases)
      {
         fSaved.swap(fMap);
         fSavedTransposes.swap(fTransposes);
         fSavedAliases.swap(fAliases);
      }
      ~FusedOperatorsGuard()
      {
         fMap.swap(fSaved);
         fTransposes.swap(fSavedTransposes);
         fAliases.swap(fSavedAliases);
      }
   } fusedOperatorsGuard{fFusedOperators, fFusedTransposeInputs, fTensorAliases};

   std::unordered_set<std::string> initializer_names;
   for (int i = 0; i < graph.initializer_size(); i++) {
      initializer_names.insert(graph.initializer(i).name());
   }

   if (verbose)
      std::cout << "Parsing model inputs...." << std::endl;
   /// Loop on model inputs
   for (int i = 0; i < graph.input_size(); i++) {
      RegisterTensorType(graph.input(i).name(),
                         static_cast<ETensorType>(graph.input(i).type().tensor_type().elem_type()));

      if (verbose)
         std::cout << "\tgraph input " << i << " name " << graph.input(i).name() << " type "
                   << graph.input(i).type().tensor_type().elem_type() << std::endl;

      if (initializer_names.find(graph.input(i).name()) != initializer_names.end())
         continue;

      // input data node is not a weight node (has no initializer)
      const onnx::ValueInfoProto &valueinfoproto = graph.input(i);
      std::string input_name = valueinfoproto.name();

      ETensorType type = static_cast<ETensorType>(valueinfoproto.type().tensor_type().elem_type());

      std::vector<Dim> fShape;
      bool existParam = false;
      if (!valueinfoproto.type().tensor_type().has_shape())
         throw std::runtime_error("SOFIE data node with no shape restrictions is not supported yet");
      for (int j = 0; j < valueinfoproto.type().tensor_type().shape().dim_size(); j++) {
         Dim dim;
         if (valueinfoproto.type().tensor_type().shape().dim(j).value_case() ==
             onnx::TensorShapeProto_Dimension::ValueCase::kDimValue) {
             int dim_value = valueinfoproto.type().tensor_type().shape().dim(j).dim_value();
             dim.dim = dim_value;
             // case input dim is -1 - set a parametric shape
             if (dim_value < 0) {
               dim.isParam = true;
               existParam = true;
               dim.param = UTILITY::Clean_name(input_name) + "_size";
             }
         } else if (valueinfoproto.type().tensor_type().shape().dim(j).value_case() ==
                    onnx::TensorShapeProto_Dimension::ValueCase::kDimParam) {
            dim.isParam = true;
            existParam = true;
            dim.param = valueinfoproto.type().tensor_type().shape().dim(j).dim_param();
         } else {
            throw std::runtime_error("SOFIE ONNX file error: Valueinfoproto " + input_name +
                                     " has neither dim_value nor dim_param! \n");
         }
         fShape.push_back(dim);
      }
      if (valueinfoproto.type().tensor_type().shape().dim_size() == 0) {
         Dim dim;
         dim.dim = 1;
         fShape.push_back(dim);
      } // in case this TensorShapeProto has no dimension message: ONNX IR defines this to be a scalar

      if (!existParam) {
         std::vector<size_t> fShape_sizet;
         for (auto &j : fShape) {
            fShape_sizet.push_back(j.dim);
         }

         rmodel.AddInputTensorInfo(input_name, type, fShape_sizet);
      } else {
         rmodel.AddInputTensorInfo(input_name, type, fShape);
      }
      rmodel.AddInputTensorName(input_name); // store also names in given order
   }

   std::map<std::string, int> allInitializedTensors;

   if (verbose)
      std::cout << "\nParsing graph initializer list and fill model initialized tensors" << std::endl;

   for (int i = 0; i < graph.initializer_size(); i++) {
      onnx::TensorProto *tensorproto = const_cast<onnx::TensorProto *>(&graph.initializer(i));
      std::vector<std::size_t> shape;
      std::size_t tensor_length = 1;
      for (int j = 0; j < tensorproto->dims_size(); j++) {
         shape.push_back(tensorproto->dims(j));
         tensor_length *= tensorproto->dims(j);
      }
      // in case of scalars keep an empty shape but with length =1

      std::string tensor_name = graph.initializer(i).name();

      if (verbose)
         std::cout << "\t initializer " << i << " name " << tensor_name << " type " << graph.initializer(i).data_type()
                   << " and length " << tensor_length << std::endl;


      // register also the initialized tensors
      auto tensor_type = static_cast<ETensorType>(graph.initializer(i).data_type());
      RegisterTensorType(tensor_name, tensor_type);

      std::shared_ptr<void> data = GetInitializedTensorData(tensorproto, tensor_length * GetTypeSize(tensor_type), tensor_type);
      rmodel.AddInitializedTensor(tensor_name, tensor_type, shape, data);
      allInitializedTensors[tensor_name] = i;

      if (verbose) {
         std::cout << "add initialized tensor " << tensor_name << "with shape " << ConvertShapeToString(shape) << "and  ";
         if (tensor_type == ETensorType::FLOAT) {
            std::cout << " float data: ";
            for (int j = 0; j < std::min(int(tensor_length),3); j++) std::cout << static_cast<float*>(data.get())[j] << "  ";
         }
         else if (tensor_type == ETensorType::INT64) {
            std::cout << " int64 data: ";
            for (int j = 0; j < std::min(int(tensor_length),3); j++) std::cout << static_cast<int64_t*>(data.get())[j] << "  ";
         }
         else if (tensor_type == ETensorType::UINT8) {
            std::cout << " uint8 data: ";
            for (int j = 0; j < std::min(int(tensor_length),3); j++) std::cout << static_cast<uint8_t*>(data.get())[j] << "  ";
         }
         else if (tensor_type == ETensorType::BOOL) {
            std::cout << " Boolean data: ";
            for (int j = 0; j < std::min(int(tensor_length),3); j++) std::cout << static_cast<bool*>(data.get())[j] << "  ";
         }
         std::cout << std::endl;
      }
   }

   // Initial operator order
   if (verbose) {
      std::cout << "\nGraph operator list (ONNX order)\n";
      for (int i = 0; i < graph.node_size(); i++) {
         std::cout << "\tOperator " << i << " : " << graph.node(i).op_type() << " , " << graph.node(i).input_size()
                   << " inputs : {";
         for (int j = 0; j < graph.node(i).input_size(); j++) {
            std::cout << graph.node(i).input(j);
            if (j < graph.node(i).input_size() - 1)
               std::cout << ", ";
         }
         std::cout << " }" << std::endl;
      }
   }

   // make order of nodes:
   if (verbose)
      std::cout << "\n***********************\nRe-Order graph operator list\n*************************\n";
   std::vector<size_t> nodesOrder;
   nodesOrder.reserve(graph.node_size());
   std::vector<bool> foundNodes(graph.node_size());

   // Pre-compute the set of all tensor names that belong to THIS graph:
   // graph inputs, initializers, and node outputs.  A tensor is an "outer-scope
   // reference" (from an enclosing graph) only if it is NOT in this set.
   std::unordered_set<std::string> graphLocalTensors;
   for (int i = 0; i < graph.input_size(); i++)
      graphLocalTensors.insert(graph.input(i).name());
   for (int i = 0; i < graph.initializer_size(); i++)
      graphLocalTensors.insert(graph.initializer(i).name());
   for (int i = 0; i < graph.node_size(); i++)
      for (int j = 0; j < graph.node(i).output_size(); j++)
         graphLocalTensors.insert(graph.node(i).output(j));

   // loop at graph inputs
   std::map<std::string, int> allInputs;
   for (int i = 0; i < graph.input_size(); i++) {
      allInputs[graph.input(i).name()] = -1;
   }
   do {
      auto psize = nodesOrder.size();
      for (int i = 0; i < graph.node_size(); i++) {
         if (foundNodes[i])
            continue;
         // check if all input exists add to list
         bool existInputs = true;
         int input_size = graph.node(i).input_size();
         // special case for Reshape where shape is input and not a weight tensor
         if (fVerbose)
            std::cout << "Checking input of  Node " << i << " : " << graph.node(i).name() << std::endl;
         for (int j = 0; j < input_size; j++) {
            std::string name = graph.node(i).input(j);
            // skip empty names
            if (!name.empty()) {
               // A tensor is available if it is: a graph input/previously computed node output
               // (allInputs), an initializer (allInitializedTensors), or an outer-scope tensor
               // referenced from a subgraph.  Outer-scope means: registered in the parser's type
               // map AND not produced by any node/input/initializer of the current graph.  The
               // second condition prevents cross-model contamination from prior parsing passes.
               bool isOuterScope = !graphLocalTensors.count(name) && IsRegisteredTensorType(name);
               bool available = (allInputs.find(name) != allInputs.end() ||
                                 allInitializedTensors.find(name) != allInitializedTensors.end() ||
                                 isOuterScope);
               existInputs &= available;
               if (fVerbose) {
                  std::cout << "\t\t input " << name << " "
                     << bool(allInputs.find(name) != allInputs.end()) << "  " <<
                     bool(allInitializedTensors.find(name) != allInitializedTensors.end()) << "  " <<
                     bool(isOuterScope) << "  "
                     << existInputs << std::endl;
               }
            }
         }
         if (!existInputs) {
            if (fVerbose) {
               std::cout << "skip node " << graph.node(i).op_type() << "  " << graph.node(i).name() << " inputs are not existing ";
               for (int j = 0; j < input_size; j++) {
                  std::cout << graph.node(i).input(j) << " ";
               }
               std::cout << std::endl;
            }
            continue;
         }

         // adding node to the currectly ordered list
         if (verbose)
            std::cout << "===> New node " << graph.node(i).op_type() << "  " << graph.node(i).name() << " order " << i << std::endl;

         nodesOrder.push_back(i);
         foundNodes[i] = true;
         // register the outputs
         for (int j = 0; j < graph.node(i).output_size(); j++) {
            if (fVerbose) std::cout << "\toutput : " << graph.node(i).output(j) << std::endl;
            allInputs[graph.node(i).output(j)] = i;
         }
      }
      // no increment in nodes - something wrong
      if (nodesOrder.size() == psize) {
         int ilast = nodesOrder.back();
         std::cout << "cannot find a new node after " << graph.node(ilast).op_type() << " " << graph.node(ilast).name() << std::endl;
         throw std::runtime_error("SOFIE - cannot find a new node ");
      }
   } while ((int)nodesOrder.size() < graph.node_size());


   // find list of children for each operator (used for fusing oiperators)
   std::vector<std::vector<int>> nodesChildren(graph.node_size());

   for (int k = 0; k < graph.node_size(); k++) {
      int i = nodesOrder[k];
      // compute the number of output for the operators
      if (graph.node(i).output_size() > 0) nodesChildren[i].reserve(graph.node(i).output_size());
      for (const auto& output_name : graph.node(i).output()) {
         // loop on all nodes
         for (int l = k; l < graph.node_size(); l++) {
            int j = nodesOrder[l];
            for (const auto& input_name : graph.node(j).input()) {
               if (input_name == output_name)
                  nodesChildren[i].push_back(j);
            }
         }
      }
   }

   // print lit of order operators with list of inputs and list of children nodes
   if (verbose) {
      std::cout << "\nGraph operator list (re-ordered)\n";
      for (int k = 0; k < graph.node_size(); k++) {
         int i = nodesOrder[k];
         std::cout << "\tOperator " << i << " : " << graph.node(i).op_type() << " , " << graph.node(i).name() << " input tensors : {";
            for (int j = 0; j < graph.node(i).input_size(); j++) {
            std::cout << graph.node(i).input(j);
            if (j < graph.node(i).input_size() - 1)
               std::cout << ", ";
         }
         std::cout << " } ";
         std::cout << " children : {";
         for ( const auto & ichild : nodesChildren[i]) {
            std::cout << " [ " << ichild << " " << graph.node(ichild).op_type() << " , " << graph.node(ichild).name() << "]";
         }
         std::cout << "}" << std::endl;
      }
   }

   // fill model with operators
   if (verbose) {
      std::cout << "Fill RModel with operators...\n";
   }

   // we have to record order of node execution separately to
   // account for fused operators.
   size_t node_order_exec = 0;
   for (int i = 0; i < graph.node_size(); i++) {
      std::string op_type = graph.node(nodesOrder[i]).op_type();

      if (verbose) {
         std::cout << "\t" << i << "  " << nodesOrder[i] << " parsing operator " << op_type << std::endl;
      }

      std::unique_ptr<ROperator> op = ParseOperator(i, graph, nodesOrder, nodesChildren[nodesOrder[i]]);
      if (!op) {
         if (verbose) {
            std::cout << "\t\tskipping operator since it is fused with previous one" << std::endl;
         }
         // for skipping the fused nodes like Add after MatMul
         continue;
      }
      // assign operator name for profiling
      const auto &nodeproto = graph.node(nodesOrder[i]);
      op->fName = nodeproto.name();
      if (op->fName.empty()) {
         op->fName = nodeproto.op_type() + "_" + std::to_string(i);
      }
      rmodel.AddOperator(std::move(op), node_order_exec++);
   }

   std::vector<std::string> outputnames;
   if (verbose)
      std::cout << "\nParsing Graph output list\n";
   for (int i = 0; i < graph.output_size(); i++) {
      if (verbose)
         std::cout << "\toutput " << i << " name " << graph.output(i).name() << std::endl;
      outputnames.push_back(graph.output(i).name());
   }
   rmodel.AddOutputTensorNameList(outputnames);

   return;
}

} // namespace SOFIE
