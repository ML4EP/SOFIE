#include <algorithm>
#include <cctype>
#include <climits>
#include <limits>
#include <memory>
#include <sstream>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "../inc/SOFIE/SOFIE_common.hxx"

#include "SOFIE/RModel.hxx"
#include "SOFIE/RModelFusion_ALPAKA.hxx"
#include "SOFIE/RModelProfilerGPU.hxx"
#include "SOFIE/SOFIE_common.hxx"

namespace SOFIE {

// device buffer / view alias per tensor type; must match the 'using Buf* / ViewConst*'
// aliases emitted into the generated Session
static std::string GetBufType(ETensorType t) {
   switch (t) {
      case ETensorType::FLOAT:  return "BufF1D";
      case ETensorType::DOUBLE: return "BufD1D";
      case ETensorType::INT32:  return "BufI321D";
      case ETensorType::INT64:  return "BufI641D";
      case ETensorType::BOOL:
      case ETensorType::UINT8:  return "BufUI81D";
      default:
         throw std::runtime_error("sofie: tensor type " + ConvertTypeToString(t) +
                                  " is not supported on the ALPAKA backend");
   }
}

static std::string GetViewConstType(ETensorType t) {
   switch (t) {
      case ETensorType::FLOAT:  return "ViewConstF1D";
      case ETensorType::DOUBLE: return "ViewConstD1D";
      case ETensorType::INT32:  return "ViewConstI321D";
      case ETensorType::INT64:  return "ViewConstI641D";
      case ETensorType::BOOL:
      case ETensorType::UINT8:  return "ViewConstUI81D";
      default:
         throw std::runtime_error("sofie: tensor type " + ConvertTypeToString(t) +
                                  " is not supported on the ALPAKA backend");
   }
}

// declaration + allocation line for one device buffer
static std::string AllocBufLine(const std::string &name, ETensorType t, const std::string &length) {
   return GetBufType(t) + " deviceBuf_" + name + " = alpaka::allocBuf<" + ConvertOutputTypeToString(t) +
          ", Idx>(devAcc, Ext1D::all(Idx{" + length + "}));\n";
}

static std::string GetViewType(ETensorType t) {
   switch (t) {
      case ETensorType::FLOAT:  return "ViewF1D";
      case ETensorType::DOUBLE: return "ViewD1D";
      case ETensorType::INT32:  return "ViewI321D";
      case ETensorType::INT64:  return "ViewI641D";
      case ETensorType::BOOL:
      case ETensorType::UINT8:  return "ViewUI81D";
      default:
         throw std::runtime_error("sofie: tensor type " + ConvertTypeToString(t) +
                                  " is not supported on the ALPAKA backend");
   }
}

// declaration line for a non-owning device view (null placeholder, assigned from pool later)
static std::string DeclareViewLine(const std::string &name, ETensorType t) {
   return GetViewType(t) + " deviceBuf_" + name +
          "{static_cast<" + ConvertOutputTypeToString(t) + "*>(nullptr), devAcc, Ext1D::all(Idx{0})};\n";
}

void RModel::GenerateInitializedTensorInfo_GPU_ALPAKA() {
   if (!fInitializedTensors.empty()){
      fGC += "\n// initialized tensors for weights\n";
   }

   for (auto &i : fInitializedTensors) {
      if (!fUseWeightFile || i.second.IsConstantTensor()) {
         if (i.second.type() == ETensorType::FLOAT)
            fGC += GenerateConstantTensorCode<float>(i);
         else if (i.second.type() == ETensorType::INT64)
            fGC += GenerateConstantTensorCode<int64_t>(i);
         else if (i.second.type() == ETensorType::INT32)
            fGC += GenerateConstantTensorCode<int32_t>(i);

         else if (i.second.type() == ETensorType::BOOL ||
                  i.second.type() == ETensorType::UINT8)
            fGC += GenerateConstantTensorCode<uint8_t>(i);
      }

         size_t length = ConvertShapeToLength(i.second.shape());
         fGC += AllocBufLine(i.first, i.second.type(), std::to_string(length));

   }
}

void RModel::GeneratePersistentTensorInfo_GPU_ALPAKA()
{
   std::set<std::string> persistentTensors;

   for (size_t id = 0; id < fOperators.size(); ++id) {
      if (fFusion.skip.count(id)) continue;

      for (const auto &name : fOperators[id]->GetPersistentTensorNames_GPU_ALPAKA())
         persistentTensors.insert(name);
   }

   if (persistentTensors.empty())
      return;

   fGC += "\n// persistent state tensors\n";

   for (const auto &name : persistentTensors) {
      const ETensorType type = GetTensorType(name);
      const size_t length = ConvertShapeToLength(GetTensorShape(name));

      if (type == ETensorType::FLOAT)
         fGC += "BufF1D deviceBuf_" + name + " = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(length) + "}));\n";
      else if (type == ETensorType::DOUBLE)
         fGC += "BufD1D deviceBuf_" + name + " = alpaka::allocBuf<double, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(length) + "}));\n";
      else if (type == ETensorType::INT32)
         fGC += "BufI321D deviceBuf_" + name + " = alpaka::allocBuf<int32_t, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(length) + "}));\n";
      else if (type == ETensorType::INT64)
         fGC += "BufI641D deviceBuf_" + name + " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(length) + "}));\n";
      else if (type == ETensorType::BOOL || type == ETensorType::UINT8)
         fGC += "BufUI81D deviceBuf_" + name + " = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(length) + "}));\n";
      else
         throw std::runtime_error("Unsupported persistent GPU tensor type: " + name);
   }
}

void RModel::GenerateTemporaryInitializedTensorContainers_GPU_ALPAKA()
{
   if (!fInitializedTensors.empty())
      fGC += "// temporary initialized tensors for loading weights\n";

   for (auto &i : fInitializedTensors) {
      if (fUseWeightFile && !i.second.IsConstantTensor()) {
         // case of tensors which are read from a file
         size_t length = ConvertShapeToLength(i.second.shape());
         fGC += "std::vector<" + ConvertOutputTypeToString(i.second.type()) + "> tensor_" + i.first + "(" +
                std::to_string(length) + ");\n";
      }
   }
}

namespace {
// A dynamic Dim's .param can be a bare identifier  or a
// computed expression combining several dims. Checking the whole string
// against the known-params map only works for the bare-identifier case; for
// an expression every individual symbol inside it must be checked instead.
// Returns false as soon as any identifier-like token isn't a known param,
// which really is only known at inference
// time, not at Session-construction time.
bool AllShapeExprTokensKnown(const std::string &expr,
                             const std::unordered_map<std::string, std::string> &knownParams)
{
   size_t i = 0;
   while (i < expr.size()) {
      if (std::isalpha(static_cast<unsigned char>(expr[i])) || expr[i] == '_') {
         size_t j = i;
         while (j < expr.size() && (std::isalnum(static_cast<unsigned char>(expr[j])) || expr[j] == '_'))
            ++j;
         bool isNamespaceQualifier = (j + 1 < expr.size() && expr[j] == ':' && expr[j + 1] == ':');
         bool isFunctionCall = (j < expr.size() && expr[j] == '(');
         if (!isNamespaceQualifier && !isFunctionCall && knownParams.count(expr.substr(i, j - i)) == 0)
            return false;
         i = j;
      } else {
         ++i;
      }
   }
   return true;
}
} // namespace

void RModel::GenerateGPU_ALPAKA_Buffers() {
   if (!fIntermediateTensorInfos.empty()) {
      std::string tensor_declaration_block = "";

      for (auto &i : fIntermediateTensorInfos) {
         // Skip tensors that are purely intermediate within a fused kernel chain
         if (fFusion.internal.count(i.first)) continue;

         size_t length = ConvertShapeToLength(i.second.shape);

         tensor_declaration_block += AllocBufLine(i.first, i.second.type, std::to_string(length));
      }

      if (tensor_declaration_block.length()) {
         fGC += "\n//--- declare and allocate the intermediate tensors\n" + tensor_declaration_block;
      }
   }

   // add also the dynamic tensors (only declarations, allocation will be done later)
   if (!fDynamicTensorInfos.empty()) {
      fGC += "//--- declare the dynamic tensors\n";
      for (auto &i : fDynamicTensorInfos) {
         if (fFusion.internal.count(i.first)) continue;
         bool runtimeShape = false;
         for (const auto &dim : i.second.shape) {
            if (dim.isParam && !AllShapeExprTokensKnown(dim.param, fShapeParams)) {
               runtimeShape = true;
               break;
            }
         }
         if (runtimeShape)
            fGC += AllocBufLine(i.first, i.second.type, "1");
         else
            fGC += DeclareViewLine(i.first, i.second.type);
      }
   }

   if (!fShapeTensors.empty()) {
      fGC += "//--- declare the shape tensors\n";
      for (auto &i : fShapeTensors) {
         size_t len = i.second.first.size();
         if (len == 0) continue;
         fGC += "int64_t tensor_" + i.first + "[" + std::to_string(len) + "];\n";
         fGC += "BufI641D deviceBuf_" + i.first + " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{" +
                std::to_string(len) + "}));\n";
      }
   }
}

void RModel::GenerateDynamicTensorInfo_GPU_ALPAKA() {
   fGC += "//---- allocate unified intermediate tensor pool\n";
   std::stringstream out;

   constexpr std::size_t kPoolAlign = 256;
   auto alignUp = [](std::size_t v, std::size_t a) -> std::size_t { return (v + a - 1) & ~(a - 1); };

   // --- Unified pool: lifetime-aware packing of all pooled intermediates ---
   if (!fUnifiedPoolTensorNames.empty()) {
      struct PoolEntry {
         std::string name;
         ETensorType type;
         std::string sizeExpr;   // bytes (only meaningful when !isConst)
         std::string lengthExpr; // elements
         size_t producerOp;
         size_t lastUseOp;
         bool isConst;
         std::size_t constBytes; // valid only when isConst
      };
      std::vector<PoolEntry> entries;

      // Build producer map: tensor → first op that outputs it
      std::unordered_map<std::string, size_t> producerMap;
      for (size_t opIdx = 0; opIdx < fOperators.size(); ++opIdx) {
         if (fFusion.skip.count(opIdx)) continue;
         for (const auto &outName : fOperators[opIdx]->GetOpOutputTensors()) {
            std::string name(outName);
            if (producerMap.find(name) == producerMap.end())
               producerMap[name] = opIdx;
         }
      }

      for (const auto &tname : fUnifiedPoolTensorNames) {
         ETensorType type;
         std::string lengthExpr;
         size_t typeSize;
         bool isConst;
         std::size_t constBytes = 0;

         // Look up in dynamic tensors first, then static
         auto dynIt = fDynamicTensorInfos.find(tname);
         auto statIt = fIntermediateTensorInfos.find(tname);
         if (dynIt != fDynamicTensorInfos.end()) {
            type = dynIt->second.type;
            typeSize = GetTypeSize(type);
            lengthExpr = ConvertDimShapeToLength(dynIt->second.shape);
            isConst = std::none_of(dynIt->second.shape.begin(), dynIt->second.shape.end(),
                                    [](const Dim &d) { return d.isParam; });
            if (isConst)
               constBytes = alignUp(ConvertShapeToLength(dynIt->second.shape) * typeSize, kPoolAlign);
         } else if (statIt != fIntermediateTensorInfos.end()) {
            type = statIt->second.type;
            typeSize = GetTypeSize(type);
            lengthExpr = std::to_string(ConvertShapeToLength(statIt->second.shape));
            isConst = true;
            constBytes = alignUp(ConvertShapeToLength(statIt->second.shape) * typeSize, kPoolAlign);
         } else {
            continue;
         }

         std::string sizeExpr = "(" + lengthExpr + ") * " + std::to_string(typeSize);

         size_t producer = 0;
         auto pIt = producerMap.find(tname);
         if (pIt != producerMap.end()) producer = pIt->second;

         size_t lastUse = producer;
         auto fIt = fIntermediateTensorFrequencyLookup.find(tname);
         if (fIt != fIntermediateTensorFrequencyLookup.end()) lastUse = fIt->second;

         entries.push_back({tname, type, sizeExpr, lengthExpr, producer, lastUse, isConst, constBytes});
      }

      if (!entries.empty()) {
         std::vector<size_t> constIdx, runtimeIdx;
         for (size_t i = 0; i < entries.size(); ++i)
            (entries[i].isConst ? constIdx : runtimeIdx).push_back(i);

         std::vector<size_t> constOffsets;
         std::size_t constTotalBytes = 0;
         if (!constIdx.empty()) {
            std::vector<TensorLifeInfo> lifeInfos;
            lifeInfos.reserve(constIdx.size());
            for (size_t idx : constIdx) {
               auto &e = entries[idx];
               int end = (int)(std::max(e.lastUseOp, e.producerOp) + 1);
               lifeInfos.push_back({(int)e.producerOp, end, e.constBytes});
            }
            auto result = OrganizeMemory(lifeInfos);
            constOffsets = result.offsets;
            constTotalBytes = result.total_bytes;
         }

         out << "\n" << SP << "// --- Unified tensor pool: lifetime-aware packing ---\n";
         out << SP << "{\n";
         out << SP << SP << "constexpr std::size_t kConstPoolBytes = " << constTotalBytes << ";\n";

         // Tensors whose size depends on a shape parameter only known once
         // the constructor runs are packed here, at construction time, with
         // the same coalescing planner used for the compile-time-constant
         // tensors above, appended after the constant region in the pool.
         if (!runtimeIdx.empty()) {
            out << SP << SP << "constexpr std::size_t kPoolAlign = 256;\n";
            out << SP << SP << "auto alignUp = [](std::size_t v, std::size_t a) -> std::size_t {\n";
            out << SP << SP << SP << "return (v + a - 1) & ~(a - 1);\n";
            out << SP << SP << "};\n";
            out << SP << SP << "std::vector<SOFIE::TensorLifeInfo> runtimePoolInfos;\n";
            out << SP << SP << "runtimePoolInfos.reserve(" << runtimeIdx.size() << ");\n";
            for (size_t idx : runtimeIdx) {
               auto &e = entries[idx];
               int end = (int)(std::max(e.lastUseOp, e.producerOp) + 1);
               out << SP << SP << "runtimePoolInfos.push_back({" << e.producerOp << ", " << end
                   << ", alignUp(" << e.sizeExpr << ", kPoolAlign)});\n";
            }
            out << SP << SP << "auto runtimePoolResult = OrganizeMemory(runtimePoolInfos);\n";
            out << SP << SP << "std::size_t poolTotalBytes = kConstPoolBytes + runtimePoolResult.total_bytes;\n";
         } else {
            out << SP << SP << "std::size_t poolTotalBytes = kConstPoolBytes;\n";
         }

         out << "\n" << SP << SP << "// Allocate unified pool buffer\n";
         out << SP << SP << "if (poolTotalBytes > 0)\n";
         out << SP << SP << SP << "fUnifiedPool = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{poolTotalBytes}));\n";
         out << SP << SP << "auto* poolBase = (poolTotalBytes > 0) ? alpaka::getPtrNative(fUnifiedPool) : nullptr;\n";
         out << SP << SP << "fUnifiedPoolSize = poolTotalBytes;\n\n";

         out << SP << SP << "// Assign tensor views into unified pool\n";
         for (size_t i = 0; i < constIdx.size(); ++i) {
            auto &e = entries[constIdx[i]];
            std::string typeName = ConvertOutputTypeToString(e.type);
            std::string viewType = GetViewType(e.type);
            out << SP << SP << "deviceBuf_" << e.name << " = " << viewType
                << "{reinterpret_cast<" << typeName << "*>(poolBase + " << constOffsets[i]
                << "), devAcc, Ext1D::all(Idx{" << e.lengthExpr << "})};\n";
         }
         for (size_t i = 0; i < runtimeIdx.size(); ++i) {
            auto &e = entries[runtimeIdx[i]];
            std::string typeName = ConvertOutputTypeToString(e.type);
            std::string viewType = GetViewType(e.type);
            out << SP << SP << "deviceBuf_" << e.name << " = " << viewType
                << "{reinterpret_cast<" << typeName << "*>(poolBase + kConstPoolBytes + runtimePoolResult.offsets["
                << i << "]), devAcc, Ext1D::all(Idx{" << e.lengthExpr << "})};\n";
         }

         out << SP << "}\n";
      }
   }

   fGC += out.str();
}

std::vector<std::string> RModel::GetOperatorKernelParams(size_t opIdx, const std::vector<std::string> &baseDynParamNames) const
{
   if (fInternalDynamicParams.empty() || opIdx >= fOperators.size())
      return baseDynParamNames;

   std::vector<std::string> params = baseDynParamNames;
   std::unordered_set<std::string> have(params.begin(), params.end());

   auto isIdentChar = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
   auto containsToken = [&](const std::string &expr, const std::string &token) {
      size_t pos = 0;
      while ((pos = expr.find(token, pos)) != std::string::npos) {
         bool leftOk = (pos == 0) || !isIdentChar(expr[pos - 1]);
         size_t end = pos + token.size();
         bool rightOk = (end == expr.size()) || !isIdentChar(expr[end]);
         if (leftOk && rightOk)
            return true;
         pos = end;
      }
      return false;
   };

   auto scan = [&](std::span<const std::string_view> names) {
      for (const auto &name : names) {
         std::vector<Dim> shape;
         try {
            shape = GetDimTensorShape(std::string(name));
         } catch (...) {
            continue; // not a tensor with a trackable shape (e.g. an attribute-only input)
         }
         for (const auto &d : shape) {
            if (!d.isParam)
               continue;
            for (const auto &internalParam : fInternalDynamicParams) {
               if (containsToken(d.param, internalParam) && have.insert(internalParam).second)
                  params.push_back(internalParam);
            }
         }
      }
   };

   scan(fOperators[opIdx]->GetOpInputTensors());
   scan(fOperators[opIdx]->GetOpOutputTensors());
   return params;
}

void RModel::ForEachInferArg_GPU_ALPAKA(const std::function<void(const std::string &)> &onParam,
                                        const std::function<void(const std::string &)> &onInput) const
{
   std::unordered_map<std::string, int> seen;
   for (auto &name : fInputTensorNames) {
      if (IsDimInputTensor(name)) {
         for (auto &d : GetDynamicTensorShape(name)) {
            if (d.isParam && seen.count(d.param) == 0) {
               seen[d.param] = 1;
               onParam(d.param);
            }
         }
      }
      onInput(name);
   }
}

std::string RModel::GenerateInferSignature_GPU_ALPAKA(bool isdecl) {

   std::string rGC;
   ForEachInferArg_GPU_ALPAKA(
      [&](const std::string &p) {
         if (isdecl) rGC += "size_t ";
         rGC += p + ",";
      },
      [&](const std::string &name) {
         if (isdecl) rGC += GetBufType(GetTensorType(name)) + " const ";
         rGC += "deviceBuf_" + name + ",";
      });

   if (fInputTensorNames.size() > 0) rGC.pop_back(); // remove last ","
   return rGC;
}

std::string RModel::GenerateImplSignature_GPU_ALPAKA(bool isdecl) {

   std::string rGC;
   ForEachInferArg_GPU_ALPAKA(
      [&](const std::string &p) {
         if (isdecl) rGC += "size_t ";
         rGC += p + ",";
      },
      [&](const std::string &name) {
         if (isdecl) rGC += GetViewConstType(GetTensorType(name)) + " const& ";
         rGC += "deviceBuf_" + name + ",";
      });

   if (fInputTensorNames.size() > 0) rGC.pop_back();
   return rGC;
}

void RModel::GenerateOutput_GPU_ALPAKA() {
   if (fVerbose)
      std::cout << "Generating main inference code for " << fName << std::endl;

   size_t outputSize = fOutputTensorNames.size();
   if (outputSize == 0)
      throw std::runtime_error("sofie: output size=0 are not supported");

   ETensorType eFirstOutputType = GetTensorType(*fOutputTensorNames.begin());
   bool sameOutputTypes = true;
   for (std::string const &name : fOutputTensorNames) {
      if (GetTensorType(name) != eFirstOutputType)
         sameOutputTypes = false;
   }

   auto IsPooledIntermediate = [this](const std::string &name) -> bool {
      return fIntermediateTensorInfos.count(name) > 0 &&
             fInitializedTensors.count(name) == 0 &&
             fDynamicTensorInfos.count(name) == 0 &&
             fFusion.internal.count(name) == 0;
   };

   auto GetOutputReturnType = [&](const std::string &name) -> std::string {
      ETensorType type = GetTensorType(name);
      const std::string storageName = ResolveAliasTensor(name);

      if (IsPooledIntermediate(storageName)) {
         if (type == ETensorType::FLOAT)  return "ViewF1D";
         if (type == ETensorType::DOUBLE) return "ViewD1D";
         if (type == ETensorType::INT32)  return "ViewI321D";
         if (type == ETensorType::INT64)  return "ViewI641D";
         if (type == ETensorType::BOOL)   return "ViewUI81D";
      }

      if (type == ETensorType::FLOAT)  return "BufF1D";
      if (type == ETensorType::DOUBLE) return "BufD1D";
      if (type == ETensorType::INT32)  return "BufI321D";
      if (type == ETensorType::INT64)  return "BufI641D";
      if (type == ETensorType::BOOL)   return "BufUI81D";

      throw std::runtime_error("Unsupported output tensor type: " + name);
   };

   auto GetOutputBufferName = [this](const std::string &name) -> std::string {
      const std::string storageName = ResolveAliasTensor(name);

      return "deviceBuf_" + storageName;
   };

   // A dynamic-shaped tensor that isn't in the unified pool (e.g. a graph
   // output, or a tensor whose shape isn't known until the constructor's
   // shape params) was declared with a placeholder size of 1 element.
   // Reallocate it here to its actual per-call size, computed from the
   // dynamic shape params available in _infer_impl, before the producing
   // operator (or fused kernel) writes into it.
   const std::set<std::string> pooledTensorNames(fUnifiedPoolTensorNames.begin(), fUnifiedPoolTensorNames.end());
   auto GenerateDynamicOutputReallocForNames = [this, &pooledTensorNames](const std::vector<std::string> &names) -> std::string {
      std::string code;
      for (const auto &name : names) {
         if (pooledTensorNames.count(name)) continue;
         auto it = fDynamicTensorInfos.find(name);
         if (it == fDynamicTensorInfos.end()) continue;
         // Shapes that depend on values only known at inference time (e.g. a
         // NonZero-derived count) aren't expressible from _infer_impl's
         // parameters alone; those operators reallocate their own output
         // buffer once the count is available.
         bool allKnown = true;
         for (const auto &dim : it->second.shape) {
            if (dim.isParam && !AllShapeExprTokensKnown(dim.param, fShapeParams)) {
               allKnown = false;
               break;
            }
         }
         if (!allKnown) continue;
         std::string lengthExpr = ConvertDimShapeToLength(it->second.shape);
         code += SP + "deviceBuf_" + name + " = alpaka::allocBuf<" + ConvertOutputTypeToString(it->second.type) +
                 ", Idx>(devAcc, Ext1D::all(Idx{static_cast<Idx>(" + lengthExpr + ")}));\n";
      }
      return code;
   };
   auto GenerateDynamicOutputRealloc = [this, &GenerateDynamicOutputReallocForNames](size_t opIdx) -> std::string {
      std::vector<std::string> names;
      for (const auto &outName : fOperators[opIdx]->GetOpOutputTensors())
         names.push_back(std::string(outName));
      return GenerateDynamicOutputReallocForNames(names);
   };

   // Collect deduplicated dynamic dimension parameter names in declaration order
   std::vector<std::string> dynParamNames;
   ForEachInferArg_GPU_ALPAKA([&](const std::string &p) { dynParamNames.push_back(p); },
                              [](const std::string &) {});

   fGC += "\n\n";

   fGC += "void _infer_impl(";
   fGC += GenerateImplSignature_GPU_ALPAKA();
   fGC += "){\n";
   // device buffers were sized in the ctor from its shape params; refuse larger ones
   // (same check and message as the CPU session)
   ForEachInferArg_GPU_ALPAKA(
      [&](const std::string &p) {
         fGC += SP + "if (" + p + " > " + GetMemberNameForDimShape(p) + ") {\n";
         fGC += SP + SP + "throw std::runtime_error(\"sofie: dynamic input tensor shape parameter " + p +
                " exceeds the initialized maximum allowed shape.\");\n";
         fGC += SP + "}\n";
      },
      [](const std::string &) {});

   // GPU profiling: _infer_impl is a member of Session, so fProfilingResults
   // is directly accessible without any alias.
   if (fProfile) {
      fGC += RModelProfilerGPU::GenerateBeginInferCode();
   }

   for (size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx) {
      if (fVerbose)
         std::cout << "Generating code for operator .... " << op_idx << std::endl;

      if (fFusion.skip.count(op_idx)) continue;

      if (const auto *kernelGroup = fFusion.FindKernel(op_idx)) {
         if (kernelGroup->launchOpIndex == op_idx) {
            for (const auto &branch : kernelGroup->branches)
               fGC += GenerateDynamicOutputReallocForNames(branch.outputTensors);
            fGC += Fusion::Launch(*this, *kernelGroup);
         }

         continue;
      }

      if (const auto *eltwiseGroup = fFusion.FindEltwise(op_idx)) {
         if (eltwiseGroup->launchOpIndex == op_idx) {
            fGC += GenerateDynamicOutputReallocForNames(eltwiseGroup->outputTensors);
            fGC += Fusion::Launch(*this, *eltwiseGroup);
         }
      } else {
         auto opDynParamNames = GetOperatorKernelParams(op_idx, dynParamNames);
         fGC += GenerateDynamicOutputRealloc(op_idx);
         if (fProfile) {
            fGC += RModelProfilerGPU::GenerateOperatorCode(*fOperators[op_idx], op_idx, opDynParamNames);
         } else {
            fGC += fOperators[op_idx]->Generate_GPU_ALPAKA(std::to_string(op_idx), opDynParamNames);
         }
      }
   }

   if (fProfile) {
      fGC += RModelProfilerGPU::GenerateEndInferCode();
   }

   fGC += "}\n\n";


   std::string spanDynDecl;
   for (auto &p : dynParamNames)
      spanDynDecl += ", size_t " + p;

   fGC += "void infer(std::span<ViewConstF1D const> inputs, std::span<ViewF1D> outputs" + spanDynDecl + "){\n";

   {
      // each symbol goes just before the first input whose shape introduces it
      fGC += SP + "_infer_impl(";
      bool first = true;
      size_t i_input = 0;
      auto appendArg = [&](const std::string &arg) {
         if (!first) fGC += ", ";
         fGC += arg;
         first = false;
      };
      ForEachInferArg_GPU_ALPAKA(appendArg, [&](const std::string &) {
         appendArg("inputs[" + std::to_string(i_input++) + "]");
      });
      fGC += ");\n";
   }

   // Copy member output buffers into caller-provided output views
   for (size_t i = 0; i < outputSize; i++) {
      std::string tensorName = *(fOutputTensorNames.begin() + i);
      fGC += SP + "alpaka::memcpy(queue, outputs[" + std::to_string(i) + "], " + GetOutputBufferName(tensorName) + ");\n";
   }
   fGC += SP + "alpaka::wait(queue);\n";
   fGC += "}\n\n";


   std::string returnType;

   if (outputSize == 1) {
      std::string tname = *fOutputTensorNames.begin();
      returnType = GetOutputReturnType(tname);
   } else {
      returnType = "std::tuple<";
      for (size_t i = 0; i < outputSize; i++) {
         std::string tname = *(fOutputTensorNames.begin() + i);
         returnType += GetOutputReturnType(tname);
         if (i < outputSize - 1) returnType += ",";
      }
      returnType += ">";
   }

   fGC += returnType + " infer(";
   fGC += GenerateInferSignature_GPU_ALPAKA();
   fGC += "){\n";

   std::vector<std::string> typedImplArgs;
   ForEachInferArg_GPU_ALPAKA(
      [&](const std::string &p) { typedImplArgs.push_back(p); },
      [&](const std::string &name) {
         std::string viewType = GetViewConstType(GetTensorType(name));
         fGC += SP + viewType + " const view_" + name +
                "{alpaka::getPtrNative(deviceBuf_" + name + "), devAcc, alpaka::getExtents(deviceBuf_" + name + ")};\n";
         typedImplArgs.push_back("view_" + name);
      });

   // Helper lambda: emit the _infer_impl call and return statement
   auto emitImplCallAndReturn = [&]() {
      fGC += SP + "_infer_impl(";
      for (size_t i = 0; i < typedImplArgs.size(); i++) {
         if (i > 0) fGC += ", ";
         fGC += typedImplArgs[i];
      }
      fGC += ");\n";

      fGC += SP + "return ";
      if (outputSize > 1) fGC += "{";
      for (size_t i = 0; i < outputSize; i++) {
         std::string tensorName = *(fOutputTensorNames.begin() + i);
         fGC += GetOutputBufferName(tensorName);
         if (i < outputSize - 1) fGC += ",";
      }
      if (outputSize > 1) fGC += "}";
      fGC += ";\n";
   };

   // Synchronous infer(): waits for all GPU work before returning
   fGC += SP + "_infer_impl(";
   for (size_t i = 0; i < typedImplArgs.size(); i++) {
      if (i > 0) fGC += ", ";
      fGC += typedImplArgs[i];
   }
   fGC += ");\n";
   fGC += SP + "alpaka::wait(queue);\n";
   fGC += SP + "return ";
   if (outputSize > 1) fGC += "{";
   for (size_t i = 0; i < outputSize; i++) {
      std::string tensorName = *(fOutputTensorNames.begin() + i);
      fGC += GetOutputBufferName(tensorName);
      if (i < outputSize - 1) fGC += ",";
   }
   if (outputSize > 1) fGC += "}";
   fGC += ";\n";
   fGC += "}\n\n";

   // Async infer_nosync(): returns immediately after enqueueing GPU work.
   // The caller must call alpaka::wait(session.queue) before reading results.
   // Used by benchmark loops to pipeline multiple inferences without
   // per-iteration CPU↔GPU synchronisation
   fGC += returnType + " infer_nosync(";
   fGC += GenerateInferSignature_GPU_ALPAKA();
   fGC += "){\n";

   // Re-emit the view declarations for the nosync overload
   typedImplArgs.clear();
   ForEachInferArg_GPU_ALPAKA(
      [&](const std::string &p) { typedImplArgs.push_back(p); },
      [&](const std::string &name) {
         std::string viewType = GetViewConstType(GetTensorType(name));
         fGC += SP + viewType + " const view_" + name +
                "{alpaka::getPtrNative(deviceBuf_" + name + "), devAcc, alpaka::getExtents(deviceBuf_" + name + ")};\n";
         typedImplArgs.push_back("view_" + name);
      });

   emitImplCallAndReturn();
   fGC += "}\n";
}

// Get the data member name corresponding to a tensor with a given name.
std::string TensorMember(std::string const &name) {
   return "tensor_" + name;
}

void RModel::GenerateSessionCode_GPU_ALPAKA() {
   // the model's dynamic shape parameters in infer-argument order
   std::vector<std::string> dynParamNames;
   ForEachInferArg_GPU_ALPAKA([&](const std::string &p) { dynParamNames.push_back(p); },
                              [](const std::string &) {});

   std::set<SOFIE::OperatorKind> registered_operators;

   std::set<SOFIE::OperatorKind> single_initialized_operators = {
      SOFIE::OperatorKind::RELU,
      SOFIE::OperatorKind::SIGMOID,
      SOFIE::OperatorKind::TANH,
      SOFIE::OperatorKind::LEAKYRELU,
      SOFIE::OperatorKind::EINSUM,
      SOFIE::OperatorKind::ELU,
      SOFIE::OperatorKind::UNARY_RECIPROCAL,
      SOFIE::OperatorKind::UNARY_SQRT,
      SOFIE::OperatorKind::UNARY_NEG,
      SOFIE::OperatorKind::UNARY_EXP,
      SOFIE::OperatorKind::UNARY_LOG,
      SOFIE::OperatorKind::UNARY_SIN,
      SOFIE::OperatorKind::UNARY_COS,
      SOFIE::OperatorKind::UNARY_ABS,
      SOFIE::OperatorKind::UNARY_SOFTPLUS,
      SOFIE::OperatorKind::UNARY_ATAN,
      SOFIE::OperatorKind::UNARY_FLOOR,
      SOFIE::OperatorKind::NOT,
      SOFIE::OperatorKind::SELU,
      SOFIE::OperatorKind::GEMM
   };

   bool OpNeedsBlas = false;

   fGC += "\n//--- ALPAKA Kernels\n";
   for (size_t id = 0; id < fOperators.size(); id++) {
      if(fOperators[id]->GetKind() == OperatorKind::GEMM || fOperators[id]->GetKind() == OperatorKind::CONV) {
         OpNeedsBlas = true;
      }

      if (const auto *kernelGroup = fFusion.FindKernel(id)) {
         if (kernelGroup->branches.front().opIndices.front() == id)
            fGC += Fusion::Kernel(*this, *kernelGroup);
      } else {
         if (const auto *eltwiseGroup = fFusion.FindEltwise(id)) {
            if (eltwiseGroup->opIndices[0] == id)
               fGC += Fusion::Kernel(*this, *eltwiseGroup);
         } else {
            auto idDynParamNames = GetOperatorKernelParams(id, dynParamNames);
            if (single_initialized_operators.find(fOperators[id]->GetKind()) != single_initialized_operators.end()) {
               if (registered_operators.find(fOperators[id]->GetKind()) == registered_operators.end()) {
                  if (fVerbose)
                     std::cout << "Generating ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
                  fGC += fOperators[id]->Generate_GPU_Kernel_ALPAKA(std::to_string(id), idDynParamNames);
                  registered_operators.insert(fOperators[id]->GetKind());
               }
            } else {
               if (fVerbose)
                  std::cout << "Generating ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
               fGC += fOperators[id]->Generate_GPU_Kernel_ALPAKA(std::to_string(id), idDynParamNames);
            }
         }
      }
   }



   if (fKernelOnly)
      return;

   // define the Session struct (for GNN this is generated in RModel_GNN)
  fGC += "\n\ntemplate <typename tagAcc>\n";
   if (fUseSession) {
      fGC += "struct Session {\n\n";
   }

   // define host and device accelerators
    fGC += "using Idx = std::size_t;\n";
    fGC += "using Dim = alpaka::DimInt<1>;\n";
    fGC += "using Acc = alpaka::TagToAcc<tagAcc, Dim, Idx>;\n";
    fGC += "using DevAcc = alpaka::Dev<Acc>;\n\n";
    fGC += "using QueueProperty = alpaka::NonBlocking;\n";
    fGC += "using QueueAcc = alpaka::Queue<Acc, QueueProperty>;\n\n";
    fGC += "using BufF1D = alpaka::Buf<Acc, float, Dim, Idx>;\n";
    fGC += "using BufD1D = alpaka::Buf<Acc, double, Dim, Idx>;\n";
    fGC += "using BufI321D = alpaka::Buf<Acc, int32_t, Dim, Idx>;\n";
    fGC += "using BufI641D = alpaka::Buf<Acc, int64_t, Dim, Idx>;\n";
    fGC += "using BufUI81D = alpaka::Buf<Acc, uint8_t, Dim, Idx>;\n\n";
    fGC += "// Non-owning device view types (ViewPlainPtr) for the span-based infer interface\n";
    fGC += "using ViewF1D = alpaka::ViewPlainPtr<DevAcc, float, Dim, Idx>;\n";
    fGC += "using ViewConstF1D = alpaka::ViewPlainPtr<DevAcc, const float, Dim, Idx>;\n";
    fGC += "using ViewD1D = alpaka::ViewPlainPtr<DevAcc, double, Dim, Idx>;\n";
    fGC += "using ViewConstD1D = alpaka::ViewPlainPtr<DevAcc, const double, Dim, Idx>;\n";
    fGC += "using ViewI321D = alpaka::ViewPlainPtr<DevAcc, int32_t, Dim, Idx>;\n";
    fGC += "using ViewConstI321D = alpaka::ViewPlainPtr<DevAcc, const int32_t, Dim, Idx>;\n";
    fGC += "using ViewI641D = alpaka::ViewPlainPtr<DevAcc, int64_t, Dim, Idx>;\n";
    fGC += "using ViewConstI641D = alpaka::ViewPlainPtr<DevAcc, const int64_t, Dim, Idx>;\n";
    fGC += "using ViewUI81D = alpaka::ViewPlainPtr<DevAcc, uint8_t, Dim, Idx>;\n";
    fGC += "using ViewConstUI81D = alpaka::ViewPlainPtr<DevAcc, const uint8_t, Dim, Idx>;\n\n";

    fGC += "\nalpaka::Platform<Acc> const platform{};\n";
    fGC += "DevAcc devAcc = alpaka::getDevByIdx(platform, 0);\n";
    fGC += "alpaka::PlatformCpu platformHost{};\n";
    fGC += "alpaka::DevCpu hostAcc = alpaka::getDevByIdx(platformHost, 0);\n";
    fGC += "QueueAcc queue{devAcc};\n";
    fGC += "Idx threadsPerBlock = 256;\n";
    fGC += "\nusing Ext1D = alpaka::Vec<Dim, Idx>;\n";
    fGC += "using Vec = alpaka::Vec<Dim, Idx>;\n";
    if (OpNeedsBlas) {
         fGC += "\n\n// BLAS declarations\n";
         fGC += "sofieBLAS<tagAcc> blas{queue};\n";
    }

   /// Allocate memory efficiently
   GenerateInitializedTensorInfo_GPU_ALPAKA();
   GeneratePersistentTensorInfo_GPU_ALPAKA();

   // --- Unified intermediate tensor classification and pool declaration ---
   // Both static (fIntermediateTensorInfos) and dynamic (fDynamicTensorInfos)
   // intermediates go into a single runtime-allocated pool when their shapes
   // can be determined at constructor time.
   {
      std::set<std::string> outputSet(fOutputTensorNames.begin(), fOutputTensorNames.end());

      // Identify operator outputs and persistent tensors (to distinguish scratch tensors)
      std::set<std::string> opOutputTensors;
      std::set<std::string> persistentTensors;
      for (const auto &op : fOperators) {
         for (const auto &name : op->GetOpOutputTensors())
            opOutputTensors.insert(std::string(name));
         for (const auto &name : op->GetPersistentTensorNames_GPU_ALPAKA())
            persistentTensors.insert(name);
      }

      std::string tensorDecls;
      std::vector<std::string> unifiedPoolTensors;

      // 1. Static intermediates
      for (auto &i : fIntermediateTensorInfos) {
         if (fFusion.internal.count(i.first)) continue;
         if (IsAliasTensor(i.first)) continue;
         if (persistentTensors.count(i.first)) continue;

         bool isOpOutput = opOutputTensors.count(i.first) > 0;
         if (isOpOutput) {
            // Poolable static intermediate → null view, assigned from pool in constructor
            tensorDecls += DeclareViewLine(i.first, i.second.type);
            unifiedPoolTensors.push_back(i.first);
         } else {
            // Scratch tensor (not any operator's output) → own owning buffer
            size_t length = ConvertShapeToLength(i.second.shape);
            tensorDecls += AllocBufLine(i.first, i.second.type, std::to_string(length));
         }
      }

      // 2. Dynamic intermediates
      for (auto &i : fDynamicTensorInfos) {
         if (fFusion.internal.count(i.first)) continue;

         bool runtimeShape = false;
         for (const auto &dim : i.second.shape) {
            if (dim.isParam && !AllShapeExprTokensKnown(dim.param, fShapeParams)) {
               runtimeShape = true;
               break;
            }
         }
         bool isOutput = outputSet.count(i.first) > 0;
         if (runtimeShape || isOutput) {
            tensorDecls += AllocBufLine(i.first, i.second.type, "1");
         } else {
            tensorDecls += DeclareViewLine(i.first, i.second.type);
            unifiedPoolTensors.push_back(i.first);
         }
      }

      if (!tensorDecls.empty())
         fGC += "\n//--- declare intermediate tensors\n" + tensorDecls;

      // 3. Shape tensors: host-side arrays + device buffers
      if (!fShapeTensors.empty()) {
         fGC += "\n//--- declare the shape tensors\n";
         for (auto &i : fShapeTensors) {
            size_t len = i.second.first.size();
            if (len == 0) continue;
            fGC += "int64_t tensor_" + i.first + "[" + std::to_string(len) + "];\n";
            fGC += "BufI641D deviceBuf_" + i.first + " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{"
                   + std::to_string(len) + "}));\n";
         }
      }

      // Unified pool buffer (allocated to final size in constructor)
      if (!unifiedPoolTensors.empty()) {
         fGC += "\n//--- unified memory pool buffer (allocated in constructor)\n";
         fGC += "BufUI81D fUnifiedPool = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
         fGC += "std::size_t fUnifiedPoolSize = 0;\n";
         fGC += "std::size_t GetIntermediateMemoryPoolSize() const { return fUnifiedPoolSize; }\n";
         // The unified pool is packed once, in the constructor, into a single
         // fixed-size buffer with no further runtime alloc/free churn, so
         // there is no live free space or fragmentation to report post-hoc
         fGC += "std::size_t GetLargestFreeBlock() const { return 0; }\n";
         fGC += "std::size_t GetTotalFreeMemory() const { return 0; }\n";
         fGC += "double GetFragmentation() const { return 0.0; }\n";
      }

      fUnifiedPoolTensorNames = std::move(unifiedPoolTensors);
   }

   GenerateOperatorDeclarations();

   // inject profiling session data member
   if (fProfile) {
      fGC += RModelProfilerGPU::GenerateSessionMembers();
   }

   // Session constructor(s)
   if (fUseSession) {
      std::string sessionName = "Session";

      std::string fileName;
      if (fUseWeightFile) {
         fileName = fName;
         if (fWeightFile == WeightFileType::Text)
            fileName += ".dat";
      }

      // ---- build constructor body into a temporary string ----
      {
         std::string savedGC = fGC;
         fGC.clear();

         GenerateTemporaryInitializedTensorContainers_GPU_ALPAKA();
         if (fUseWeightFile) {
            fGC += "\n//--- reading weights from file\n";
            fGC += "using SOFIE::ReadTensorFromStream;\n";
            ReadInitializedTensorsFromFile();
            fGC += "\n";
         }
         MoveInitializedTensorsToBuffers_ALPAKA();
         GenerateDynamicTensorInfo_GPU_ALPAKA();
         for (auto &op : fOperators) {
            for (const auto &outName : op->GetOpOutputTensors()) {
               if (std::find(fUnifiedPoolTensorNames.begin(), fUnifiedPoolTensorNames.end(), outName) != fUnifiedPoolTensorNames.end())
                  op->MarkOutputAsPooled(outName);
            }
         }
         for (size_t id = 0; id < fOperators.size(); id++) {
            if (fFusion.skip.count(id)) continue;
            fGC += fOperators[id]->GenerateInitCode_GPU_ALPAKA();
            if (fOperators[id]->GetKind() == OperatorKind::GEMM || fOperators[id]->GetKind() == OperatorKind::CONV) {
               for (auto &blasCfg : fOperators[id]->GetBlasConfigs()) {
                  if (!blasCfg.empty())
                     fGC += "\nblas.addOperationConfig(" + blasCfg + ");\n";
               }
            }
         }
         fGC += "\nalpaka::wait(queue);\n";

         /*
          * Constructor shape parameters: first the ones the infer signature introduces, in that
          * order, so positional constructor arguments cannot permute on multi-symbol models; then
          * the ones an operator registered itself. The same list drives the constructor signature, 
          * the Session members that keep the construction-time values, and their assignment at 
          * the top of the constructor body.
          */
         std::vector<std::string> ctorParamNames;
         ForEachInferArg_GPU_ALPAKA(
            [&](const std::string &p) { ctorParamNames.push_back(p); },
            [](const std::string &) {});
         for (auto &p : fShapeParams) {
            if (std::find(ctorParamNames.begin(), ctorParamNames.end(), p.first) == ctorParamNames.end())
               ctorParamNames.push_back(p.first);
         }

         std::string ctorBody;
         for (auto &p : ctorParamNames)
            ctorBody += SP + GetMemberNameForDimShape(p) + " = " + p + ";\n";
         ctorBody += fGC;
         fGC = savedGC;

         std::string ctorParams;
         for (auto &p : ctorParamNames)
            ctorParams += ",\n        size_t " + p + " = " + fShapeParams[p];

         /*
          * One Session member per shape parameter, the same members the CPU session declares.
          * The infer arguments are checked against them at the top of _infer_impl; a parameter
          * registered by an operator is checked by that operator.
          */
         for (auto &p : ctorParamNames)
            fGC += "size_t " + GetMemberNameForDimShape(p) + ";\n";

         std::string initArgs;
         for (auto &p : ctorParamNames)
            initArgs += ", " + p;

         std::string initParams = "std::string filename";
         for (auto &p : ctorParamNames)
            initParams += ", size_t " + p;

         fGC += "\nvoid InitSession(" + initParams + ") {\n";
         fGC += ctorBody;
         fGC += "}\n\n";

         fGC += "public:\n";

         // (1) default-queue constructor
         if (fUseWeightFile)
            fGC += "\n\n" + sessionName + "(std::string filename = \"" + fileName + "\"";
         else
            fGC += "\n\n" + sessionName + "(std::string filename = \"\"";
         fGC += ctorParams;
         fGC += ") {\n";
         fGC += SP + "InitSession(filename" + initArgs + ");\n";
         fGC += "}\n\n";

         // (2) external-queue constructor
         if (fUseWeightFile)
            fGC += sessionName + "(QueueAcc& extQueue, std::string filename = \"" + fileName + "\"";
         else
            fGC += sessionName + "(QueueAcc& extQueue, std::string filename = \"\"";
         fGC += ctorParams;
         fGC += ")\n    : queue(extQueue)";
         if (OpNeedsBlas)
            fGC += ", blas(queue)";
         fGC += "\n{\n";
         fGC += SP + "InitSession(filename" + initArgs + ");\n";
         fGC += "}\n\n";
      }
   }

   registered_operators.clear();

   for (size_t id = 0; id < fOperators.size(); id++) {
      // Same as the kernel-struct loop above: fused activation ops must still
      // declare their member variable (e.g. `leakyReluKernel`) even though
      // their Generate_GPU_ALPAKA call is skipped in the infer-body loop.

      if (const auto *kernelGroup = fFusion.FindKernel(id)) {
         if (kernelGroup->branches.front().opIndices.front() == id) {
            const std::string sfx = kernelGroup->suffix();
            fGC += SP + "KernelFusionKernel" + sfx + " kernelFusionKernel" + sfx + ";\n";
         }
      } else {
         if (const auto *eltwiseGroup = fFusion.FindEltwise(id)) {
            if (eltwiseGroup->opIndices[0] == id) {
               const std::string sfx = eltwiseGroup->suffix();
               fGC += SP + "FusedEltwiseKernel" + sfx + " fusedEltwiseKernel" + sfx + ";\n";
            }
         } else {
            if (single_initialized_operators.find(fOperators[id]->GetKind()) != single_initialized_operators.end()) {
               if (registered_operators.find(fOperators[id]->GetKind()) == registered_operators.end()) {
                  if (fVerbose)
                     std::cout << "Declaring ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
                  fGC += fOperators[id]->Generate_GPU_Kernel_Definitions_ALPAKA(std::to_string(id));
                  registered_operators.insert(fOperators[id]->GetKind());
               }
            } else {
               if (fVerbose)
                  std::cout << "Declaring ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
               fGC += fOperators[id]->Generate_GPU_Kernel_Definitions_ALPAKA(std::to_string(id));
            }
         }
      }
   }

   GenerateOutput_GPU_ALPAKA();

   // Emit resetState() for recurrent/stateful operator buffers, and zero all
   // intermediate tensors (static- and dynamic-shape alike) to reset model state.
   // A dynamic tensor's device buffer is zeroed over its full allocated capacity.
   if (fUseSession) {
      fGC += "\nvoid resetState(QueueAcc& queue) {\n";

      for (size_t id = 0; id < fOperators.size(); ++id) {
         if (fFusion.skip.count(id))
            continue;
         fGC += fOperators[id]->GenerateResetStateCode_GPU_ALPAKA();
      }
      for (auto &i : fIntermediateTensorInfos) {
         // Alias tensors (e.g. Reshape/Squeeze outputs) are zero-copy views declared as
         // locals inside _infer_impl, not Session members, and memsetting one would zero
         // the aliased tensor's storage out from under it anyway.
         if (fFusion.internal.count(i.first) || IsAliasTensor(i.first)) continue;
         fGC += SP + "alpaka::memset(queue, deviceBuf_" + i.first + ", 0);\n";
      }
      for (auto &i : fDynamicTensorInfos) {
         if (fFusion.internal.count(i.first) || IsAliasTensor(i.first)) continue;
         fGC += SP + "alpaka::memset(queue, deviceBuf_" + i.first + ", 0);\n";
      }
      fGC += SP + "alpaka::wait(queue);\n";
      fGC += "}\n";
   }
   // inject GPU profiling utility functions and memory report inside Session struct
   if (fProfile && fUseSession) {
      fGC += RModelProfilerGPU::GenerateUtilityFunctions();
      auto memInfo = RModelProfilerGPU::ComputeMemoryInfo(*this);
      fGC += RModelProfilerGPU::GenerateMemoryReport(memInfo);
   }

   if (fUseSession && !fIsGNNComponent) {
      fGC += "};   // end of Session\n";
   }
}

void RModel::GenerateGPU_ALPAKA(std::underlying_type_t<Options> options, int batchSize, bool verbose) {
   fProfile = static_cast<bool>(options & static_cast<std::underlying_type_t<Options>>(Options::kProfile));
   fVerbose = verbose;
   fBatchSize = batchSize;

   if (fProfile)
      RModelProfilerGPU::AddNeededStdLibs(*this);

   if (static_cast<std::underlying_type_t<Options>>(Options::kKernelOnly) & options) {
      fKernelOnly = true;
      fUseSession = false;
      fUseWeightFile = false;
      fWeightFile = WeightFileType::None;
   }
   if (static_cast<std::underlying_type_t<Options>>(Options::kNoSession) & options) {
      fUseSession = false;
      fWeightFile = WeightFileType::None;
   }
   if (static_cast<std::underlying_type_t<Options>>(Options::kNoWeightFile) & options) {
      fUseWeightFile = false;
      fWeightFile = WeightFileType::None;
   }
   if (fUseWeightFile && !fUseSession) {
      throw std::runtime_error(
          "sofie: RModel::Generate: cannot use a separate weight file without generating a Session class");
   }

   if (static_cast<std::underlying_type_t<Options>>(Options::kGNN) & options ||
       static_cast<std::underlying_type_t<Options>>(Options::kGNNComponent) & options)
      throw std::runtime_error("SOFIE GPU does not yet supports GNN Inference.");

   if (static_cast<std::underlying_type_t<Options>>(Options::kLowRankFactorize) & options)
      fLowRankFactorize = true;

   Initialize(batchSize, verbose);

   fFusion = Fusion::Compute(*this);

   std::string hgname;
   if (!fIsSubGraph) {
      fGC.clear();
      GenerateHeaderInfo_GPU_ALPAKA(hgname);
   }

   if (fVerbose)
      std::cout << "generate Main session code - model  " << fName << std::endl;

   GenerateSessionCode_GPU_ALPAKA();

   if (!fIsSubGraph) {
      fGC += ("} //SOFIE_" + fName + "\n");
      fGC += "\n#endif  // " + hgname + "\n";
   }
}

void RModel::MoveInitializedTensorsToBuffers_ALPAKA(){
      for (auto &i : fInitializedTensors) {
         if (i.second.IsNotWritable())  continue;
         std::string tensor_name = "tensor_" + i.first;
         auto length = ConvertShapeToLength(i.second.shape());
         std::string slength = std::to_string(length);
         // Use the 3-argument createView(dev, container, extent) which calls std::data()
         // internally — works for both std::vector and raw C arrays.
         fGC += "     auto hostBuf_"+i.first+" = alpaka::createView(hostAcc, tensor_"+i.first+", " + slength + ");\n";
         fGC += "     alpaka::memcpy(queue, deviceBuf_"+i.first+", hostBuf_"+i.first+");\n";
   }
}

} // namespace SOFIE
