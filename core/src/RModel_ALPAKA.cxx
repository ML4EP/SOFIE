#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cxxabi.h>
#include <typeinfo>
#include <climits>
#include <limits>
#include <memory>
#include <sstream>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "SOFIE/RModel.hxx"
#include "SOFIE/operators/ROperator_ALPAKA.hxx"
#include "SOFIE/RModelFusion_ALPAKA.hxx"
#include "SOFIE/RModelProfilerGPU.hxx"
#include "SOFIE/utils/SOFIE_common.hxx"

namespace SOFIE {

// device buffer / view alias per tensor type; must match the 'using Buf* / ViewConst*'
// aliases emitted into the generated Session
static constexpr std::size_t kPoolAlignBytes = 256;

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

std::string RModel::DeclareDeviceTensor_GPU_ALPAKA(const std::string &name, ETensorType type,
                                                   const std::string &lengthExpr) {
   Utils::TraceGeneration(*this, "declaring device tensor " + name + " (" + ConvertTypeToString(type) + ", " + lengthExpr + " elements) " +
                   (fCustomMemory ? "as a region of the memory block" : "as an alpaka buffer"));
   if (!fCustomMemory) {
      if (fDebug)
         fAllocatedBuffers.push_back({name, type, lengthExpr});
      return AllocBufLine(name, type, lengthExpr);
   }
   fDeviceRegions.push_back({name, type, lengthExpr});
   return DeclareViewLine(name, type);
}

void RModel::CheckNoDeviceAllocation_GPU_ALPAKA() const {
   if (!fDebug)
      throw std::runtime_error("sofie: CheckNoDeviceAllocation_GPU_ALPAKA is only available for a model generated with Options::kDebug");
   if (!fCustomMemory)
      throw std::runtime_error("sofie: CheckNoDeviceAllocation_GPU_ALPAKA is only available for a model generated with Options::kCustomMemory");
   const std::string needle = "alpaka::allocBuf<";
   size_t pos = 0;
   while ((pos = fGC.find(needle, pos)) != std::string::npos) {
      size_t lineEnd = fGC.find('\n', pos);
      std::string line = fGC.substr(pos, lineEnd == std::string::npos ? std::string::npos : lineEnd - pos);
      if (line.find(", Idx>(devAcc") != std::string::npos)
         throw std::runtime_error("sofie: Options::kCustomMemory: the generated code of model " + fName +
                                  " allocates device memory with alpaka, which is not supported with this option: " + line);
      pos += needle.size();
   }
}

std::string RModel::GetDeviceRegionsReport_GPU_ALPAKA() const {
   if (!fDebug)
      throw std::runtime_error("sofie: GetDeviceRegionsReport_GPU_ALPAKA is only available for a model generated with Options::kDebug");
   std::stringstream out;
   out << "Device tensors of model " << fName << ", " << fAllocatedBuffers.size() << " allocated by alpaka and " << fDeviceRegions.size()
       << " regions of a memory block:\n";
   constexpr size_t align = kPoolAlignBytes;
   size_t requested = 0, reserved = 0;
   bool allKnown = true;
   std::vector<DeviceRegion> listed = fAllocatedBuffers;
   listed.insert(listed.end(), fDeviceRegions.begin(), fDeviceRegions.end());
   for (const auto &r : listed) {
      const size_t typeSize = GetTypeSize(r.type);
      char *end = nullptr;
      const unsigned long long length = std::strtoull(r.lengthExpr.c_str(), &end, 10);
      if (!r.lengthExpr.empty() && *end == '\0') {
         const size_t bytes = length * typeSize;
         const size_t padded = (bytes + align - 1) / align * align;
         out << "  " << r.name << ": " << bytes << " bytes, " << padded << " reserved\n";
         requested += bytes;
         reserved += padded;
      } else {
         out << "  " << r.name << ": (" << r.lengthExpr << ") * " << typeSize << " bytes, known at construction\n";
         allKnown = false;
      }
   }
   out << (allKnown ? "total: " : "total of the regions with a constant size: ") << requested << " bytes, " << reserved
       << " reserved, padding " << reserved - requested << " bytes\n";
   return out.str();
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
         fGC += DeclareDeviceTensor_GPU_ALPAKA(i.first, i.second.type(), std::to_string(length));

   }
}

void RModel::GeneratePersistentTensorInfo_GPU_ALPAKA()
{
   std::set<std::string> persistentTensors;

   for (size_t id = 0; id < fOperators.size(); ++id) {
      if (fFusion.skip.count(id)) continue;

      for (const auto &name : AlpakaOp(*fOperators[id]).GetPersistentTensorNames_GPU_ALPAKA())
         persistentTensors.insert(name);
   }

   if (persistentTensors.empty())
      return;

   fGC += "\n// persistent state tensors\n";

   for (const auto &name : persistentTensors) {
      const ETensorType type = GetTensorType(name);
      const size_t length = ConvertShapeToLength(GetTensorShape(name));

      fGC += DeclareDeviceTensor_GPU_ALPAKA(name, type, std::to_string(length));
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

   constexpr std::size_t kPoolAlign = kPoolAlignBytes;
   auto alignUp = [](std::size_t v, std::size_t a) -> std::size_t { return (v + a - 1) & ~(a - 1); };

   // --- Unified pool: lifetime-aware packing of all pooled intermediates ---
   if (!fUnifiedPoolTensorNames.empty() || fCustomMemory) {
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
         Utils::TraceGeneration(*this, "pooled tensor " + tname + ": (" + sizeExpr + ") bytes, alive from operator " + std::to_string(producer) +
                         " to operator " + std::to_string(lastUse));
      }

      if (fCustomMemory) {
         AddNeededHelperFunction("CustomMemory");
         if (fDebug)
            AddNeededHelperFunction("DebugMemory");
         out << "\n" << "// --- Device memory block: planned by SOFIE, allocated and assigned through the memory provider ---\n";
         out << "{\n";
         out << "constexpr std::size_t kPoolAlign = " << kPoolAlignBytes << ";\n";
         out << "auto alignUp = [](std::size_t v, std::size_t a) -> std::size_t {\n";
         out << "return (v + a - 1) & ~(a - 1);\n";
         out << "};\n";
         out << "std::size_t staticBytes = 0;\n";
         for (size_t i = 0; i < fDeviceRegions.size(); ++i) {
            auto &r = fDeviceRegions[i];
            out << "const std::size_t regionOffset_" << i << " = staticBytes;\n";
            out << "const std::size_t regionBytes_" << i << " = alignUp((" << r.lengthExpr << ") * "
                << GetTypeSize(r.type) << ", kPoolAlign);\n";
            out << "staticBytes += regionBytes_" << i << ";\n";
         }
         out << "std::vector<TensorLifeInfo> poolInfos;\n";
         out << "poolInfos.reserve(" << entries.size() << ");\n";
         for (auto &e : entries) {
            int end = (int)(std::max(e.lastUseOp, e.producerOp) + 1);
            out << "poolInfos.push_back({" << e.producerOp << ", " << end << ", alignUp(" << e.sizeExpr
                << ", kPoolAlign)});\n";
         }
         out << "auto poolResult = OrganizeMemory(poolInfos);\n";
         out << "fPoolTotalFreeMemory = poolResult.total_free_bytes;\n";
         out << "fPoolLargestFreeBlock = poolResult.largest_free_block;\n";
         out << "std::size_t totalBytes = staticBytes + poolResult.total_bytes;\n";
         out << Utils::DebugLogCode(*this, "\"planned device memory block: \" + std::to_string(totalBytes) + \" bytes, \" + std::to_string(staticBytes) + \" bytes in "
                             + std::to_string(fDeviceRegions.size()) + " regions, \" + std::to_string(poolResult.total_bytes) + \" bytes for "
                             + std::to_string(entries.size()) + " pooled tensors, alignment " + std::to_string(kPoolAlignBytes) + "\"");
         out << "auto* blockBase = fPoolBlock.Allocate(sofieMemory, totalBytes);\n";
         out << "fUnifiedPoolSize = totalBytes;\n";
         if (fDebug) {
            out << Utils::DebugLogCode(*this, "\"memory block allocated through the provider: \" + std::to_string(totalBytes) + \" bytes at address \" + std::to_string(reinterpret_cast<std::uintptr_t>(blockBase))");
            out << "if (blockBase && reinterpret_cast<std::uintptr_t>(blockBase) % " << kPoolAlignBytes << " != 0)\n";
            out << Utils::DebugLogCode(*this, "\"warning: the memory block is not aligned to " + std::to_string(kPoolAlignBytes) + " bytes, as a cudaMalloc allocation is\"");
         }
         out << "\n";
         for (size_t i = 0; i < fDeviceRegions.size(); ++i) {
            auto &r = fDeviceRegions[i];
            const std::string typeName = ConvertOutputTypeToString(r.type);
            const std::string place = "PlaceTensor(sofieMemory, blockBase, regionOffset_" + std::to_string(i) + ", regionBytes_" +
                                      std::to_string(i) + ")";
            const std::string view = "deviceBuf_" + r.name + " = " + GetViewType(r.type) + "{reinterpret_cast<" + typeName +
                                     "*>(" + (fDebug ? "regionPtr" : place) + "), devAcc, Ext1D::all(Idx{" + r.lengthExpr + "})};\n";
            if (fDebug) {
               out << "{\nvoid* regionPtr = " << place << ";\n" << view;
               out << "fMemoryRegions.push_back({\"" << r.name << "\", regionPtr, regionOffset_" << i << ", (" << r.lengthExpr
                   << ") * " << GetTypeSize(r.type) << ", regionBytes_" << i << ", 0, std::numeric_limits<int>::max(), \"block\"});\n";
               out << Utils::DebugLogCode(*this, "\"tensor " + r.name + " placed at offset \" + std::to_string(regionOffset_" + std::to_string(i) +
                                   ") + \", \" + std::to_string(fMemoryRegions.back().requestedBytes) + \" bytes requested, \" + std::to_string(regionBytes_" +
                                   std::to_string(i) + ") + \" reserved\"");
               out << "}\n";
            } else {
               out << view;
            }
         }
         for (size_t i = 0; i < entries.size(); ++i) {
            auto &e = entries[i];
            const std::string typeName = ConvertOutputTypeToString(e.type);
            const std::string offset = "staticBytes + poolResult.offsets[" + std::to_string(i) + "]";
            const std::string reserved = "alignUp(" + e.sizeExpr + ", kPoolAlign)";
            const std::string place = "PlaceTensor(sofieMemory, blockBase, " + offset + ", " + reserved + ")";
            const std::string view = "deviceBuf_" + e.name + " = " + GetViewType(e.type) + "{reinterpret_cast<" + typeName +
                                     "*>(" + (fDebug ? "poolPtr" : place) + "), devAcc, Ext1D::all(Idx{" + e.lengthExpr + "})};\n";
            if (fDebug) {
               const int end = (int)(std::max(e.lastUseOp, e.producerOp) + 1);
               out << "{\nvoid* poolPtr = " << place << ";\n" << view;
               out << "fMemoryRegions.push_back({\"" << e.name << "\", poolPtr, " << offset << ", " << e.sizeExpr << ", " << reserved
                   << ", " << e.producerOp << ", " << end << ", \"block\"});\n";
               out << Utils::DebugLogCode(*this, "\"tensor " + e.name + " placed in the pool at offset \" + std::to_string(" + offset +
                                   ") + \", \" + std::to_string(fMemoryRegions.back().requestedBytes) + \" bytes requested, alive from operator " +
                                   std::to_string(e.producerOp) + " to " + std::to_string(end) + "\"");
               out << "}\n";
            } else {
               out << view;
            }
         }
         if (fDebug)
            out << "ValidateMemoryRegions(fMemoryRegions, kPoolAlign);\n";
         out << "}\n";
      } else if (!entries.empty()) {
         std::vector<size_t> constIdx, runtimeIdx;
         for (size_t i = 0; i < entries.size(); ++i)
            (entries[i].isConst ? constIdx : runtimeIdx).push_back(i);

         std::vector<size_t> constOffsets;
         std::size_t constTotalBytes = 0;
         std::size_t constFreeBytes = 0;
         std::size_t constLargestFree = 0;
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
            constFreeBytes = result.total_free_bytes;
            constLargestFree = result.largest_free_block;
         }

         out << "\n" << "// --- Unified tensor pool: lifetime-aware packing ---\n";
         out << "{\n";
         out << "constexpr std::size_t kConstPoolBytes = " << constTotalBytes << ";\n";
         out << "fPoolTotalFreeMemory = " << constFreeBytes << ";\n";
         out << "fPoolLargestFreeBlock = " << constLargestFree << ";\n";

         // Tensors whose size depends on a shape parameter only known once
         // the constructor runs are packed here, at construction time, with
         // the same coalescing planner used for the compile-time-constant
         // tensors above, appended after the constant region in the pool.
         if (!runtimeIdx.empty()) {
            // TensorLifeInfo and OrganizeMemory are emitted in the generated header
            AddNeededHelperFunction("DynamicMemory");
            out << "constexpr std::size_t kPoolAlign = " << kPoolAlignBytes << ";\n";
            out << "auto alignUp = [](std::size_t v, std::size_t a) -> std::size_t {\n";
            out << "return (v + a - 1) & ~(a - 1);\n";
            out << "};\n";
            out << "std::vector<TensorLifeInfo> runtimePoolInfos;\n";
            out << "runtimePoolInfos.reserve(" << runtimeIdx.size() << ");\n";
            for (size_t idx : runtimeIdx) {
               auto &e = entries[idx];
               int end = (int)(std::max(e.lastUseOp, e.producerOp) + 1);
               out << "runtimePoolInfos.push_back({" << e.producerOp << ", " << end
                   << ", alignUp(" << e.sizeExpr << ", kPoolAlign)});\n";
            }
            out << "auto runtimePoolResult = OrganizeMemory(runtimePoolInfos);\n";
            out << "std::size_t poolTotalBytes = kConstPoolBytes + runtimePoolResult.total_bytes;\n";
            out << "fPoolTotalFreeMemory += runtimePoolResult.total_free_bytes;\n";
            out << "fPoolLargestFreeBlock = std::max(fPoolLargestFreeBlock, runtimePoolResult.largest_free_block);\n";
         } else {
            out << "std::size_t poolTotalBytes = kConstPoolBytes;\n";
         }

         out << "\n" << "// Allocate unified pool buffer\n";
         out << "if (poolTotalBytes > 0)\n";
         out << "fUnifiedPool = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{poolTotalBytes}));\n";
         out << "auto* poolBase = (poolTotalBytes > 0) ? alpaka::getPtrNative(fUnifiedPool) : nullptr;\n";
         out << "fUnifiedPoolSize = poolTotalBytes;\n";
         out << Utils::DebugLogCode(*this, "\"allocated the unified pool of the intermediate tensors: \" + std::to_string(poolTotalBytes) + \" bytes for "
                             + std::to_string(entries.size()) + " tensors\"");
         out << "\n";

         out << "// Assign tensor views into unified pool\n";
         for (size_t i = 0; i < constIdx.size(); ++i) {
            auto &e = entries[constIdx[i]];
            std::string typeName = ConvertOutputTypeToString(e.type);
            std::string viewType = GetViewType(e.type);
            out << "deviceBuf_" << e.name << " = " << viewType
                << "{reinterpret_cast<" << typeName << "*>(poolBase + " << constOffsets[i]
                << "), devAcc, Ext1D::all(Idx{" << e.lengthExpr << "})};\n";
            out << Utils::DebugLogCode(*this, "\"tensor " + e.name + " placed in the pool at offset " + std::to_string(constOffsets[i]) + "\"");
            if (fDebug && fUseSession) {
               out << "fMemoryRegions.push_back({\"" << e.name << "\", poolBase + " << constOffsets[i] << ", " << constOffsets[i] << ", " << e.sizeExpr
                   << ", " << e.constBytes << ", " << e.producerOp << ", " << (int)(std::max(e.lastUseOp, e.producerOp) + 1) << ", \"pool\"});\n";
            }
         }
         for (size_t i = 0; i < runtimeIdx.size(); ++i) {
            auto &e = entries[runtimeIdx[i]];
            std::string typeName = ConvertOutputTypeToString(e.type);
            std::string viewType = GetViewType(e.type);
            out << "deviceBuf_" << e.name << " = " << viewType
                << "{reinterpret_cast<" << typeName << "*>(poolBase + kConstPoolBytes + runtimePoolResult.offsets["
                << i << "]), devAcc, Ext1D::all(Idx{" << e.lengthExpr << "})};\n";
            out << Utils::DebugLogCode(*this, "\"tensor " + e.name + " placed in the pool at offset \" + std::to_string(kConstPoolBytes + runtimePoolResult.offsets[" +
                                std::to_string(i) + "])");
            if (fDebug && fUseSession) {
               const std::string offset = "kConstPoolBytes + runtimePoolResult.offsets[" + std::to_string(i) + "]";
               out << "fMemoryRegions.push_back({\"" << e.name << "\", poolBase + " << offset << ", " << offset << ", " << e.sizeExpr
                   << ", alignUp(" << e.sizeExpr << ", kPoolAlign), " << e.producerOp << ", " << (int)(std::max(e.lastUseOp, e.producerOp) + 1) << ", \"pool\"});\n";
            }
         }

         out << "}\n";
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
            continue; // not a tensor with a trackable shape
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

      const bool inArena = std::any_of(fDeviceRegions.begin(), fDeviceRegions.end(),
                                       [&](const DeviceRegion &r) { return r.name == storageName; });
      if (IsPooledIntermediate(storageName) || inArena) {
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

   // A dynamic-shaped tensor that isn't in the unified pool was declared with a placeholder size of 1 element.
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
         // Shapes that depend on values only known at inference time aren't expressible from _infer_impl's
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
         if (fArenaDynamicOutputs.count(name)) {
            code += "deviceBuf_" + name + " = " + GetViewType(it->second.type) + "{alpaka::getPtrNative(deviceBuf_" + name +
                    "), devAcc, Ext1D::all(Idx{static_cast<Idx>(" + lengthExpr + ")})};\n";
            continue;
         }
         code += "deviceBuf_" + name + " = alpaka::allocBuf<" + ConvertOutputTypeToString(it->second.type) +
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

   fGC += "// run the model on views of the input tensors and leave the results in the output buffers without waiting\n";
   fGC += "void _infer_impl(";
   fGC += GenerateImplSignature_GPU_ALPAKA();
   fGC += "){\n";
   // device buffers were sized in the ctor from its shape params; refuse larger ones
   // (same check and message as the CPU session)
   ForEachInferArg_GPU_ALPAKA(
      [&](const std::string &p) {
         fGC += "if (" + p + " > " + GetMemberNameForDimShape(p) + ") {\n";
         fGC += "throw std::runtime_error(\"sofie: dynamic input tensor shape parameter " + p +
                " exceeds the initialized maximum allowed shape.\");\n";
         fGC += "}\n";
      },
      [](const std::string &) {});

   // GPU profiling: _infer_impl is a member of Session, so fProfilingResults
   // is directly accessible without any alias.
   if (fProfile) {
      fGC += RModelProfilerGPU::GenerateBeginInferCode();
   }

   fGC += Utils::DebugLogCode(*this, Utils::DebugLiteral("running the inference of model " + fName + ": enqueueing the operators"));

   for (size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx) {
      if (fVerbose)
         std::cout << "Generating code for operator .... " << op_idx << std::endl;

      const std::string opLabel = "operator " + std::to_string(op_idx) + " (" + toString(fOperators[op_idx]->GetKind()) + ")";

      if (fFusion.skip.count(op_idx)) {
         Utils::TraceGeneration(*this, opLabel + ": fused into another kernel, no launch code of its own");
         continue;
      }

      if (const auto *kernelGroup = fFusion.FindKernel(op_idx)) {
         if (kernelGroup->launchOpIndex == op_idx) {
            for (const auto &branch : kernelGroup->branches)
               fGC += GenerateDynamicOutputReallocForNames(branch.outputTensors);
            fGC += Utils::LaunchCodeDebug_GPU_ALPAKA(*this, "fused kernel group " + kernelGroup->suffix(), Fusion::Launch(*this, *kernelGroup));
         }

         continue;
      }

      if (const auto *eltwiseGroup = fFusion.FindEltwise(op_idx)) {
         if (eltwiseGroup->launchOpIndex == op_idx) {
            fGC += GenerateDynamicOutputReallocForNames(eltwiseGroup->outputTensors);
            fGC += Utils::LaunchCodeDebug_GPU_ALPAKA(*this, "fused elementwise group " + eltwiseGroup->suffix(), Fusion::Launch(*this, *eltwiseGroup));
         }
      } else {
         auto opDynParamNames = GetOperatorKernelParams(op_idx, dynParamNames);
         fGC += GenerateDynamicOutputRealloc(op_idx);
         if (fProfile) {
            fGC += Utils::LaunchCodeDebug_GPU_ALPAKA(*this, opLabel, RModelProfilerGPU::GenerateOperatorCode(*fOperators[op_idx], op_idx, opDynParamNames));
         } else {
            fGC += Utils::LaunchCodeDebug_GPU_ALPAKA(*this, opLabel, AlpakaOp(*fOperators[op_idx]).Generate_GPU_ALPAKA(std::to_string(op_idx), opDynParamNames));
         }
      }
   }

   fGC += Utils::DebugLogCode(*this, Utils::DebugLiteral("all the operators of model " + fName + " are enqueued"));

   if (fProfile) {
      fGC += RModelProfilerGPU::GenerateEndInferCode();
   }

   fGC += "}\n\n";


   std::string spanDynDecl;
   for (auto &p : dynParamNames)
      spanDynDecl += ", size_t " + p;

   fGC += "// run the model on views of the input tensors, copy the results into the output views and wait for completion\n";
   fGC += "void infer(std::span<ViewConstF1D const> inputs, std::span<ViewF1D> outputs" + spanDynDecl + "){\n";

   {
      // each symbol goes just before the first input whose shape introduces it
      fGC += "_infer_impl(";
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
      fGC += "alpaka::memcpy(queue, outputs[" + std::to_string(i) + "], " + GetOutputBufferName(tensorName) + ");\n";
   }
   fGC += "alpaka::wait(queue);\n";
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

   fGC += "// run the model on the input buffers and return the output buffers once all the work has finished\n";
   fGC += returnType + " infer(";
   fGC += GenerateInferSignature_GPU_ALPAKA();
   fGC += "){\n";

   std::vector<std::string> typedImplArgs;
   ForEachInferArg_GPU_ALPAKA(
      [&](const std::string &p) { typedImplArgs.push_back(p); },
      [&](const std::string &name) {
         std::string viewType = GetViewConstType(GetTensorType(name));
         fGC += viewType + " const view_" + name +
                "{alpaka::getPtrNative(deviceBuf_" + name + "), devAcc, alpaka::getExtents(deviceBuf_" + name + ")};\n";
         typedImplArgs.push_back("view_" + name);
      });

   // Helper lambda: emit the _infer_impl call and return statement
   auto emitImplCallAndReturn = [&]() {
      fGC += "_infer_impl(";
      for (size_t i = 0; i < typedImplArgs.size(); i++) {
         if (i > 0) fGC += ", ";
         fGC += typedImplArgs[i];
      }
      fGC += ");\n";

      fGC += "return ";
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
   fGC += "_infer_impl(";
   for (size_t i = 0; i < typedImplArgs.size(); i++) {
      if (i > 0) fGC += ", ";
      fGC += typedImplArgs[i];
   }
   fGC += ");\n";
   fGC += "alpaka::wait(queue);\n";
   fGC += "return ";
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
   fGC += "// same as infer, but returns without waiting: call alpaka::wait on the queue before reading the results\n";
   fGC += returnType + " infer_nosync(";
   fGC += GenerateInferSignature_GPU_ALPAKA();
   fGC += "){\n";

   // Re-emit the view declarations for the nosync overload
   typedImplArgs.clear();
   ForEachInferArg_GPU_ALPAKA(
      [&](const std::string &p) { typedImplArgs.push_back(p); },
      [&](const std::string &name) {
         std::string viewType = GetViewConstType(GetTensorType(name));
         fGC += viewType + " const view_" + name +
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

void RModel::CollectOperatorsWithoutAlpaka(std::vector<MissingGPUOperator> &missing) const
{
   for (size_t id = 0; id < fOperators.size(); id++) {
      const ROperator &op = *fOperators[id];
      if (const auto *alpakaOp = GetAlpaka(op)) {
         for (const auto &graph : alpakaOp->GetSubGraphModels())
            graph->CollectOperatorsWithoutAlpaka(missing);
         continue;
      }
      int status = 0;
      char *demangled = abi::__cxa_demangle(typeid(op).name(), nullptr, nullptr, &status);
      const std::string type = status == 0 && demangled ? demangled : typeid(op).name();
      std::free(demangled);
      missing.push_back({fName, id, type});
   }
}

void RModel::CheckAlpakaOperators() const
{
   std::vector<MissingGPUOperator> missing;
   CollectOperatorsWithoutAlpaka(missing);
   if (!missing.empty())
      throw std::runtime_error("sofie: GPU (alpaka) generation of model " + missing[0].graph + ": operator " +
                               std::to_string(missing[0].index) + " (" + missing[0].type +
                               ") has no alpaka implementation");
}

bool RModel::CheckGPUOperators() const
{
   std::vector<MissingGPUOperator> missing;
   CollectOperatorsWithoutAlpaka(missing);
   if (!missing.empty()) {
      std::cout << "List of operators without a GPU (alpaka) implementation for model " << fName << std::endl;
      for (const auto &m : missing)
         std::cout << m.type << "  operator " << m.index << " of graph " << m.graph << std::endl;
      return false;
   }
   std::cout << "All operators in the model have a GPU (alpaka) implementation!\n";
   return true;
}

void RModel::GenerateSessionCode_GPU_ALPAKA() {
   CheckAlpakaOperators();

   // the model's dynamic shape parameters in infer-argument order
   std::vector<std::string> dynParamNames;
   ForEachInferArg_GPU_ALPAKA([&](const std::string &p) { dynParamNames.push_back(p); },
                              [](const std::string &) {});

   // kernels shared by all the operators of a kind are generated once; the variant reading a strided input
   // is a different kernel
   std::set<std::pair<SOFIE::OperatorKind, bool>> registered_operators;
   auto kernelKey = [this](size_t id) {
      return std::make_pair(fOperators[id]->GetKind(), fOperators[id]->HasStridedInput());
   };

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

   for (auto &op : fOperators)
      AlpakaOp(*op).SetCustomMemory(fCustomMemory);

   // the sessions of the sub-graphs are generated first, each one in its own
   // namespace to keep their kernels separate. They share the queue of this session
   for (auto &graph : fSubGraphs) {
      if (fKernelOnly)
         throw std::runtime_error("sofie: Options::kKernelOnly is not supported for models with subgraphs");
      if (fVerbose)
         std::cout << "generate session code for subgraph " << graph->fName << std::endl;
      Utils::TraceGeneration(*this, "generating the session of the sub-graph " + graph->fName);
      graph->fVerbose = fVerbose;
      graph->fBatchSize = fBatchSize;
      graph->fFusion = Fusion::Compute(*graph);
      graph->GenerateSessionCode_GPU_ALPAKA();
      fGC += "\nnamespace SubGraph_" + graph->fName + " {\n" + graph->fGC + "\n} // namespace SubGraph_" +
             graph->fName + "\n";
      fNeededHelperFunctions.insert(graph->fNeededHelperFunctions.begin(), graph->fNeededHelperFunctions.end());
   }

   fGC += "\n//--- ALPAKA Kernels\n";
   for (size_t id = 0; id < fOperators.size(); id++) {
      if(fOperators[id]->GetKind() == OperatorKind::GEMM || fOperators[id]->GetKind() == OperatorKind::CONV) {
         OpNeedsBlas = true;
      }

      const std::string opLabel = "operator " + std::to_string(id) + " (" + toString(fOperators[id]->GetKind()) + ")";
      if (const auto *kernelGroup = fFusion.FindKernel(id)) {
         if (kernelGroup->branches.front().opIndices.front() == id) {
            Utils::TraceGeneration(*this, opLabel + ": generating the fused kernel group " + kernelGroup->suffix());
            fGC += Fusion::Kernel(*this, *kernelGroup);
         }
      } else {
         if (const auto *eltwiseGroup = fFusion.FindEltwise(id)) {
            if (eltwiseGroup->opIndices[0] == id) {
               Utils::TraceGeneration(*this, opLabel + ": generating the fused elementwise group " + eltwiseGroup->suffix());
               fGC += Fusion::Kernel(*this, *eltwiseGroup);
            }
         } else {
            auto idDynParamNames = GetOperatorKernelParams(id, dynParamNames);
            if (single_initialized_operators.find(fOperators[id]->GetKind()) != single_initialized_operators.end()) {
               if (registered_operators.find(kernelKey(id)) == registered_operators.end()) {
                  if (fVerbose)
                     std::cout << "Generating ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
                  const std::string kernelCode = AlpakaOp(*fOperators[id]).Generate_GPU_Kernel_ALPAKA(std::to_string(id), idDynParamNames);
                  Utils::TraceGeneration(*this, opLabel + ": generating its kernel (" + std::to_string(kernelCode.size()) + " characters), shared by the operators of the same kind");
                  fGC += kernelCode;
                  registered_operators.insert(kernelKey(id));
               } else {
                  Utils::TraceGeneration(*this, opLabel + ": uses the kernel already generated for its kind");
               }
            } else {
               if (fVerbose)
                  std::cout << "Generating ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
               const std::string kernelCode = AlpakaOp(*fOperators[id]).Generate_GPU_Kernel_ALPAKA(std::to_string(id), idDynParamNames);
               Utils::TraceGeneration(*this, opLabel + ": generating its kernel (" + std::to_string(kernelCode.size()) + " characters)");
               fGC += kernelCode;
            }
         }
      }
   }



   if (fKernelOnly)
      return;

   // define the Session struct (for GNN this is generated in RModel_GNN)
  fGC += "\n\n// --- Session: owns the device buffers and kernels and provides the inference methods ---\n";
  fGC += "template <typename tagAcc>\n";
   if (fUseSession) {
      fGC += "struct Session {\n\n";
   }

   // define host and device accelerators
    fGC += "// index, dimension, accelerator, device and queue types\n";
    fGC += "using Idx = std::size_t;\n";
    fGC += "using Dim = alpaka::DimInt<1>;\n";
    fGC += "using Acc = alpaka::TagToAcc<tagAcc, Dim, Idx>;\n";
    fGC += "using DevAcc = alpaka::Dev<Acc>;\n\n";
    fGC += "using QueueProperty = alpaka::NonBlocking;\n";
    fGC += "using QueueAcc = alpaka::Queue<Acc, QueueProperty>;\n\n";
    fGC += "// owning device buffer types\n";
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

    fGC += "\n// platform, device and queue of the accelerator and of the host\n";
    fGC += "alpaka::Platform<Acc> const platform{};\n";
    if (fExternalStream) {
       fGC += "QueueAcc queue;\n";
       fGC += "DevAcc devAcc = alpaka::getDev(queue);\n";
    } else {
       fGC += "DevAcc devAcc = alpaka::getDevByIdx(platform, 0);\n";
    }
    fGC += "alpaka::PlatformCpu platformHost{};\n";
    fGC += "alpaka::DevCpu hostAcc = alpaka::getDevByIdx(platformHost, 0);\n";
    if (!fExternalStream)
       fGC += "QueueAcc queue{devAcc};\n";
    fGC += "// default number of threads per block\n";
    fGC += "Idx threadsPerBlock = 256;\n";
    fGC += "\n// vector types for extents and work divisions\n";
    fGC += "using Ext1D = alpaka::Vec<Dim, Idx>;\n";
    fGC += "using Vec = alpaka::Vec<Dim, Idx>;\n";
    if (OpNeedsBlas) {
         fGC += "\n\n// BLAS declarations\n";
         fGC += "sofieBLAS<tagAcc> blas{queue};\n";
    }

   /// Allocate memory efficiently
   GenerateInitializedTensorInfo_GPU_ALPAKA();
   GeneratePersistentTensorInfo_GPU_ALPAKA();

   std::unordered_map<std::string, std::string> outputCapacities;
   if (fCustomMemory) {
      bool scratchHeader = false;
      for (size_t id = 0; id < fOperators.size(); ++id) {
         if (fFusion.skip.count(id)) continue;
         for (const auto &scratch : AlpakaOp(*fOperators[id]).GetDeviceScratch_GPU_ALPAKA()) {
            if (!scratchHeader) {
               fGC += "\n// scratch memory of the operators\n";
               scratchHeader = true;
            }
            fGC += DeclareDeviceTensor_GPU_ALPAKA(scratch.name, scratch.type, scratch.lengthExpr);
         }
         for (const auto &cap : AlpakaOp(*fOperators[id]).GetOutputCapacities_GPU_ALPAKA())
            outputCapacities[cap.first] = cap.second;
      }
   }

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
         for (const auto &name : AlpakaOp(*op).GetPersistentTensorNames_GPU_ALPAKA())
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
            tensorDecls += DeclareDeviceTensor_GPU_ALPAKA(i.first, i.second.type, std::to_string(length));
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
         if (fCustomMemory && isOutput && !runtimeShape) {
            tensorDecls += DeclareDeviceTensor_GPU_ALPAKA(i.first, i.second.type, ConvertDimShapeToLength(i.second.shape));
            fArenaDynamicOutputs.insert(i.first);
         } else if (fCustomMemory && runtimeShape && outputCapacities.count(i.first)) {
            tensorDecls += DeclareDeviceTensor_GPU_ALPAKA(i.first, i.second.type, outputCapacities[i.first]);
            fArenaDynamicOutputs.insert(i.first);
         } else if (runtimeShape || isOutput) {
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
            fGC += DeclareDeviceTensor_GPU_ALPAKA(i.first, ETensorType::INT64, std::to_string(len));
         }
      }

      if (fCustomMemory) {
         fGC += "\n//--- device memory block (allocated through the memory provider in the constructor)\n";
         fGC += "PoolBlock fPoolBlock;\n";
      } else if (!unifiedPoolTensors.empty()) {
         fGC += "\n//--- unified memory pool buffer (allocated in constructor)\n";
         fGC += "BufUI81D fUnifiedPool = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
      }
      if (fDebug && fUseSession) {
         fGC += "\n//--- debug: the memory regions of the session and the checks on their placement\n";
         fGC += "std::vector<MemoryRegionInfo> fMemoryRegions;\n";
         fGC += "std::vector<MemoryRegionInfo> const &GetMemoryRegions() const { return fMemoryRegions; }\n";
         fGC += "std::size_t GetPaddingBytes() const {\n";
         fGC += "std::size_t padding = 0;\n";
         fGC += "for (auto const &r : fMemoryRegions)\n";
         fGC += "padding += r.reservedBytes - r.requestedBytes;\n";
         fGC += "return padding;\n";
         fGC += "}\n";
         fGC += "std::string GetMemoryReport() const { return FormatMemoryReport(fMemoryRegions, fUnifiedPoolSize); }\n";
         fGC += "void ValidateMemory() const { ValidateMemoryRegions(fMemoryRegions, " + std::to_string(kPoolAlignBytes) + "); }\n";
      }
      fGC += "\n//--- memory pool statistics\n";
      fGC += "std::size_t fUnifiedPoolSize = 0;\n";
      fGC += "std::size_t fPoolTotalFreeMemory = 0;\n";
      fGC += "std::size_t fPoolLargestFreeBlock = 0;\n";
      fGC += "std::size_t GetIntermediateMemoryPoolSize() const { return fUnifiedPoolSize; }\n";
      fGC += "std::size_t GetLargestFreeBlock() const { return fPoolLargestFreeBlock; }\n";
      fGC += "std::size_t GetTotalFreeMemory() const { return fPoolTotalFreeMemory; }\n";
      fGC += "double GetFragmentation() const { return fPoolTotalFreeMemory == 0 ? 0.0 : 1.0 - static_cast<double>(fPoolLargestFreeBlock) / static_cast<double>(fPoolTotalFreeMemory); }\n";

      fUnifiedPoolTensorNames = std::move(unifiedPoolTensors);
   }

   GenerateOperatorDeclarations();

   // sessions of the sub-graphs, running on the queue of this session
   if (!fSubGraphs.empty())
      fGC += "\n//   subgraph sessions\n";
   if (fCustomMemory && !fSubGraphs.empty()) {
      AddNeededHelperFunction("CustomMemory");
      fGC += "MemoryProvider fMemory;\n";
   }
   for (auto &graph : fSubGraphs)
      fGC += "SubGraph_" + graph->fName + "::Session<tagAcc> fSession_" + graph->fName +
             (fCustomMemory ? "{queue, fMemory};\n" : "{queue};\n");

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

         if (fCustomMemory)
            GenerateDynamicTensorInfo_GPU_ALPAKA();
         GenerateTemporaryInitializedTensorContainers_GPU_ALPAKA();
         if (fUseWeightFile) {
            fGC += "\n//--- reading weights from file\n";
            fGC += Utils::DebugLogCode(*this, "\"reading the weights from file \" + filename");
            ReadInitializedTensorsFromFile();
            fGC += "\n";
         }
         MoveInitializedTensorsToBuffers_ALPAKA();
         if (!fCustomMemory)
            GenerateDynamicTensorInfo_GPU_ALPAKA();
         for (auto &op : fOperators) {
            for (const auto &outName : op->GetOpOutputTensors()) {
               if (std::find(fUnifiedPoolTensorNames.begin(), fUnifiedPoolTensorNames.end(), outName) != fUnifiedPoolTensorNames.end() ||
                   fArenaDynamicOutputs.count(std::string(outName)))
                  AlpakaOp(*op).MarkOutputAsPooled(outName);
            }
         }
         for (size_t id = 0; id < fOperators.size(); id++) {
            if (fFusion.skip.count(id)) continue;
            const std::string initCode = AlpakaOp(*fOperators[id]).GenerateInitCode_GPU_ALPAKA();
            const std::string opLabel = "operator " + std::to_string(id) + " (" + toString(fOperators[id]->GetKind()) + ")";
            Utils::TraceGeneration(*this, opLabel + ": " + (initCode.empty() ? std::string("no init code") : "init code of " + std::to_string(initCode.size()) + " characters"));
            if (!initCode.empty())
               fGC += Utils::DebugLogCode(*this, Utils::DebugLiteral("initializing " + opLabel)) + initCode;
            if (fOperators[id]->GetKind() == OperatorKind::GEMM || fOperators[id]->GetKind() == OperatorKind::CONV) {
               for (auto &blasCfg : AlpakaOp(*fOperators[id]).GetBlasConfigs()) {
                  if (!blasCfg.empty()) {
                     fGC += Utils::DebugLogCode(*this, Utils::DebugLiteral("adding the BLAS configuration of " + opLabel));
                     fGC += "\nblas.addOperationConfig(" + blasCfg + ");\n";
                  }
               }
            }
         }
         fGC += "\nalpaka::wait(queue);\n";
         if (fDebug && !fCustomMemory)
            fGC += "ValidateMemory();\n";
         fGC += Utils::DebugLogCode(*this, Utils::DebugLiteral("the device is initialized: the session of model " + fName + " is constructed"));

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
            ctorBody += GetMemberNameForDimShape(p) + " = " + p + ";\n";
         if (fDebug) {
            ctorBody += Utils::DebugLogCode(*this, Utils::DebugLiteral("constructing the session of model " + fName));
            for (const auto &b : fAllocatedBuffers)
            {
               ctorBody += Utils::DebugLogCode(*this, "\"device buffer deviceBuf_" + b.name + " allocated by alpaka: \" + std::to_string((" + b.lengthExpr +
                                        ") * " + std::to_string(GetTypeSize(b.type)) + ") + \" bytes\"");
               const std::string bytes = "(" + b.lengthExpr + ") * " + std::to_string(GetTypeSize(b.type));
               ctorBody += "fMemoryRegions.push_back({\"" + b.name + "\", alpaka::getPtrNative(deviceBuf_" + b.name + "), 0, " + bytes + ", " +
                           bytes + ", 0, std::numeric_limits<int>::max(), \"alpaka\"});\n";
            }
            ctorBody += "@@SOFIE_KERNEL_OBJECTS_LOG@@\n";
         }
         ctorBody += fGC;
         fGC = savedGC;

         std::string ctorParams;
         for (auto &p : ctorParamNames)
            ctorParams += ",\nsize_t " + p + " = " + fShapeParams[p];
         if (fStridedInput)
            ctorParams += ",\nstd::vector<std::vector<size_t>> inputStrides = {}";

         /*
          * One Session member per shape parameter, the same members the CPU session declares.
          * The infer arguments are checked against them at the top of _infer_impl; a parameter
          * registered by an operator is checked by that operator.
          */
         if (!ctorParamNames.empty())
            fGC += "\n// maximum values of the dynamic shape parameters, set at construction\n";
         for (auto &p : ctorParamNames)
            fGC += "size_t " + GetMemberNameForDimShape(p) + ";\n";

         std::string initArgs;
         for (auto &p : ctorParamNames)
            initArgs += ", " + p;

         std::string initParams = fCustomMemory ? "MemoryProvider const &sofieMemory, std::string filename" : "std::string filename";
         for (auto &p : ctorParamNames)
            initParams += ", size_t " + p;

         // strides (in elements) of the input tensors; empty means contiguous input
         if (fStridedInput) {
            initArgs += ", inputStrides";
            initParams += ", std::vector<std::vector<size_t>> const &inputStrides";
            std::string strideCode = GenerateInputStrideInitCode();
            for (auto &graph : fSubGraphs)
               strideCode += "fSession_" + graph->fName + ".SetInputStrides(inputStrides);\n";
            ctorBody = strideCode + ctorBody;
            fGC += "\n//   strides of the input tensors (empty = contiguous)\n";
            for (const auto &name : fInputTensorNames)
               fGC += "std::vector<size_t> " + GetInputStrideMember(name) + ";\n";
         }

         // the session of a sub-graph is created by the session of the main model, which gives it the strides
         if (fStridedInput && fIsSubGraph) {
            fGC += "\nvoid SetInputStrides(std::vector<std::vector<size_t>> const &inputStrides) {\n";
            fGC += GenerateInputStrideInitCode();
            for (auto &graph : fSubGraphs)
               fGC += "fSession_" + graph->fName + ".SetInputStrides(inputStrides);\n";
            fGC += "}\n";
         }

         fGC += "\n// load the weights, allocate the device buffers and plan the memory pool\n";
         fGC += "void InitSession(" + initParams + ") {\n";
         fGC += ctorBody;
         fGC += "}\n\n";

         fGC += "public:\n";

         const std::string nameArg = fUseWeightFile ? "std::string filename = \"" + fileName + "\"" : "std::string filename = \"\"";
         const std::string memParam = fCustomMemory ? "MemoryProvider sofieMemory, " : "";
         const std::string initCall = std::string("InitSession(") + (fCustomMemory ? "sofieMemory, " : "") + "filename" + initArgs + ");\n";
         const std::string memInit = fCustomMemory && !fSubGraphs.empty() ? "fMemory(sofieMemory)" : "";
         auto initList = [&](std::vector<std::string> members) {
            if (OpNeedsBlas)
               members.push_back("blas(queue)");
            if (!memInit.empty())
               members.push_back(memInit);
            std::string list;
            for (auto &m : members)
               list += (list.empty() ? "\n: " : ", ") + m;
            return list;
         };

         if (fExternalStream && !fIsSubGraph) {
            fGC += "\n// native CUDA/HIP stream type of the accelerator queue\n";
            fGC += "using NativeStream = decltype(std::declval<QueueAcc const&>().getNativeHandle());\n\n";
            fGC += "// constructor wrapping a native CUDA/HIP stream as a non-owning queue: the stream must outlive the session\n";
            fGC += "// and belong to the current device\n";
            fGC += sessionName + "(NativeStream stream, " + memParam + nameArg + ctorParams + ")" + initList({"queue(stream)"}) + "\n{\n";
            fGC += initCall;
            fGC += "}\n\n";
         } else {
            if (!fExternalStream) {
               // (1) default-queue constructor
               fGC += "\n// constructor using the default queue\n";
               fGC += "\n" + sessionName + "(" + memParam + nameArg + ctorParams + ")" + initList({}) + "\n{\n";
               fGC += initCall;
               fGC += "}\n\n";
            }

            // (2) external-queue constructor
            fGC += "// constructor using an external queue\n";
            fGC += sessionName + "(QueueAcc& extQueue, " + memParam + nameArg + ctorParams + ")";
            if (fExternalStream)
               fGC += initList({"queue(extQueue)"});
            else
               fGC += initList({"devAcc(alpaka::getDev(extQueue))", "queue(extQueue)"});
            fGC += "\n{\n";
            fGC += initCall;
            fGC += "}\n\n";
         }

         if (fCustomMemory) {
            fGC += "// the pool block is returned to the memory provider once the queue has finished\n";
            fGC += "~" + sessionName + "() {\n";
            fGC += "alpaka::wait(queue);\n";
            fGC += Utils::DebugLogCode(*this, Utils::DebugLiteral("the queue has finished: the device memory block of model " + fName + " is released through the provider"));
            fGC += "}\n\n";
         }
      }
   }

   registered_operators.clear();

   const std::size_t kernelObjectsPos = fGC.size();
   for (size_t id = 0; id < fOperators.size(); id++) {
      // Same as the kernel-struct loop above: fused activation ops must still
      // declare their member variable even though
      // their Generate_GPU_ALPAKA call is skipped in the infer-body loop.

      if (const auto *kernelGroup = fFusion.FindKernel(id)) {
         if (kernelGroup->branches.front().opIndices.front() == id) {
            const std::string sfx = kernelGroup->suffix();
            fGC += "KernelFusionKernel" + sfx + " kernelFusionKernel" + sfx + ";\n";
         }
      } else {
         if (const auto *eltwiseGroup = fFusion.FindEltwise(id)) {
            if (eltwiseGroup->opIndices[0] == id) {
               const std::string sfx = eltwiseGroup->suffix();
               fGC += "FusedEltwiseKernel" + sfx + " fusedEltwiseKernel" + sfx + ";\n";
            }
         } else {
            if (single_initialized_operators.find(fOperators[id]->GetKind()) != single_initialized_operators.end()) {
               if (registered_operators.find(kernelKey(id)) == registered_operators.end()) {
                  if (fVerbose)
                     std::cout << "Declaring ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
                  fGC += AlpakaOp(*fOperators[id]).Generate_GPU_Kernel_Definitions_ALPAKA(std::to_string(id));
                  registered_operators.insert(kernelKey(id));
               }
            } else {
               if (fVerbose)
                  std::cout << "Declaring ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
               fGC += AlpakaOp(*fOperators[id]).Generate_GPU_Kernel_Definitions_ALPAKA(std::to_string(id));
            }
         }
      }
   }

   if (fGC.size() > kernelObjectsPos)
      fGC.insert(kernelObjectsPos, "\n// kernel objects launched by the inference methods\n");

   if (fDebug) {
      std::vector<std::string> kernelMembers;
      Utils::CollectKernelMembers_GPU_ALPAKA(fGC.substr(kernelObjectsPos), kernelMembers);
      std::string members;
      for (const auto &m : kernelMembers)
         members += (members.empty() ? "" : ", ") + m;
      Utils::TraceGeneration(*this, "kernel objects of the session: " + (members.empty() ? std::string("none") : members));
      const std::string marker = "@@SOFIE_KERNEL_OBJECTS_LOG@@\n";
      const size_t markerPos = fGC.find(marker);
      if (markerPos != std::string::npos)
         fGC.replace(markerPos, marker.size(), Utils::DebugLogCode(*this, Utils::DebugLiteral("kernel objects instantiated with the session: " + (members.empty() ? std::string("none") : members))));
   }

   GenerateOutput_GPU_ALPAKA();

   // Emit resetState() for recurrent/stateful operator buffers, and zero all
   // intermediate tensors (static- and dynamic-shape alike) to reset model state.
   // A dynamic tensor's device buffer is zeroed over its full allocated capacity.
   if (fUseSession) {
      fGC += "\n// zero the persistent state and the intermediate buffers\n";
      fGC += "void resetState(QueueAcc& queue) {\n";

      for (size_t id = 0; id < fOperators.size(); ++id) {
         if (fFusion.skip.count(id))
            continue;
         fGC += AlpakaOp(*fOperators[id]).GenerateResetStateCode_GPU_ALPAKA();
      }
      for (auto &i : fIntermediateTensorInfos) {
         // Alias tensors are zero-copy views declared as
         // locals inside _infer_impl, not Session members, and memsetting one would zero
         // the aliased tensor's storage out from under it anyway.
         if (fFusion.internal.count(i.first) || IsAliasTensor(i.first)) continue;
         fGC += "alpaka::memset(queue, deviceBuf_" + i.first + ", 0);\n";
      }
      for (auto &i : fDynamicTensorInfos) {
         if (fFusion.internal.count(i.first) || IsAliasTensor(i.first)) continue;
         fGC += "alpaka::memset(queue, deviceBuf_" + i.first + ", 0);\n";
      }
      fGC += "alpaka::wait(queue);\n";
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

   if (fDebug) {
      std::string helpers;
      for (const auto &h : fNeededHelperFunctions)
         helpers += (helpers.empty() ? "" : ", ") + h;
      Utils::TraceGeneration(*this, "session code generated (" + std::to_string(fGC.size()) + " characters), helper functions needed: " +
                      (helpers.empty() ? std::string("none") : helpers));
   }

   if (fCustomMemory && fDebug) {
      CheckNoDeviceAllocation_GPU_ALPAKA();
      if (fVerbose)
         std::cout << GetDeviceRegionsReport_GPU_ALPAKA();
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

   if (static_cast<std::underlying_type_t<Options>>(Options::kStridedInput) & options) {
      // without a Session (Options::kKernelOnly) the strides are not stored: the strided kernels take them as an
      // argument (sofie_strided_layout), and the header defines the layout and the offset function
      if (!fUseSession && !fKernelOnly)
         throw std::runtime_error("sofie: RModel::Generate: Options::kStridedInput requires a Session class "
                                  "(the strides are passed to its constructor)");
      fStridedInput = true;
   }

   if (static_cast<std::underlying_type_t<Options>>(Options::kExternalStream) & options) {
      if (!fUseSession)
         throw std::runtime_error("sofie: RModel::Generate: Options::kExternalStream requires a Session class "
                                  "(the stream is passed to its constructor)");
      fExternalStream = true;
   }

   if (static_cast<std::underlying_type_t<Options>>(Options::kCustomMemory) & options) {
      if (!fUseSession)
         throw std::runtime_error("sofie: RModel::Generate: Options::kCustomMemory requires a Session class "
                                  "(the memory provider is passed to its constructor)");
      fCustomMemory = true;
      AddNeededHelperFunction("CustomMemory");
   }

   if (static_cast<std::underlying_type_t<Options>>(Options::kDebug) & options) {
      fDebug = true;
      AddNeededHelperFunction("DebugLog");
      if (fUseSession)
         AddNeededHelperFunction("DebugMemory");
   }

   Utils::TraceGeneration(*this, std::string("generating the GPU (alpaka) code, options: custom memory ") + (fCustomMemory ? "on" : "off") +
                   ", external stream " + (fExternalStream ? "on" : "off") + ", strided input " + (fStridedInput ? "on" : "off") +
                   ", profile " + (fProfile ? "on" : "off") + ", kernel only " + (fKernelOnly ? "on" : "off") + ", session " +
                   (fUseSession ? "on" : "off") + ", " + std::to_string(fOperators.size()) + " operators");

   Initialize(batchSize, verbose);
   Utils::TraceGeneration(*this, "model initialized: " + std::to_string(fInitializedTensors.size()) + " initialized tensors, " +
                   std::to_string(fIntermediateTensorInfos.size()) + " intermediate tensors, " +
                   std::to_string(fDynamicTensorInfos.size()) + " dynamic tensors, " + std::to_string(fSubGraphs.size()) + " sub-graphs");

   CheckAlpakaOperators();
   Utils::TraceGeneration(*this, "all the operators have an alpaka implementation");

   fFusion = Fusion::Compute(*this);
   Utils::TraceGeneration(*this, "fusion: " + std::to_string(fFusion.kernel.size()) + " kernel groups, " + std::to_string(fFusion.eltwise.size()) +
                   " elementwise groups, " + std::to_string(fFusion.skip.size()) + " operators fused into another one");

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
      FinalizeGeneratedCode();
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
         fGC += Utils::DebugLogCode(*this, "\"uploading the weights of tensor " + i.first + " to the device (" + std::to_string(length * GetTypeSize(i.second.type())) + " bytes)\"");
         fGC += "auto hostBuf_"+i.first+" = alpaka::createView(hostAcc, tensor_"+i.first+", " + slength + ");\n";
         fGC += "alpaka::memcpy(queue, deviceBuf_"+i.first+", hostBuf_"+i.first+");\n";
   }
}

} // namespace SOFIE
