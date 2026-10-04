#include <algorithm>
#include <cctype>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SOFIE/RModel.hxx"
#include "SOFIE/RModelFusion_ALPAKA.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_LeakyRelu.hxx"
#include "SOFIE/ROperator_Relu.hxx"

namespace SOFIE {

namespace {

Dim MultiplyDims(const Dim &a, const Dim &b)
{
   if (!a.isParam && !b.isParam)
      return Dim{a.dim * b.dim};
   if (a.GetVal() == "1")
      return b;
   if (b.GetVal() == "1")
      return a;
   return Dim{a.GetVal() + " * " + b.GetVal()};
}

void AddInput(Fusion::Group &group, const Fusion::Input &input)
{
   const auto existing = std::find_if(group.externalInputs.begin(), group.externalInputs.end(),
                                      [&](const Fusion::Input &i) { return i.tensorName == input.tensorName; });

   if (existing == group.externalInputs.end()) {
      group.externalInputs.push_back(input);
      return;
   }

   if (existing->access != input.access || existing->alignedStrides != input.alignedStrides ||
       existing->customIndexExpression != input.customIndexExpression)
      throw std::runtime_error("Conflicting fused access modes for tensor " + input.tensorName);
}

} // anonymous

bool Fusion::ResolveAccess(const RModel &model, const std::string &tensorName, const std::vector<Dim> &outputShape,
                           Access &access, std::vector<Dim> &alignedStrides)
{
   alignedStrides.clear();

   try {
      const auto inputShape = model.GetDimTensorShape(tensorName);

      if (inputShape == outputShape) {
         access = Access::Elementwise;
         return true;
      }

      if (inputShape.size() > outputShape.size())
         return false;

      const std::string inputLength = inputShape.empty() ? "1" : ConvertDimShapeToLength(inputShape);

      if (inputLength == "1") {
         access = Access::Scalar;
         return true;
      }

      alignedStrides.assign(outputShape.size(), Dim{static_cast<size_t>(0)});
      Dim inputStride{static_cast<size_t>(1)};

      for (size_t inputDimIdx = inputShape.size(); inputDimIdx-- > 0;) {
         const size_t outputDimIdx = outputShape.size() - inputShape.size() + inputDimIdx;
         const Dim &inputDim = inputShape[inputDimIdx];
         const bool isOne = inputDim.GetVal() == "1";

         if (!isOne && inputDim != outputShape[outputDimIdx])
            return false;

         if (!isOne)
            alignedStrides[outputDimIdx] = inputStride;

         inputStride = MultiplyDims(inputStride, inputDim);
      }

      access = Access::Broadcast;
      return true;
   } catch (...) {
      return false;
   }
}

class FusionPlanner {
public:
   explicit FusionPlanner(RModel &model) : m(model) {}

   Fusion::Plan Run()
   {
      FuseGemmActivations();
      BuildUses();
      EnumerateSpecial();
      EnumerateDag();
      EnumerateLinear();

      for (const size_t idx : Select())
         plan.eltwise.push_back(BuildGroup(candidates[idx]));

      std::sort(plan.eltwise.begin(), plan.eltwise.end(), [](const Group &left, const Group &right) {
         return left.opIndices.front() < right.opIndices.front();
      });

      for (const auto &group : plan.eltwise) {
         plan.internal.insert(group.internalTensors.begin(), group.internalTensors.end());

         if (m.fVerbose) {
            std::cout << "[SOFIE elementwise fusion] operators";
            for (const size_t opIdx : group.opIndices)
               std::cout << " " << opIdx;
            std::cout << " ->";
            for (const auto &outputName : group.outputTensors)
               std::cout << " " << outputName;
            std::cout << " with " << group.externalInputs.size() << " external input(s)" << std::endl;
         }
      }

      for (const auto &group : plan.eltwise)
         KeepAlive(group, group.launchOpIndex);

      plan.kernel = SelectKernelGroups(EnumerateKernelGroups(BuildUnits()));

      for (const auto &group : plan.kernel) {
         for (const auto &branch : group.branches)
            KeepAlive(branch, group.launchOpIndex);
      }

      return std::move(plan);
   }

private:
   using Group = Fusion::Group;
   using Input = Fusion::Input;
   using Access = Fusion::Access;

   struct Uses {
      std::unordered_map<std::string, std::vector<size_t>> consumers;
      std::unordered_map<std::string, size_t> producers;
   };

   struct Score {
      size_t launchesRemoved = 0;
      size_t liveRangeExtensionByteSteps = 0;
      size_t eliminatedBytes = 0;
      size_t materializedOutputs = 0;
      size_t externalInputs = 0;

      bool operator==(const Score &) const = default;

      void Add(const Score &other)
      {
         launchesRemoved += other.launchesRemoved;
         eliminatedBytes += other.eliminatedBytes;
         materializedOutputs += other.materializedOutputs;
         externalInputs += other.externalInputs;
      }

      bool Better(const Score &other) const
      {
         if (launchesRemoved != other.launchesRemoved)
            return launchesRemoved > other.launchesRemoved;
         if (liveRangeExtensionByteSteps != other.liveRangeExtensionByteSteps)
            return liveRangeExtensionByteSteps < other.liveRangeExtensionByteSteps;
         if (eliminatedBytes != other.eliminatedBytes)
            return eliminatedBytes > other.eliminatedBytes;
         if (materializedOutputs != other.materializedOutputs)
            return materializedOutputs < other.materializedOutputs;
         return externalInputs < other.externalInputs;
      }
   };

   struct Candidate {
      std::vector<size_t> opIndices;
      std::vector<std::string> externalInputs;
      std::vector<std::string> materializedOutputs;
      std::vector<std::string> internalTensors;
      Score score;
      size_t launchOpIndex = 0;
      std::optional<Group> prebuilt;
   };

   struct Chain {
      Group group;
      std::vector<std::string> produced;
      std::vector<Dim> outputShape;
      std::vector<Dim> logicalShape;
      size_t current = 0;
      bool reorganized = false;
   };

   RModel &m;
   Fusion::Plan plan;
   Uses uses;
   std::vector<Candidate> candidates;
   std::set<std::pair<std::vector<size_t>, size_t>> seen;

   const ROperator &Op(size_t opIdx) const { return *m.fOperators[opIdx]; }

   bool IsOutput(const std::string &tensorName) const
   {
      return std::find(m.fOutputTensorNames.begin(), m.fOutputTensorNames.end(), tensorName) !=
             m.fOutputTensorNames.end();
   }

   size_t Bytes(const std::string &tensorName) const
   {
      size_t length = 1;

      for (const auto &d : m.GetDimTensorShape(tensorName)) {
         if (!d.isParam) {
            length *= d.dim;
            continue;
         }

         size_t value = 1;
         const auto it = m.fShapeParams.find(d.param);

         if (it != m.fShapeParams.end()) {
            try {
               value = std::stoul(it->second);
            } catch (...) {
            }
         }

         length *= value;
      }

      return GetTypeSize(m.GetTensorType(tensorName)) * length;
   }

   std::optional<std::pair<std::string, size_t>> Overrun(const std::string &input, size_t launch) const
   {
      std::string name = m.ResolveAliasTensor(input);
      const auto it = m.fIntermediateTensorFrequencyLookup.find(name);

      if (it == m.fIntermediateTensorFrequencyLookup.end() || launch <= it->second)
         return std::nullopt;

      return std::make_pair(std::move(name), it->second);
   }

   size_t ExtensionBytes(const std::vector<std::pair<std::string, size_t>> &inputs) const
   {
      std::unordered_map<std::string, std::pair<size_t, size_t>> required;

      for (const auto &[input, launch] : inputs) {
         const auto overrun = Overrun(input, launch);

         if (!overrun)
            continue;

         auto [it, inserted] = required.try_emplace(overrun->first, launch, overrun->second);
         if (!inserted)
            it->second.first = std::max(it->second.first, launch);
      }

      size_t cost = 0;

      for (const auto &[name, range] : required)
         cost += Bytes(name) * (range.first - range.second);

      return cost;
   }

   void KeepAlive(const Group &group, size_t launch)
   {
      for (const auto &input : group.externalInputs) {
         const std::string name = m.ResolveAliasTensor(input.tensorName);
         const auto it = m.fIntermediateTensorFrequencyLookup.find(name);

         if (it != m.fIntermediateTensorFrequencyLookup.end())
            it->second = std::max(it->second, launch);
      }
   }

   void FuseGemmActivations()
   {
      std::unordered_map<std::string, size_t> consumerCount;

      for (const auto &op : m.fOperators) {
         for (const auto &inputName : op->GetOpInputTensors())
            ++consumerCount[std::string(inputName)];
      }

      for (size_t opIdx = 0; opIdx + 1 < m.fOperators.size(); ++opIdx) {
         const size_t activationOpIdx = opIdx + 1;

         if (plan.skip.count(opIdx) || plan.skip.count(activationOpIdx))
            continue;

         auto *gemm = dynamic_cast<ROperator_Gemm<float> *>(m.fOperators[opIdx].get());

         if (!gemm || gemm->GetActivationType() != EActivationType::UNDEFINED)
            continue;

         auto *leakyRelu = dynamic_cast<ROperator_LeakyRelu<float> *>(m.fOperators[activationOpIdx].get());
         auto *relu = dynamic_cast<ROperator_Relu<float> *>(m.fOperators[activationOpIdx].get());

         if (!leakyRelu && !relu)
            continue;

         const auto gemmOutputs = Op(opIdx).GetOpOutputTensors();
         const auto activationInputs = Op(activationOpIdx).GetOpInputTensors();
         const auto activationOutputs = Op(activationOpIdx).GetOpOutputTensors();

         if (gemmOutputs.size() != 1 || activationInputs.size() != 1 || activationOutputs.size() != 1)
            continue;

         const std::string gemmOutput(gemmOutputs[0]);

         if (gemmOutput != std::string(activationInputs[0]) || consumerCount[gemmOutput] != 1 || IsOutput(gemmOutput))
            continue;

         if (relu && !gemm->HasBias())
            continue;

         if (leakyRelu)
            gemm->SetActivation(EActivationType::LEAKYRELU, leakyRelu->GetAlpha());
         else
            gemm->SetActivation(EActivationType::RELU);

         gemm->UpdateFusableTensorName(std::string(activationOutputs[0]), [](const std::string &) {});
         plan.skip.insert(activationOpIdx);
      }
   }

   void BuildUses()
   {
      for (size_t opIdx = 0; opIdx < m.fOperators.size(); ++opIdx) {
         const auto outputs = Op(opIdx).GetOpOutputTensors();

         const bool hasRuntimeOutput = std::any_of(outputs.begin(), outputs.end(), [&](const auto &outputName) {
            return !m.IsInitializedTensor(std::string(outputName));
         });

         if (hasRuntimeOutput) {
            for (const auto &inputName : Op(opIdx).GetOpInputTensors())
               uses.consumers[std::string(inputName)].push_back(opIdx);
         }

         for (const auto &outputName : outputs) {
            if (!m.IsInitializedTensor(std::string(outputName)))
               uses.producers[std::string(outputName)] = opIdx;
         }
      }
   }

   bool Supported(size_t opIdx, bool allowShuffle, bool allowReorganize, bool allowManyToMany = false) const
   {
      if (opIdx >= m.fOperators.size() || plan.skip.count(opIdx))
         return false;

      const auto &op = Op(opIdx);
      const auto mapping = op.GetFusionMappingType();
      const bool shuffle = mapping == EFusionMappingType::Shuffle;
      const bool reorganize = mapping == EFusionMappingType::Reorganize;
      const bool manyToMany = mapping == EFusionMappingType::ManyToMany;
      const bool pointwise = mapping == EFusionMappingType::OneToOne || mapping == EFusionMappingType::OneToMany;

      if (!(pointwise || (allowShuffle && shuffle) || (allowReorganize && reorganize) ||
            (allowManyToMany && manyToMany)))
         return false;

      const auto inputs = op.GetOpInputTensors();
      const auto outputs = op.GetOpOutputTensors();
      const auto dataInputs = op.GetFusionDataInputIndices();
      const bool multiOutput = mapping == EFusionMappingType::OneToMany && outputs.size() > 1;

      if (dataInputs.empty() || outputs.empty() || (outputs.size() > 1 && !multiOutput))
         return false;

      for (const size_t inputIdx : dataInputs) {
         if (inputIdx >= inputs.size())
            return false;
      }

      const auto hasExpression = [&] {
         std::vector<std::string> args;
         for (size_t i = 0; i < dataInputs.size(); ++i)
            args.push_back("x" + std::to_string(i));
         return !op.GetFusionExpr(args).empty();
      };

      const auto inputShape = [&](size_t inputIdx, std::vector<Dim> &shape) {
         try {
            shape = m.GetDimTensorShape(std::string(inputs[inputIdx]));
            return true;
         } catch (...) {
            return false;
         }
      };

      if (multiOutput) {
         try {
            std::vector<ETensorType> inputTypes;
            for (const size_t inputIdx : dataInputs)
               inputTypes.push_back(m.GetTensorType(std::string(inputs[inputIdx])));

            for (size_t outputIdx = 0; outputIdx < outputs.size(); ++outputIdx) {
               const std::string outputName(outputs[outputIdx]);

               if (m.IsAliasTensor(outputName))
                  return false;

               const auto outputShape = m.GetDimTensorShape(outputName);

               if (!op.SupportsFusionTypes(inputTypes, m.GetTensorType(outputName)))
                  return false;

               for (const size_t inputIdx : dataInputs) {
                  const auto shape = m.GetDimTensorShape(std::string(inputs[inputIdx]));

                  if (op.GetFusionInputIndexExprForOutput(inputIdx, outputIdx, "idx", shape, outputShape).empty())
                     return false;
               }
            }
         } catch (...) {
            return false;
         }

         return hasExpression();
      }

      const std::string outputName(outputs[0]);

      if (m.IsAliasTensor(outputName) && !reorganize)
         return false;

      std::vector<Dim> outputShape;
      std::vector<ETensorType> inputTypes;
      ETensorType outputType = ETensorType::UNDEFINED;

      try {
         outputShape = m.GetDimTensorShape(outputName);
         outputType = m.GetTensorType(outputName);

         for (const size_t inputIdx : dataInputs)
            inputTypes.push_back(m.GetTensorType(std::string(inputs[inputIdx])));
      } catch (...) {
         return false;
      }

      if (!op.SupportsFusionTypes(inputTypes, outputType))
         return false;

      std::vector<Dim> shape;

      if (manyToMany) {
         if (op.IsFusionReduction()) {
            if (dataInputs.size() != 1 || !inputShape(dataInputs[0], shape))
               return false;

            const std::string inputLength = ConvertDimShapeToLength(shape);
            const std::string outputLength = ConvertDimShapeToLength(outputShape);
            bool numeric = true;
            size_t inputValue = 0, outputValue = 0;

            try {
               inputValue = std::stoul(inputLength);
               outputValue = std::stoul(outputLength);
            } catch (...) {
               numeric = false;
            }

            if (numeric && (outputValue == 0 || inputValue % outputValue != 0))
               return false;

            const std::string reducedLength = "(" + inputLength + ") / (" + outputLength + ")";

            return !op.GetFusionReductionInitExpr().empty() &&
                   !op.GetFusionReductionAccumulateExpr("acc", "value").empty() &&
                   !op.GetFusionReductionCombineExpr("left", "right").empty() &&
                   !op.GetFusionReductionFinalizeExpr("acc", reducedLength).empty() &&
                   !op.GetFusionReductionInputIndexExpr("out_idx", "r", shape, outputShape).empty();
         }

         if (dataInputs.size() < 2)
            return false;

         for (size_t dataIdx = 0; dataIdx < dataInputs.size(); ++dataIdx) {
            const size_t inputIdx = dataInputs[dataIdx];

            if (!inputShape(inputIdx, shape) || op.GetFusionInputIndexExpr(inputIdx, "idx", shape, outputShape).empty())
               return false;

            if (dataIdx + 1 < dataInputs.size() &&
                op.GetFusionInputConditionExpr(inputIdx, "idx", shape, outputShape).empty())
               return false;
         }

         return !op.GetFusionExpr({"x"}).empty();
      }

      if (shuffle) {
         if (dataInputs.size() != 1 || !inputShape(dataInputs[0], shape) ||
             op.GetFusionInputIndexExpr(dataInputs[0], "idx", shape, outputShape).empty())
            return false;
      } else if (reorganize) {
         if (dataInputs.size() != 1)
            return false;

         try {
            if (ConvertDimShapeToLength(m.GetDimTensorShape(std::string(inputs[dataInputs[0]]))) !=
                ConvertDimShapeToLength(outputShape))
               return false;
         } catch (...) {
            return false;
         }
      } else {
         for (const size_t inputIdx : dataInputs) {
            Access access;
            std::vector<Dim> strides;

            if (!Fusion::ResolveAccess(m, std::string(inputs[inputIdx]), outputShape, access, strides))
               return false;
         }
      }

      return hasExpression();
   }

   Chain Start(size_t firstOpIdx) const
   {
      Chain chain;
      const auto &op = Op(firstOpIdx);
      const std::string firstOutput(op.GetOpOutputTensors()[0]);

      chain.outputShape = m.GetDimTensorShape(firstOutput);
      chain.group.opIndices.push_back(firstOpIdx);

      const auto inputs = op.GetOpInputTensors();
      const bool shuffle = op.GetFusionMappingType() == EFusionMappingType::Shuffle;

      for (const size_t inputIdx : op.GetFusionDataInputIndices()) {
         const std::string inputName(inputs[inputIdx]);

         if (shuffle) {
            std::vector<Dim> inputShape;

            try {
               inputShape = m.GetDimTensorShape(inputName);
            } catch (...) {
               throw std::runtime_error("Cannot resolve Shuffle input shape for fusion: " + inputName);
            }

            const std::string indexExpression = op.GetFusionInputIndexExpr(inputIdx, "idx", inputShape, chain.outputShape);

            if (indexExpression.empty())
               throw std::runtime_error("Shuffle operator does not provide a fused input index expression");

            AddInput(chain.group, {inputName, Access::Elementwise, {}, indexExpression});
            continue;
         }

         Access access;
         std::vector<Dim> strides;

         if (!Fusion::ResolveAccess(m, inputName, chain.outputShape, access, strides))
            throw std::runtime_error("Invalid external input for fusion group: " + inputName);

         AddInput(chain.group, {inputName, access, strides, ""});
      }

      chain.produced.push_back(firstOutput);
      chain.logicalShape = chain.outputShape;
      chain.current = firstOpIdx;
      return chain;
   }

   bool Extend(Chain &chain, bool allowReorganize) const
   {
      const auto currentOutputs = Op(chain.current).GetOpOutputTensors();

      if (currentOutputs.size() != 1)
         return false;

      const std::string currentOutput(currentOutputs[0]);
      const auto consumerIt = uses.consumers.find(currentOutput);

      if (IsOutput(currentOutput) || consumerIt == uses.consumers.end() || consumerIt->second.size() != 1)
         return false;

      const size_t nextOpIdx = consumerIt->second.front();

      if (nextOpIdx <= chain.current)
         return false;

      const auto nextInputs = Op(nextOpIdx).GetOpInputTensors();
      const auto nextDataInputs = Op(nextOpIdx).GetFusionDataInputIndices();

      const auto isProduced = [&](const std::string &name) {
         return std::find(chain.produced.begin(), chain.produced.end(), name) != chain.produced.end();
      };

      for (const size_t inputIdx : nextDataInputs) {
         const std::string inputName(nextInputs[inputIdx]);

         if (inputName == currentOutput || isProduced(inputName))
            continue;

         const auto producerIt = uses.producers.find(inputName);

         if (producerIt != uses.producers.end() && producerIt->second >= nextOpIdx)
            return false;
      }

      if (!Supported(nextOpIdx, false, allowReorganize))
         return false;

      const auto nextOutputs = Op(nextOpIdx).GetOpOutputTensors();

      if (nextOutputs.size() != 1)
         return false;

      const std::string nextOutput(nextOutputs[0]);
      std::vector<Dim> nextOutputShape;

      try {
         nextOutputShape = m.GetDimTensorShape(nextOutput);
      } catch (...) {
         return false;
      }

      const bool nextIsReorganize = Op(nextOpIdx).GetFusionMappingType() == EFusionMappingType::Reorganize;

      if (nextIsReorganize) {
         if (ConvertDimShapeToLength(nextOutputShape) != ConvertDimShapeToLength(chain.logicalShape))
            return false;

         const bool hasBroadcast = std::any_of(chain.group.externalInputs.begin(), chain.group.externalInputs.end(),
                                               [](const Input &input) { return input.access == Access::Broadcast; });

         if (hasBroadcast)
            return false;
      } else if (nextOutputShape != chain.logicalShape) {
         return false;
      }

      std::vector<Input> pending;

      for (const size_t inputIdx : nextDataInputs) {
         const std::string inputName(nextInputs[inputIdx]);

         if (isProduced(inputName))
            continue;

         Access access;
         std::vector<Dim> strides;

         if (!Fusion::ResolveAccess(m, inputName, nextIsReorganize ? chain.logicalShape : nextOutputShape, access, strides))
            return false;

         if (chain.reorganized && access == Access::Broadcast)
            return false;

         pending.push_back({inputName, access, strides, ""});
      }

      for (const auto &input : pending)
         AddInput(chain.group, input);

      chain.group.opIndices.push_back(nextOpIdx);
      chain.produced.push_back(nextOutput);
      chain.current = nextOpIdx;

      if (nextIsReorganize) {
         chain.logicalShape = nextOutputShape;
         chain.reorganized = true;
      }

      return true;
   }

   template <class Visit>
   void ForEachChain(bool allowReorganize, Visit &&visit) const
   {
      for (size_t firstOpIdx = 0; firstOpIdx < m.fOperators.size(); ++firstOpIdx) {
         if (!Supported(firstOpIdx, true, false) || Op(firstOpIdx).GetOpOutputTensors().size() != 1)
            continue;

         Chain chain = Start(firstOpIdx);

         while (Extend(chain, allowReorganize))
            visit(chain);
      }
   }

   Candidate BuildCandidate(std::vector<size_t> opIndices) const
   {
      Candidate candidate;
      candidate.opIndices = std::move(opIndices);

      std::sort(candidate.opIndices.begin(), candidate.opIndices.end());
      candidate.opIndices.erase(std::unique(candidate.opIndices.begin(), candidate.opIndices.end()),
                                candidate.opIndices.end());

      for (const size_t opIdx : candidate.opIndices) {
         if (opIdx >= m.fOperators.size())
            throw std::runtime_error("Invalid operator index in fusion candidate: " + std::to_string(opIdx));
      }

      const auto contains = [&](size_t opIdx) {
         return std::binary_search(candidate.opIndices.begin(), candidate.opIndices.end(), opIdx);
      };

      const auto addUnique = [](std::vector<std::string> &tensors, const std::string &name) {
         if (std::find(tensors.begin(), tensors.end(), name) == tensors.end())
            tensors.push_back(name);
      };

      for (const size_t opIdx : candidate.opIndices) {
         const auto inputs = Op(opIdx).GetOpInputTensors();

         for (const size_t inputIdx : Op(opIdx).GetFusionDataInputIndices()) {
            if (inputIdx >= inputs.size())
               throw std::runtime_error("Invalid fusion data input index for operator " + std::to_string(opIdx));

            const std::string inputName(inputs[inputIdx]);
            const auto producerIt = uses.producers.find(inputName);

            if (producerIt == uses.producers.end() || !contains(producerIt->second))
               addUnique(candidate.externalInputs, inputName);
         }

         for (const auto &outputView : Op(opIdx).GetOpOutputTensors()) {
            const std::string outputName(outputView);
            const auto consumerIt = uses.consumers.find(outputName);
            const bool hasConsumers = consumerIt != uses.consumers.end() && !consumerIt->second.empty();
            const bool hasExternalConsumer =
               hasConsumers && std::any_of(consumerIt->second.begin(), consumerIt->second.end(),
                                           [&](size_t consumerOpIdx) { return !contains(consumerOpIdx); });

            if (!hasConsumers || hasExternalConsumer || IsOutput(outputName))
               addUnique(candidate.materializedOutputs, outputName);
            else
               addUnique(candidate.internalTensors, outputName);
         }
      }

      return candidate;
   }

   bool Valid(const Candidate &candidate) const
   {
      if (candidate.opIndices.size() < 2)
         return false;

      const std::set<size_t> candidateOps(candidate.opIndices.begin(), candidate.opIndices.end());
      std::unordered_map<size_t, std::vector<size_t>> adjacency;

      for (const size_t opIdx : candidate.opIndices) {
         const auto inputs = Op(opIdx).GetOpInputTensors();

         for (const size_t inputIdx : Op(opIdx).GetFusionDataInputIndices()) {
            if (inputIdx >= inputs.size())
               return false;

            const auto producerIt = uses.producers.find(std::string(inputs[inputIdx]));
            if (producerIt == uses.producers.end() || !candidateOps.count(producerIt->second))
               continue;

            adjacency[opIdx].push_back(producerIt->second);
            adjacency[producerIt->second].push_back(opIdx);
         }
      }

      std::set<size_t> visited;
      std::vector<size_t> pending{candidate.opIndices.front()};

      while (!pending.empty()) {
         const size_t opIdx = pending.back();
         pending.pop_back();

         if (!visited.insert(opIdx).second)
            continue;

         for (const size_t neighborIdx : adjacency[opIdx]) {
            if (!visited.count(neighborIdx))
               pending.push_back(neighborIdx);
         }
      }

      if (visited.size() != candidate.opIndices.size())
         return false;

      std::vector<size_t> reductionOps;

      for (const size_t opIdx : candidate.opIndices) {
         if (!Supported(opIdx, true, true, true))
            return false;

         const auto outputs = Op(opIdx).GetOpOutputTensors();

         if (outputs.empty() || (outputs.size() > 1 && Op(opIdx).GetFusionMappingType() != EFusionMappingType::OneToMany))
            return false;

         try {
            for (const auto &output : outputs)
               m.GetDimTensorShape(std::string(output));
         } catch (...) {
            return false;
         }
      }

      for (const size_t opIdx : candidate.opIndices) {
         if (Op(opIdx).IsFusionReduction())
            reductionOps.push_back(opIdx);
      }

      if (reductionOps.size() > 1)
         return false;

      if (reductionOps.size() == 1) {
         const size_t reductionOpIdx = reductionOps[0];
         const auto &reductionOp = Op(reductionOpIdx);
         const auto reductionInputs = reductionOp.GetOpInputTensors();
         const auto reductionOutputs = reductionOp.GetOpOutputTensors();
         const auto reductionDataInputs = reductionOp.GetFusionDataInputIndices();

         if (reductionDataInputs.size() != 1 || reductionOutputs.size() != 1)
            return false;

         const auto reductionInputShape = m.GetDimTensorShape(std::string(reductionInputs[reductionDataInputs[0]]));
         const auto reductionOutputShape = m.GetDimTensorShape(std::string(reductionOutputs[0]));

         for (const auto &outputName : candidate.materializedOutputs) {
            const auto outputShape = m.GetDimTensorShape(outputName);
            if (outputShape != reductionInputShape && outputShape != reductionOutputShape)
               return false;
         }

         for (const size_t opIdx : candidate.opIndices) {
            if (opIdx == reductionOpIdx)
               continue;

            const auto mapping = Op(opIdx).GetFusionMappingType();
            if (mapping != EFusionMappingType::OneToOne && mapping != EFusionMappingType::OneToMany)
               return false;
         }
      }

      if (candidate.materializedOutputs.empty())
         return false;

      const std::string length = ConvertDimShapeToLength(m.GetDimTensorShape(candidate.materializedOutputs.front()));
      const ETensorType type = m.GetTensorType(candidate.materializedOutputs.front());

      for (const auto &outputName : candidate.materializedOutputs) {
         if (m.IsAliasTensor(outputName) || ConvertDimShapeToLength(m.GetDimTensorShape(outputName)) != length ||
             m.GetTensorType(outputName) != type)
            return false;
      }

      return true;
   }

   Score ScoreOf(const Candidate &candidate) const
   {
      Score score;
      score.launchesRemoved = candidate.opIndices.empty() ? 0 : candidate.opIndices.size() - 1;
      score.materializedOutputs = candidate.materializedOutputs.size();
      score.externalInputs = candidate.externalInputs.size();

      for (const auto &input : candidate.externalInputs)
         score.liveRangeExtensionByteSteps += ExtensionBytes({{input, candidate.launchOpIndex}});

      for (const auto &outputName : candidate.materializedOutputs) {
         const auto producerIt = uses.producers.find(outputName);

         if (producerIt != uses.producers.end() && candidate.launchOpIndex < producerIt->second)
            score.liveRangeExtensionByteSteps += Bytes(outputName) * (producerIt->second - candidate.launchOpIndex);
      }

      for (const auto &tensorName : candidate.internalTensors)
         score.eliminatedBytes += Bytes(tensorName);

      return score;
   }

   void Add(Candidate candidate)
   {
      if (seen.insert({candidate.opIndices, candidate.launchOpIndex}).second)
         candidates.push_back(std::move(candidate));
   }

   void AddOptions(const Candidate &candidate)
   {
      const auto schedulable = [&](size_t launchOpIdx) {
         for (const auto &inputName : candidate.externalInputs) {
            const auto producerIt = uses.producers.find(inputName);

            if (producerIt != uses.producers.end() && producerIt->second >= launchOpIdx)
               return false;
         }

         for (const auto &outputName : candidate.materializedOutputs) {
            const auto producerIt = uses.producers.find(outputName);

            if (producerIt == uses.producers.end() || producerIt->second > launchOpIdx)
               return false;

            const auto consumerIt = uses.consumers.find(outputName);

            if (consumerIt == uses.consumers.end())
               continue;

            for (const size_t consumerOpIdx : consumerIt->second) {
               if (!std::binary_search(candidate.opIndices.begin(), candidate.opIndices.end(), consumerOpIdx) &&
                   consumerOpIdx < launchOpIdx)
                  return false;
            }
         }

         return true;
      };

      for (const size_t launchOpIdx : candidate.opIndices) {
         if (!schedulable(launchOpIdx))
            continue;

         Candidate option = candidate;
         option.launchOpIndex = launchOpIdx;
         option.score = ScoreOf(option);
         Add(std::move(option));
      }
   }

   void EnumerateSpecial()
   {
      ForEachChain(true, [&](const Chain &chain) {
         const bool special = std::any_of(chain.group.opIndices.begin(), chain.group.opIndices.end(), [&](size_t opIdx) {
            const auto mapping = Op(opIdx).GetFusionMappingType();
            return mapping == EFusionMappingType::Shuffle || mapping == EFusionMappingType::Reorganize;
         });

         const std::string output(Op(chain.current).GetOpOutputTensors()[0]);

         if (chain.group.opIndices.size() < 2 || !special || m.IsAliasTensor(output))
            return;

         Group group = chain.group;
         group.outputTensors = {output};
         group.numElements = ConvertDimShapeToLength(chain.outputShape);
         group.launchOpIndex = chain.current;

         for (size_t i = 0; i + 1 < group.opIndices.size(); ++i) {
            const auto outputs = Op(group.opIndices[i]).GetOpOutputTensors();
            if (!outputs.empty())
               group.internalTensors.push_back(std::string(outputs[0]));
         }

         Candidate candidate;
         candidate.opIndices = group.opIndices;
         candidate.materializedOutputs = group.outputTensors;
         candidate.internalTensors = group.internalTensors;
         candidate.launchOpIndex = group.launchOpIndex;

         for (const auto &input : group.externalInputs)
            candidate.externalInputs.push_back(input.tensorName);

         candidate.prebuilt = std::move(group);
         candidate.score = ScoreOf(candidate);
         Add(std::move(candidate));
      });
   }

   void EnumerateDag()
   {
      constexpr size_t maxCandidateOps = 8;

      const size_t n = m.fOperators.size();
      std::vector<bool> supported(n, false);
      std::vector<std::vector<size_t>> adjacency(n);

      for (size_t opIdx = 0; opIdx < n; ++opIdx)
         supported[opIdx] = Supported(opIdx, true, true, true);

      for (size_t consumerIdx = 0; consumerIdx < n; ++consumerIdx) {
         if (!supported[consumerIdx])
            continue;

         const auto inputs = Op(consumerIdx).GetOpInputTensors();

         for (const size_t inputIdx : Op(consumerIdx).GetFusionDataInputIndices()) {
            if (inputIdx >= inputs.size())
               continue;

            const auto producerIt = uses.producers.find(std::string(inputs[inputIdx]));

            if (producerIt == uses.producers.end())
               continue;

            const size_t producerIdx = producerIt->second;

            if (producerIdx == consumerIdx || !supported[producerIdx])
               continue;

            adjacency[producerIdx].push_back(consumerIdx);
            adjacency[consumerIdx].push_back(producerIdx);
         }
      }

      for (auto &neighbors : adjacency) {
         std::sort(neighbors.begin(), neighbors.end());
         neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
      }

      std::set<std::vector<size_t>> visited;

      for (size_t seedIdx = 0; seedIdx < n; ++seedIdx) {
         std::vector<size_t> seed{seedIdx};

         if (!supported[seedIdx] || !visited.insert(seed).second)
            continue;

         std::vector<std::vector<size_t>> pending;
         pending.push_back(std::move(seed));

         while (!pending.empty()) {
            std::vector<size_t> opIndices = std::move(pending.back());
            pending.pop_back();

            if (opIndices.size() >= 2) {
               const Candidate candidate = BuildCandidate(opIndices);

               if (Valid(candidate))
                  AddOptions(candidate);
            }

            if (opIndices.size() >= maxCandidateOps)
               continue;

            std::vector<size_t> expansion;

            for (const size_t opIdx : opIndices) {
               for (const size_t neighborIdx : adjacency[opIdx]) {
                  if (!std::binary_search(opIndices.begin(), opIndices.end(), neighborIdx))
                     expansion.push_back(neighborIdx);
               }
            }

            std::sort(expansion.begin(), expansion.end());
            expansion.erase(std::unique(expansion.begin(), expansion.end()), expansion.end());

            for (const size_t nextOpIdx : expansion) {
               std::vector<size_t> next = opIndices;
               next.insert(std::lower_bound(next.begin(), next.end(), nextOpIdx), nextOpIdx);

               if (visited.insert(next).second)
                  pending.push_back(std::move(next));
            }
         }
      }
   }

   void EnumerateLinear()
   {
      ForEachChain(false, [&](const Chain &chain) {
         const Candidate candidate = BuildCandidate(chain.group.opIndices);

         if (Valid(candidate))
            AddOptions(candidate);
      });
   }

   bool Conflict(const Candidate &left, const Candidate &right) const
   {
      size_t leftIdx = 0;
      size_t rightIdx = 0;

      while (leftIdx < left.opIndices.size() && rightIdx < right.opIndices.size()) {
         if (left.opIndices[leftIdx] == right.opIndices[rightIdx])
            return true;

         if (left.opIndices[leftIdx] < right.opIndices[rightIdx])
            ++leftIdx;
         else
            ++rightIdx;
      }

      const auto misordered = [&](const Candidate &producer, const Candidate &consumer) {
         for (const auto &outputName : producer.materializedOutputs) {
            const auto consumerIt = uses.consumers.find(outputName);
            if (consumerIt == uses.consumers.end())
               continue;

            for (const size_t consumerOpIdx : consumerIt->second) {
               if (std::binary_search(consumer.opIndices.begin(), consumer.opIndices.end(), consumerOpIdx))
                  return producer.launchOpIndex >= consumer.launchOpIndex;
            }
         }

         return false;
      };

      return misordered(left, right) || misordered(right, left);
   }

   std::vector<size_t> Select() const
   {
      std::vector<size_t> selection;

      if (candidates.empty())
         return selection;

      const auto planExtension = [&](const std::vector<size_t> &indices) {
         std::vector<std::pair<std::string, size_t>> inputs;

         for (const size_t idx : indices) {
            for (const auto &input : candidates[idx].externalInputs)
               inputs.emplace_back(input, candidates[idx].launchOpIndex);
         }

         return ExtensionBytes(inputs);
      };

      std::vector<std::vector<size_t>> conflicts(candidates.size());

      for (size_t leftIdx = 0; leftIdx < candidates.size(); ++leftIdx) {
         for (size_t rightIdx = leftIdx + 1; rightIdx < candidates.size(); ++rightIdx) {
            if (!Conflict(candidates[leftIdx], candidates[rightIdx]))
               continue;

            conflicts[leftIdx].push_back(rightIdx);
            conflicts[rightIdx].push_back(leftIdx);
         }
      }

      for (auto &candidateConflicts : conflicts)
         std::sort(candidateConflicts.begin(), candidateConflicts.end());

      std::vector<std::vector<size_t>> neighborsOf = conflicts;
      std::unordered_map<std::string, std::vector<size_t>> lifetimeUsers;

      for (size_t candidateIdx = 0; candidateIdx < candidates.size(); ++candidateIdx) {
         for (const auto &input : candidates[candidateIdx].externalInputs) {
            if (const auto overrun = Overrun(input, candidates[candidateIdx].launchOpIndex))
               lifetimeUsers[overrun->first].push_back(candidateIdx);
         }
      }

      for (auto &[tensorName, users] : lifetimeUsers) {
         std::sort(users.begin(), users.end());
         users.erase(std::unique(users.begin(), users.end()), users.end());

         for (size_t idx = 1; idx < users.size(); ++idx) {
            neighborsOf[users.front()].push_back(users[idx]);
            neighborsOf[users[idx]].push_back(users.front());
         }
      }

      for (auto &neighbors : neighborsOf) {
         std::sort(neighbors.begin(), neighbors.end());
         neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
      }

      std::vector<bool> visitedComponent(candidates.size(), false);

      for (size_t seedIdx = 0; seedIdx < candidates.size(); ++seedIdx) {
         if (visitedComponent[seedIdx])
            continue;

         std::vector<size_t> component;
         std::vector<size_t> pending{seedIdx};
         visitedComponent[seedIdx] = true;

         while (!pending.empty()) {
            const size_t candidateIdx = pending.back();
            pending.pop_back();
            component.push_back(candidateIdx);

            for (const size_t neighborIdx : neighborsOf[candidateIdx]) {
               if (visitedComponent[neighborIdx])
                  continue;

               visitedComponent[neighborIdx] = true;
               pending.push_back(neighborIdx);
            }
         }

         std::sort(component.begin(), component.end());

         Score bestScore;
         std::vector<size_t> bestSelection;
         bool hasBest = false;

         std::function<void(const std::vector<size_t> &, Score, std::vector<size_t>)> search;
         search = [&](const std::vector<size_t> &remaining, Score current, std::vector<size_t> selected) {
            if (remaining.empty()) {
               std::sort(selected.begin(), selected.end());

               if (!hasBest || current.Better(bestScore) || (current == bestScore && selected < bestSelection)) {
                  bestScore = current;
                  bestSelection = std::move(selected);
                  hasBest = true;
               }

               return;
            }

            Score optimistic = current;

            for (const size_t candidateIdx : remaining) {
               optimistic.launchesRemoved += candidates[candidateIdx].score.launchesRemoved;
               optimistic.eliminatedBytes += candidates[candidateIdx].score.eliminatedBytes;
            }

            if (hasBest && optimistic.launchesRemoved < bestScore.launchesRemoved)
               return;

            if (hasBest && optimistic.launchesRemoved == bestScore.launchesRemoved &&
                current.liveRangeExtensionByteSteps > bestScore.liveRangeExtensionByteSteps)
               return;

            if (hasBest && optimistic.launchesRemoved == bestScore.launchesRemoved &&
                current.liveRangeExtensionByteSteps == bestScore.liveRangeExtensionByteSteps &&
                optimistic.eliminatedBytes < bestScore.eliminatedBytes)
               return;

            size_t pivotIdx = remaining.front();
            size_t pivotConflictCount = 0;

            for (const size_t candidateIdx : remaining) {
               size_t conflictCount = 0;

               for (const size_t otherIdx : remaining) {
                  if (candidateIdx != otherIdx &&
                      std::binary_search(conflicts[candidateIdx].begin(), conflicts[candidateIdx].end(), otherIdx))
                     ++conflictCount;
               }

               if (conflictCount > pivotConflictCount) {
                  pivotIdx = candidateIdx;
                  pivotConflictCount = conflictCount;
               }
            }

            std::vector<size_t> includeRemaining;
            std::vector<size_t> excludeRemaining;

            for (const size_t candidateIdx : remaining) {
               if (candidateIdx == pivotIdx)
                  continue;

               excludeRemaining.push_back(candidateIdx);

               if (!std::binary_search(conflicts[pivotIdx].begin(), conflicts[pivotIdx].end(), candidateIdx))
                  includeRemaining.push_back(candidateIdx);
            }

            Score included = current;
            included.Add(candidates[pivotIdx].score);
            std::vector<size_t> includeSelected = selected;
            includeSelected.push_back(pivotIdx);
            included.liveRangeExtensionByteSteps = planExtension(includeSelected);
            search(includeRemaining, included, std::move(includeSelected));

            search(excludeRemaining, current, std::move(selected));
         };

         search(component, {}, {});

         selection.insert(selection.end(), bestSelection.begin(), bestSelection.end());
      }

      std::sort(selection.begin(), selection.end());
      return selection;
   }

   Group BuildGroup(const Candidate &candidate) const
   {
      if (candidate.prebuilt) {
         Group group = *candidate.prebuilt;
         group.usesIndexedEvaluation = true;
         return group;
      }

      if (candidate.materializedOutputs.empty())
         throw std::runtime_error("Fusion candidate has no materialized output");

      Group group;
      group.opIndices = candidate.opIndices;
      group.outputTensors = candidate.materializedOutputs;
      group.internalTensors = candidate.internalTensors;
      group.launchOpIndex = candidate.launchOpIndex;

      const auto iterationShape = m.GetDimTensorShape(group.outputTensors.front());
      group.numElements = ConvertDimShapeToLength(iterationShape);

      group.usesIndexedEvaluation = std::any_of(group.opIndices.begin(), group.opIndices.end(), [&](size_t opIdx) {
         const auto mapping = Op(opIdx).GetFusionMappingType();

         if (mapping == EFusionMappingType::Shuffle || mapping == EFusionMappingType::Reorganize ||
             mapping == EFusionMappingType::ManyToMany)
            return true;

         const auto outputs = Op(opIdx).GetOpOutputTensors();

         if (outputs.size() > 1)
            return true;

         return outputs.size() == 1 && m.GetDimTensorShape(std::string(outputs[0])) != iterationShape;
      });

      for (const auto &inputName : candidate.externalInputs) {
         if (group.usesIndexedEvaluation) {
            AddInput(group, {inputName, Access::Elementwise, {}, ""});
            continue;
         }

         Access access;
         std::vector<Dim> strides;

         if (!Fusion::ResolveAccess(m, inputName, iterationShape, access, strides))
            throw std::runtime_error("Cannot resolve external input for fusion candidate: " + inputName);

         AddInput(group, {inputName, access, strides, ""});
      }

      return group;
   }

   std::vector<Group> BuildUnits() const
   {
      std::vector<Group> units;
      std::set<size_t> covered;

      for (const auto &group : plan.eltwise) {
         covered.insert(group.opIndices.begin(), group.opIndices.end());

         if (!group.usesIndexedEvaluation)
            units.push_back(group);
      }

      for (size_t opIdx = 0; opIdx < m.fOperators.size(); ++opIdx) {
         if (covered.count(opIdx) || !Supported(opIdx, false, false))
            continue;

         Candidate candidate = BuildCandidate({opIdx});

         if (candidate.materializedOutputs.empty())
            continue;

         candidate.launchOpIndex = opIdx;
         units.push_back(BuildGroup(candidate));
      }

      std::sort(units.begin(), units.end(), [](const Group &left, const Group &right) {
         return left.launchOpIndex < right.launchOpIndex;
      });

      return units;
   }

   bool CanHorizontallyFuse(const std::vector<Group> &branches, size_t &launchOpIndex) const
   {
      std::unordered_map<size_t, size_t> opToBranch;
      size_t commonLaunch = 0;

      for (size_t branchIdx = 0; branchIdx < branches.size(); ++branchIdx) {
         const auto &branch = branches[branchIdx];

         if (branch.opIndices.empty() || branch.outputTensors.empty())
            return false;

         commonLaunch = std::max(commonLaunch, branch.launchOpIndex);

         for (const size_t opIdx : branch.opIndices) {
            if (!opToBranch.emplace(opIdx, branchIdx).second)
               return false;
         }
      }

      for (size_t branchIdx = 0; branchIdx < branches.size(); ++branchIdx) {
         const auto &branch = branches[branchIdx];

         for (const auto &input : branch.externalInputs) {
            const auto producerIt = uses.producers.find(input.tensorName);

            if (producerIt == uses.producers.end())
               continue;

            const auto branchIt = opToBranch.find(producerIt->second);

            if (branchIt != opToBranch.end() ? branchIt->second != branchIdx : producerIt->second >= commonLaunch)
               return false;
         }

         for (const auto &outputName : branch.outputTensors) {
            const auto consumerIt = uses.consumers.find(outputName);

            if (consumerIt == uses.consumers.end())
               continue;

            for (const size_t consumerOpIdx : consumerIt->second) {
               const auto branchIt = opToBranch.find(consumerOpIdx);

               if (branchIt != opToBranch.end()) {
                  if (branchIt->second != branchIdx)
                     return false;
               } else if (consumerOpIdx < commonLaunch) {
                  return false;
               }
            }
         }
      }

      launchOpIndex = commonLaunch;
      return true;
   }

   std::vector<Fusion::KernelGroup> EnumerateKernelGroups(const std::vector<Group> &units) const
   {
      constexpr size_t maxBranches = 2;
      constexpr size_t maxLookaheadUnits = 8;

      struct Window {
         size_t earliest = 0;
         size_t latest = 0;
         bool valid = false;
      };

      std::vector<Window> windows(units.size());

      for (size_t unitIdx = 0; unitIdx < units.size(); ++unitIdx) {
         const auto &unit = units[unitIdx];
         auto &window = windows[unitIdx];

         if (unit.opIndices.empty() || unit.outputTensors.empty() || m.fOperators.empty())
            continue;

         window.latest = m.fOperators.size() - 1;

         for (const auto &input : unit.externalInputs) {
            const auto producerIt = uses.producers.find(input.tensorName);

            if (producerIt != uses.producers.end())
               window.earliest = std::max(window.earliest, producerIt->second + 1);
         }

         for (const auto &outputName : unit.outputTensors) {
            const auto consumerIt = uses.consumers.find(outputName);

            if (consumerIt == uses.consumers.end())
               continue;

            for (const size_t consumerOpIdx : consumerIt->second) {
               if (std::find(unit.opIndices.begin(), unit.opIndices.end(), consumerOpIdx) == unit.opIndices.end())
                  window.latest = std::min(window.latest, consumerOpIdx);
            }
         }

         window.valid = window.earliest <= window.latest;
      }

      const auto inWindow = [&](size_t unitIdx, size_t launch) {
         return windows[unitIdx].valid && launch >= windows[unitIdx].earliest && launch <= windows[unitIdx].latest;
      };

      std::vector<Fusion::KernelGroup> groups;
      std::set<std::vector<size_t>> emitted;

      for (size_t seedIdx = 0; seedIdx < units.size(); ++seedIdx) {
         if (!windows[seedIdx].valid)
            continue;

         const size_t endIdx = std::min(units.size(), seedIdx + maxLookaheadUnits);
         std::vector<std::vector<size_t>> pending{{seedIdx}};

         while (!pending.empty()) {
            std::vector<size_t> unitIndices = std::move(pending.back());
            pending.pop_back();

            const size_t commonLaunch = units[unitIndices.back()].launchOpIndex;

            if (!std::all_of(unitIndices.begin(), unitIndices.end(),
                             [&](size_t unitIdx) { return inWindow(unitIdx, commonLaunch); }))
               continue;

            if (unitIndices.size() >= 2) {
               std::vector<Group> branches;

               for (const size_t unitIdx : unitIndices)
                  branches.push_back(units[unitIdx]);

               size_t launchOpIndex = 0;

               if (!CanHorizontallyFuse(branches, launchOpIndex))
                  continue;

               if (emitted.insert(unitIndices).second) {
                  Fusion::KernelGroup group;
                  group.unitIndices = unitIndices;
                  group.branches = std::move(branches);
                  group.launchOpIndex = launchOpIndex;

                  for (const auto &branch : group.branches)
                     group.numElements = std::max(group.numElements, branch.numElements);

                  groups.push_back(std::move(group));
               }
            }

            if (unitIndices.size() >= maxBranches)
               continue;

            for (size_t nextIdx = unitIndices.back() + 1; nextIdx < endIdx; ++nextIdx) {
               if (!windows[nextIdx].valid)
                  continue;

               const size_t nextLaunch = units[nextIdx].launchOpIndex;

               if (!std::all_of(unitIndices.begin(), unitIndices.end(), [&](size_t unitIdx) {
                      return nextLaunch >= windows[unitIdx].earliest && nextLaunch <= windows[unitIdx].latest;
                   }))
                  continue;

               std::vector<size_t> next = unitIndices;
               next.push_back(nextIdx);
               pending.push_back(std::move(next));
            }
         }
      }

      return groups;
   }

   std::vector<Fusion::KernelGroup> SelectKernelGroups(std::vector<Fusion::KernelGroup> groups) const
   {
      std::vector<size_t> costs(groups.size());

      for (size_t idx = 0; idx < groups.size(); ++idx) {
         std::vector<std::pair<std::string, size_t>> inputs;

         for (const auto &branch : groups[idx].branches) {
            for (const auto &input : branch.externalInputs)
               inputs.emplace_back(input.tensorName, groups[idx].launchOpIndex);
         }

         costs[idx] = ExtensionBytes(inputs);
      }

      std::vector<size_t> order(groups.size());
      for (size_t idx = 0; idx < order.size(); ++idx)
         order[idx] = idx;

      std::sort(order.begin(), order.end(), [&](size_t leftIdx, size_t rightIdx) {
         const auto &left = groups[leftIdx];
         const auto &right = groups[rightIdx];

         if (left.branches.size() != right.branches.size())
            return left.branches.size() > right.branches.size();

         if (costs[leftIdx] != costs[rightIdx])
            return costs[leftIdx] < costs[rightIdx];

         if (left.numElements != right.numElements)
            return left.numElements < right.numElements;

         return left.unitIndices < right.unitIndices;
      });

      std::set<size_t> usedUnits;
      std::vector<Fusion::KernelGroup> selected;

      for (const size_t idx : order) {
         const auto &group = groups[idx];

         if (std::any_of(group.unitIndices.begin(), group.unitIndices.end(),
                         [&](size_t unitIdx) { return usedUnits.count(unitIdx) != 0; }))
            continue;

         selected.push_back(group);
         usedUnits.insert(group.unitIndices.begin(), group.unitIndices.end());
      }

      std::sort(selected.begin(), selected.end(), [](const Fusion::KernelGroup &left, const Fusion::KernelGroup &right) {
         return left.launchOpIndex < right.launchOpIndex;
      });

      return selected;
   }
};

Fusion::Plan Fusion::Compute(RModel &model)
{
   return FusionPlanner(model).Run();
}

namespace {

const std::string SP = "   ";

std::string DimLiteral(const Dim &d)
{
   return d.isParam ? d.GetVal() : (d.GetVal() + "u");
}

void CollectKnownShapeExprParams(const std::string &expr, const std::unordered_map<std::string, std::string> &known,
                                 std::vector<std::string> &out, std::set<std::string> &seen)
{
   size_t i = 0;
   while (i < expr.size()) {
      if (std::isalpha(static_cast<unsigned char>(expr[i])) || expr[i] == '_') {
         size_t j = i;
         while (j < expr.size() && (std::isalnum(static_cast<unsigned char>(expr[j])) || expr[j] == '_'))
            ++j;
         std::string token = expr.substr(i, j - i);
         if (known.count(token) && seen.insert(token).second)
            out.push_back(token);
         i = j;
      } else {
         ++i;
      }
   }
}

std::string Profiled(const std::string &launchCode, const std::string &title, const std::string &name)
{
   std::string code;
   code += "   // -- GPU Profiling " + title + ": " + name + " --\n";
   code += "   tp_start = std::chrono::steady_clock::now();\n";
   code += launchCode;
   code += "   alpaka::wait(queue);\n";
   code += "   fProfilingResults[\"" + name + "\"].push_back(\n";
   code += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   code += "         std::chrono::steady_clock::now() - tp_start).count());\n\n";
   return code;
}

std::string Pointers(const Fusion::Group &group)
{
   std::string code;
   for (const auto &input : group.externalInputs)
      code += ", alpaka::getPtrNative(deviceBuf_" + input.tensorName + ")";
   for (const auto &outputName : group.outputTensors)
      code += ", alpaka::getPtrNative(deviceBuf_" + outputName + ")";
   return code;
}

std::string BroadcastIndex(const std::vector<Dim> &alignedStrides, const std::vector<Dim> &outputShape,
                           const std::string &index)
{
   const auto outputStrides = UTILITY::ComputeStrideFromShape(outputShape);
   std::string expression;

   for (size_t dimIdx = 0; dimIdx < alignedStrides.size(); ++dimIdx) {
      const Dim &inputStride = alignedStrides[dimIdx];

      if (inputStride.GetVal() == "0")
         continue;

      if (!expression.empty())
         expression += " + ";

      if (outputStrides[dimIdx].GetVal() == "1")
         expression += "(" + index + " % " + DimLiteral(outputShape[dimIdx]) + ")";
      else
         expression += "((" + index + " / " + DimLiteral(outputStrides[dimIdx]) + ") % " + DimLiteral(outputShape[dimIdx]) + ")";

      if (inputStride.GetVal() != "1")
         expression += " * " + DimLiteral(inputStride);
   }

   return expression.empty() ? "0" : expression;
}

std::string KernelSignature(const Fusion::Group &group, const std::string &tail)
{
   std::string code = SP + "template<typename TAcc";

   for (size_t i = 0; i < group.externalInputs.size(); ++i)
      code += ", typename TInput" + std::to_string(i);
   for (size_t i = 0; i < group.outputTensors.size(); ++i)
      code += ", typename TOutput" + std::to_string(i);

   code += ">\n";
   code += SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc";

   for (size_t i = 0; i < group.externalInputs.size(); ++i)
      code += ", TInput" + std::to_string(i) + " const* __restrict__ input" + std::to_string(i);
   for (size_t i = 0; i < group.outputTensors.size(); ++i)
      code += ", TOutput" + std::to_string(i) + "* __restrict__ out" + std::to_string(i);

   return code + tail + ") const {\n";
}

} // anonymous

class FusionCodegen {
public:
   explicit FusionCodegen(const RModel &model) : m(model) {}

   std::string Launch(const Fusion::Group &group) const
   {
      const std::string suffix = group.suffix();
      const std::string kernelName = "fusedEltwiseKernel" + suffix;
      std::string code;

      if (const auto reductionOp = ReductionOp(group)) {
         const auto outputs = m.fOperators[*reductionOp]->GetOpOutputTensors();
         const auto inputs = m.fOperators[*reductionOp]->GetOpInputTensors();
         const auto dataInputs = m.fOperators[*reductionOp]->GetFusionDataInputIndices();

         if (outputs.size() != 1)
            throw std::runtime_error("Fused reduction must have exactly one output");
         if (dataInputs.size() != 1)
            throw std::runtime_error("Fused reduction must have exactly one data input");

         const size_t inputLength = ConvertShapeToLength(m.GetTensorShape(std::string(inputs[dataInputs[0]])));
         const size_t outputLength = ConvertShapeToLength(m.GetTensorShape(std::string(outputs[0])));

         if (outputLength == 0 || inputLength % outputLength != 0)
            throw std::runtime_error("Invalid fused reduction shape");

         code += "\n//------ FUSED_REDUCTION_GPU_ALPAKA" + suffix + "\n";
         code += SP + "{\n";
         code += SP + SP + "alpaka::WorkDivMembers<Dim, Idx> workDiv_fused" + suffix + "(\n";
         code += SP + SP + SP + "Vec::all(Idx{" + std::to_string(outputLength) + "u}),\n";
         code += SP + SP + SP + "Vec::all(Idx{" + std::to_string(BlockSize(inputLength / outputLength)) + "u}),\n";
         code += SP + SP + SP + "Vec::all(Idx{1u}));\n";
         code += SP + SP + "auto task_fused" + suffix + " = alpaka::createTaskKernel<Acc>(workDiv_fused" + suffix + ", " + kernelName;
         code += Pointers(group) + ");\n";
         code += SP + SP + "alpaka::enqueue(queue, task_fused" + suffix + ");\n";
         code += SP + "}\n";

         return m.fProfile ? Profiled(code, "fused reduction group", "FusedReduction" + suffix) : code;
      }

      code += "\n//------ FUSED_ELTWISE_GPU_ALPAKA" + suffix + "\n";
      code += SP + "{\n";
      code += SP + SP + "auto const elementsPerThread_fused" + suffix + " = Vec::all(static_cast<Idx>(1));\n";
      code += SP + SP + "auto const elementsPerGrid_fused" + suffix + " = Vec::all(Idx{" + group.numElements + "});\n";
      code += SP + SP + "auto const workDiv_fused" + suffix + " = sofie_workdiv(elementsPerGrid_fused" + suffix + ");\n";
      code += SP + SP + "auto task_fused" + suffix + " = alpaka::createTaskKernel<Acc>(workDiv_fused" + suffix + ", " + kernelName;
      code += Pointers(group);

      for (const auto &param : DynParams({&group, 1}))
         code += ", static_cast<std::size_t>(" + param + ")";

      code += ", static_cast<Idx>(" + group.numElements + "));\n";
      code += SP + SP + "alpaka::enqueue(queue, task_fused" + suffix + ");\n";
      code += SP + "}\n";

      return m.fProfile ? Profiled(code, "fused group", "FusedKernel" + suffix) : code;
   }

   std::string Launch(const Fusion::KernelGroup &group) const
   {
      const std::string suffix = group.suffix();
      std::string code;

      code += "\n//------ KERNEL_FUSION_GPU_ALPAKA" + suffix + "\n";
      code += SP + "{\n";
      code += SP + SP + "auto const elementsPerGrid_kernelFusion" + suffix + " = Vec::all(Idx{" + group.numElements + "});\n";
      code += SP + SP + "auto const workDiv_kernelFusion" + suffix + " = sofie_workdiv(elementsPerGrid_kernelFusion" + suffix + ");\n";
      code += SP + SP + "auto task_kernelFusion" + suffix + " = alpaka::createTaskKernel<Acc>(workDiv_kernelFusion" + suffix + ", kernelFusionKernel" + suffix;

      for (const auto &branch : group.branches)
         code += Pointers(branch);

      for (const auto &param : DynParams(group.branches))
         code += ", static_cast<std::size_t>(" + param + ")";

      code += ");\n";
      code += SP + SP + "alpaka::enqueue(queue, task_kernelFusion" + suffix + ");\n";
      code += SP + "}\n";

      return m.fProfile ? Profiled(code, "horizontal fusion group", "KernelFusion" + suffix) : code;
   }

   std::string Kernel(const Fusion::Group &group) const
   {
      if (const auto reductionOp = ReductionOp(group))
         return ReductionKernel(group, *reductionOp);

      const std::string suffix = group.suffix();
      std::string tail;

      for (const auto &param : DynParams({&group, 1}))
         tail += ", std::size_t const " + param;

      tail += ", std::size_t n";

      std::string code;
      code += "\n//------ FUSED_ELTWISE_KERNEL" + suffix + "\n";
      code += "struct FusedEltwiseKernel" + suffix + " {\n";
      code += KernelSignature(group, tail);
      code += SP + SP + "using T = TOutput0;\n";
      code += SP + SP + "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      code += SP + SP + "if (idx < n) {\n";

      Context ctx = MakeContext(group, false);
      std::unordered_map<std::string, std::string> cache;

      for (size_t outputIdx = 0; outputIdx < group.outputTensors.size(); ++outputIdx) {
         const std::string value = Value(ctx, group.outputTensors[outputIdx], "idx", cache, code);
         code += SP + SP + SP + "out" + std::to_string(outputIdx) + "[idx] = " + value + ";\n";
      }

      code += SP + SP + "}\n";
      code += SP + "}\n";
      code += "};\n";
      return code;
   }

   std::string Kernel(const Fusion::KernelGroup &group) const
   {
      const std::string suffix = group.suffix();
      std::string code;

      code += "\n//------ KERNEL_FUSION_KERNEL" + suffix + "\n";
      code += "struct KernelFusionKernel" + suffix + " {\n";
      code += SP + "template<typename TAcc";

      for (size_t branchIdx = 0; branchIdx < group.branches.size(); ++branchIdx) {
         const auto &branch = group.branches[branchIdx];
         const std::string prefix = std::to_string(branchIdx) + "_";

         for (size_t i = 0; i < branch.externalInputs.size(); ++i)
            code += ", typename TInput" + prefix + std::to_string(i);
         for (size_t i = 0; i < branch.outputTensors.size(); ++i)
            code += ", typename TOutput" + prefix + std::to_string(i);
      }

      code += ">\n";
      code += SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc";

      for (size_t branchIdx = 0; branchIdx < group.branches.size(); ++branchIdx) {
         const auto &branch = group.branches[branchIdx];
         const std::string prefix = std::to_string(branchIdx) + "_";

         for (size_t i = 0; i < branch.externalInputs.size(); ++i)
            code += ", TInput" + prefix + std::to_string(i) + " const* __restrict__ input" + prefix + std::to_string(i);
         for (size_t i = 0; i < branch.outputTensors.size(); ++i)
            code += ", TOutput" + prefix + std::to_string(i) + "* __restrict__ out" + prefix + std::to_string(i);
      }

      for (const auto &param : DynParams(group.branches))
         code += ", std::size_t const " + param;

      code += ") const {\n";
      code += SP + SP + "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";

      for (size_t branchIdx = 0; branchIdx < group.branches.size(); ++branchIdx) {
         const auto &branch = group.branches[branchIdx];
         const std::string prefix = std::to_string(branchIdx) + "_";
         const auto outputShape = m.GetDimTensorShape(branch.outputTensors.front());

         code += "\n";
         code += SP + SP + "if (idx < " + branch.numElements + ") {\n";
         code += SP + SP + SP + "using T = TOutput" + prefix + "0;\n";

         std::unordered_map<std::string, std::string> values;

         for (size_t inputIdx = 0; inputIdx < branch.externalInputs.size(); ++inputIdx) {
            const auto &input = branch.externalInputs[inputIdx];
            const std::string localName = "v_input_" + prefix + std::to_string(inputIdx);
            std::string index;

            if (!input.customIndexExpression.empty())
               index = input.customIndexExpression;
            else if (input.access == Fusion::Access::Elementwise)
               index = "idx";
            else if (input.access == Fusion::Access::Scalar)
               index = "0";
            else
               index = BroadcastIndex(input.alignedStrides, outputShape, "idx");

            code += SP + SP + SP + "auto " + localName + " = input" + prefix + std::to_string(inputIdx) + "[" + index + "];\n";
            values[input.tensorName] = localName;
         }

         for (const size_t opIdx : branch.opIndices) {
            const auto &op = *m.fOperators[opIdx];
            const auto opInputs = op.GetOpInputTensors();
            const auto outputs = op.GetOpOutputTensors();
            std::vector<std::string> inputExpressions;

            for (const size_t inputIdx : op.GetFusionDataInputIndices()) {
               const auto valueIt = values.find(std::string(opInputs[inputIdx]));

               if (valueIt == values.end())
                  throw std::runtime_error("Missing horizontal fused value for tensor " + std::string(opInputs[inputIdx]));

               inputExpressions.push_back(valueIt->second);
            }

            const std::string expression = op.GetFusionExpr(inputExpressions);

            if (expression.empty())
               throw std::runtime_error("Operator " + std::to_string(opIdx) + " does not provide a horizontal fused expression");

            if (outputs.size() != 1)
               throw std::runtime_error("Horizontally fused operator " + std::to_string(opIdx) + " must have exactly one output");

            const std::string localName = "v_op_" + std::to_string(opIdx);
            code += SP + SP + SP + "auto " + localName + " = " + expression + ";\n";
            values[std::string(outputs[0])] = localName;
         }

         for (size_t outputIdx = 0; outputIdx < branch.outputTensors.size(); ++outputIdx) {
            const auto valueIt = values.find(branch.outputTensors[outputIdx]);

            if (valueIt == values.end())
               throw std::runtime_error("Missing horizontal fused output value for tensor " + branch.outputTensors[outputIdx]);

            code += SP + SP + SP + "out" + prefix + std::to_string(outputIdx) + "[idx] = " + valueIt->second + ";\n";
         }

         code += SP + SP + "}\n";
      }

      code += SP + "}\n";
      code += "};\n";
      return code;
   }

private:
   struct Context {
      std::unordered_map<std::string, size_t> producers;
      std::unordered_map<std::string, size_t> externalIndices;
      const std::unordered_map<std::string, std::string> *overrides = nullptr;
      size_t counter = 0;
   };

   const RModel &m;

   static size_t BlockSize(size_t reducedLength)
   {
      size_t blockSize = 32;
      while (blockSize < reducedLength && blockSize < 256)
         blockSize *= 2;
      return blockSize;
   }

   std::optional<size_t> ReductionOp(const Fusion::Group &group) const
   {
      for (const size_t opIdx : group.opIndices) {
         if (m.fOperators[opIdx]->IsFusionReduction())
            return opIdx;
      }

      return std::nullopt;
   }

   std::vector<std::string> DynParams(std::span<const Fusion::Group> groups) const
   {
      std::vector<std::string> params;
      std::set<std::string> seen;

      const auto collect = [&](const std::string &name) {
         for (const auto &dim : m.GetDimTensorShape(name)) {
            if (dim.isParam)
               CollectKnownShapeExprParams(dim.param, m.fShapeParams, params, seen);
         }
      };

      for (const auto &group : groups) {
         for (const auto &name : group.outputTensors)
            collect(name);
         for (const auto &name : group.internalTensors)
            collect(name);
         for (const auto &input : group.externalInputs)
            collect(input.tensorName);
      }

      return params;
   }

   Context MakeContext(const Fusion::Group &group, bool singleOutput) const
   {
      Context ctx;

      for (const size_t opIdx : group.opIndices) {
         const auto outputs = m.fOperators[opIdx]->GetOpOutputTensors();

         if (singleOutput ? outputs.size() != 1 : outputs.empty())
            throw std::runtime_error("Fused operator " + std::to_string(opIdx) +
                                     (singleOutput ? " must have exactly one output" : " has no outputs"));

         for (const auto &output : outputs)
            ctx.producers[std::string(output)] = opIdx;
      }

      for (size_t inputIdx = 0; inputIdx < group.externalInputs.size(); ++inputIdx)
         ctx.externalIndices[group.externalInputs[inputIdx].tensorName] = inputIdx;

      return ctx;
   }

   std::string InputIndex(const std::string &inputName, const std::vector<Dim> &outputShape, const std::string &outputIndex) const
   {
      Fusion::Access access;
      std::vector<Dim> alignedStrides;

      if (!Fusion::ResolveAccess(m, inputName, outputShape, access, alignedStrides))
         throw std::runtime_error("Cannot resolve fused input index for tensor " + inputName);

      if (access == Fusion::Access::Elementwise)
         return outputIndex;

      if (access == Fusion::Access::Scalar)
         return "0";

      return BroadcastIndex(alignedStrides, outputShape, "(" + outputIndex + ")");
   }

   std::string Value(Context &ctx, const std::string &tensorName, const std::string &index,
                     std::unordered_map<std::string, std::string> &cache, std::string &code) const
   {
      if (ctx.overrides != nullptr) {
         const auto overrideIt = ctx.overrides->find(tensorName);
         if (overrideIt != ctx.overrides->end())
            return overrideIt->second;
      }

      const std::string cacheKey = tensorName + "@" + index;
      const auto cacheIt = cache.find(cacheKey);

      if (cacheIt != cache.end())
         return cacheIt->second;

      const auto externalIt = ctx.externalIndices.find(tensorName);

      if (externalIt != ctx.externalIndices.end()) {
         const std::string localName = "v_input_" + std::to_string(ctx.counter++);
         code += SP + SP + SP + "auto " + localName + " = input" + std::to_string(externalIt->second) + "[" + index + "];\n";
         cache[cacheKey] = localName;
         return localName;
      }

      const auto producerIt = ctx.producers.find(tensorName);

      if (producerIt == ctx.producers.end())
         throw std::runtime_error("Missing fused producer for tensor " + tensorName);

      const size_t opIdx = producerIt->second;
      const auto &op = *m.fOperators[opIdx];
      const auto outputs = op.GetOpOutputTensors();

      const auto outputIt = std::find_if(outputs.begin(), outputs.end(), [&](const auto &output) {
         return std::string(output) == tensorName;
      });

      if (outputIt == outputs.end())
         throw std::runtime_error("Invalid fused producer for tensor " + tensorName);

      const size_t outputTensorIndex = static_cast<size_t>(std::distance(outputs.begin(), outputIt));
      const auto outputShape = m.GetDimTensorShape(tensorName);
      const auto opInputs = op.GetOpInputTensors();
      const auto dataInputs = op.GetFusionDataInputIndices();
      const auto mapping = op.GetFusionMappingType();

      if (mapping == EFusionMappingType::ManyToMany) {
         const std::string localName = "v_op_" + std::to_string(opIdx) + "_" + std::to_string(ctx.counter++);
         code += SP + SP + SP + ConvertTypeToString(m.GetTensorType(tensorName)) + " " + localName + "{};\n";

         for (size_t dataIdx = 0; dataIdx < dataInputs.size(); ++dataIdx) {
            const size_t inputIdx = dataInputs[dataIdx];
            const std::string inputName(opInputs[inputIdx]);
            const auto inputShape = m.GetDimTensorShape(inputName);
            const std::string inputIndex = op.GetFusionInputIndexExpr(inputIdx, index, inputShape, outputShape);

            if (inputIndex.empty())
               throw std::runtime_error("Missing ManyToMany index expression for operator " + std::to_string(opIdx));

            std::string condition;

            if (dataIdx + 1 < dataInputs.size()) {
               condition = op.GetFusionInputConditionExpr(inputIdx, index, inputShape, outputShape);

               if (condition.empty())
                  throw std::runtime_error("Missing ManyToMany input condition for operator " + std::to_string(opIdx));
            }

            auto branchCache = cache;
            std::string branchCode;
            const std::string branchValue = Value(ctx, inputName, inputIndex, branchCache, branchCode);

            if (dataIdx == 0)
               code += SP + SP + SP + "if (" + condition + ") {\n";
            else if (dataIdx + 1 < dataInputs.size())
               code += SP + SP + SP + "else if (" + condition + ") {\n";
            else
               code += SP + SP + SP + "else {\n";

            code += branchCode;
            code += SP + SP + SP + SP + localName + " = " + op.GetFusionExpr({branchValue}) + ";\n";
            code += SP + SP + SP + "}\n";
         }

         cache[cacheKey] = localName;
         return localName;
      }

      std::vector<std::string> inputExpressions;

      for (const size_t inputIdx : dataInputs) {
         const std::string inputName(opInputs[inputIdx]);
         const auto inputShape = m.GetDimTensorShape(inputName);
         std::string inputIndex;

         if (mapping == EFusionMappingType::Shuffle) {
            inputIndex = op.GetFusionInputIndexExpr(inputIdx, index, inputShape, outputShape);

            if (inputIndex.empty())
               throw std::runtime_error("Missing Shuffle index expression for operator " + std::to_string(opIdx));
         } else if (mapping == EFusionMappingType::OneToMany && outputs.size() > 1) {
            inputIndex = op.GetFusionInputIndexExprForOutput(inputIdx, outputTensorIndex, index, inputShape, outputShape);

            if (inputIndex.empty())
               throw std::runtime_error("Missing OneToMany output index expression for operator " + std::to_string(opIdx));
         } else if (mapping == EFusionMappingType::Reorganize) {
            if (ConvertDimShapeToLength(inputShape) != ConvertDimShapeToLength(outputShape))
               throw std::runtime_error("Invalid Reorganize mapping for operator " + std::to_string(opIdx));

            inputIndex = index;
         } else {
            inputIndex = InputIndex(inputName, outputShape, index);
         }

         inputExpressions.push_back(Value(ctx, inputName, inputIndex, cache, code));
      }

      const std::string expression = op.GetFusionExpr(inputExpressions);

      if (expression.empty())
         throw std::runtime_error("Operator " + std::to_string(opIdx) + " does not provide a fused expression");

      const std::string localName = "v_op_" + std::to_string(opIdx) + "_" + std::to_string(ctx.counter++);
      code += SP + SP + SP + "auto " + localName + " = " + expression + ";\n";
      cache[cacheKey] = localName;

      return localName;
   }

   std::string ReductionKernel(const Fusion::Group &group, size_t reductionOpIdx) const
   {
      const auto &op = *m.fOperators[reductionOpIdx];
      const auto inputs = op.GetOpInputTensors();
      const auto outputs = op.GetOpOutputTensors();
      const auto dataInputs = op.GetFusionDataInputIndices();

      if (dataInputs.size() != 1 || outputs.size() != 1)
         throw std::runtime_error("Fused reduction must have one data input and one output");

      const std::string inputName(inputs[dataInputs[0]]);
      const std::string outputName(outputs[0]);
      const auto inputShape = m.GetDimTensorShape(inputName);
      const auto outputShape = m.GetDimTensorShape(outputName);
      const size_t inputLength = ConvertShapeToLength(m.GetTensorShape(inputName));
      const size_t outputLength = ConvertShapeToLength(m.GetTensorShape(outputName));

      if (outputLength == 0 || inputLength % outputLength != 0)
         throw std::runtime_error("Invalid fused reduction shape");

      const size_t reducedLength = inputLength / outputLength;
      const size_t blockSize = BlockSize(reducedLength);
      const std::string inputIndexExpression = op.GetFusionReductionInputIndexExpr("out_idx", "r", inputShape, outputShape);

      if (inputIndexExpression.empty())
         throw std::runtime_error("Fused reduction does not provide an input index expression");

      Context ctx = MakeContext(group, true);
      const std::string suffix = group.suffix();
      const std::string reducedLengthStr = std::to_string(reducedLength) + "u";
      const std::string blockSizeStr = std::to_string(blockSize) + "u";
      std::string code;

      code += "\n//------ FUSED_REDUCTION_KERNEL" + suffix + "\n";
      code += "struct FusedEltwiseKernel" + suffix + " {\n";
      code += KernelSignature(group, "");
      code += SP + SP + "using T = TOutput0;\n";
      code += SP + SP + "auto& shmem = alpaka::declareSharedVar<T[" + std::to_string(blockSize) + "], __COUNTER__>(acc);\n";
      code += SP + SP + "const auto out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
      code += SP + SP + "const auto thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      code += SP + SP + "if (out_idx >= " + std::to_string(outputLength) + "u) return;\n";
      code += SP + SP + "T partial = " + op.GetFusionReductionInitExpr() + ";\n";
      code += SP + SP + "for (std::size_t r = thread_id; r < " + reducedLengthStr + "; r += " + blockSizeStr + ") {\n";
      code += SP + SP + SP + "const std::size_t in_idx = " + inputIndexExpression + ";\n";

      std::unordered_map<std::string, std::string> inputCache;
      std::string inputCode;
      const std::string inputValue = Value(ctx, inputName, "in_idx", inputCache, inputCode);

      code += inputCode;
      code += SP + SP + SP + "partial = " + op.GetFusionReductionAccumulateExpr("partial", inputValue) + ";\n";
      code += SP + SP + "}\n";
      code += SP + SP + "shmem[thread_id] = partial;\n";
      code += SP + SP + "alpaka::syncBlockThreads(acc);\n";
      code += SP + SP + "for (std::size_t s = " + std::to_string(blockSize / 2) + "u; s > 0u; s >>= 1u) {\n";
      code += SP + SP + SP + "if (thread_id < s) shmem[thread_id] = " +
              op.GetFusionReductionCombineExpr("shmem[thread_id]", "shmem[thread_id + s]") + ";\n";
      code += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
      code += SP + SP + "}\n";
      code += SP + SP + "if (thread_id == 0u) shmem[0] = " +
              op.GetFusionReductionFinalizeExpr("shmem[0]", std::to_string(reducedLength)) + ";\n";
      code += SP + SP + "alpaka::syncBlockThreads(acc);\n";
      code += SP + SP + "const T reduction_value = shmem[0];\n";

      const std::unordered_map<std::string, std::string> overrides{{outputName, "reduction_value"}};
      ctx.overrides = &overrides;

      for (size_t outputIdx = 0; outputIdx < group.outputTensors.size(); ++outputIdx) {
         const std::string &name = group.outputTensors[outputIdx];
         const auto shape = m.GetDimTensorShape(name);
         const bool reduced = shape == outputShape;

         if (!reduced && shape != inputShape)
            throw std::runtime_error("Fused reduction output must match the reduction input or output shape");

         std::unordered_map<std::string, std::string> cache;
         std::string valueCode;
         const std::string value = Value(ctx, name, reduced ? "out_idx" : "element_idx", cache, valueCode);

         if (reduced) {
            code += SP + SP + "if (thread_id == 0u) {\n";
            code += valueCode;
            code += SP + SP + SP + "out" + std::to_string(outputIdx) + "[out_idx] = " + value + ";\n";
            code += SP + SP + "}\n";
            continue;
         }

         code += SP + SP + "for (std::size_t r = thread_id; r < " + reducedLengthStr + "; r += " + blockSizeStr + ") {\n";
         code += SP + SP + SP + "const std::size_t element_idx = " + inputIndexExpression + ";\n";
         code += valueCode;
         code += SP + SP + SP + "out" + std::to_string(outputIdx) + "[element_idx] = " + value + ";\n";
         code += SP + SP + "}\n";
      }

      code += SP + "}\n";
      code += "};\n";
      return code;
   }
};

std::string Fusion::Launch(const RModel &model, const Group &group)
{
   return FusionCodegen(model).Launch(group);
}

std::string Fusion::Launch(const RModel &model, const KernelGroup &group)
{
   return FusionCodegen(model).Launch(group);
}

std::string Fusion::Kernel(const RModel &model, const Group &group)
{
   return FusionCodegen(model).Kernel(group);
}

std::string Fusion::Kernel(const RModel &model, const KernelGroup &group)
{
   return FusionCodegen(model).Kernel(group);
}

} // namespace SOFIE
