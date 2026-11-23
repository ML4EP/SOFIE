#include "SOFIE/quantization/graph/RQuantization_Graph.hxx"

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/quantization/utils/RQuantization_Utils.hxx"

#include <algorithm>
#include <stdexcept>

namespace SOFIE {

// ---- QuantModelAccess -------------------------------------------------------------------

std::size_t QuantModelAccess::OperatorCount() const
{
   return fModel.fOperators.size();
}

ROperator &QuantModelAccess::Operator(std::size_t index)
{
   return *fModel.fOperators.at(index);
}

const std::vector<std::string> &QuantModelAccess::GraphOutputs() const
{
   return fModel.fOutputTensorNames;
}

void QuantModelAccess::RenameInitializer(const std::string &from, const std::string &to)
{
   auto it = fModel.fInitializedTensors.find(from);
   if (it == fModel.fInitializedTensors.end())
      throw std::runtime_error("SOFIE quantization: no initializer " + from + " to rename");
   if (fModel.fInitializedTensors.count(to) != 0)
      throw std::runtime_error("SOFIE quantization: initializer " + to + " already exists");
   auto tensor = std::move(it->second);
   fModel.fInitializedTensors.erase(it);
   fModel.fInitializedTensors.emplace(to, std::move(tensor));
}

bool QuantModelAccess::HasSubGraphs() const
{
   return !fModel.fSubGraphs.empty();
}

bool QuantModelAccess::IsGraphInput(const std::string &tensor) const
{
   const auto &inputs = fModel.fInputTensorNames;
   return std::find(inputs.begin(), inputs.end(), tensor) != inputs.end();
}

void QuantModelAccess::EraseInitializer(const std::string &tensor)
{
   fModel.fInitializedTensors.erase(tensor);
}

void QuantModelAccess::SetTensorType(const std::string &tensor, ETensorType type)
{
   if (auto it = fModel.fIntermediateTensorInfos.find(tensor); it != fModel.fIntermediateTensorInfos.end()) {
      it->second.type = type;
      return;
   }
   if (auto it = fModel.fReadyInputTensorInfos.find(tensor); it != fModel.fReadyInputTensorInfos.end()) {
      it->second.type = type;
      return;
   }
   if (auto it = fModel.fInputTensorInfos.find(tensor); it != fModel.fInputTensorInfos.end()) {
      it->second.type = type;
      return;
   }
   if (auto it = fModel.fDynamicTensorInfos.find(tensor); it != fModel.fDynamicTensorInfos.end())
      it->second.type = type;
}

void QuantModelAccess::AddAlias(const std::string &name, const std::string &origin)
{
   fModel.AddAliasTensor(name, origin);
}

// ---- QuantGraphView ---------------------------------------------------------------------

QuantGraphView::QuantGraphView(QuantModelAccess &access,
                               const std::unordered_map<std::string, TensorQuantState> *states)
   : fAccess(access), fStates(states), fOperatorCount(access.OperatorCount())
{
   for (const auto &output : access.GraphOutputs())
      fGraphOutputs.insert(output);
   for (std::size_t index = 0; index < fOperatorCount; ++index) {
      const ROperator *op = std::as_const(access.Model()).GetEmittedOperatorAt(index);
      if (op == nullptr)
         continue;
      for (const auto &output : op->GetOpOutputTensors())
         fProducers[std::string(output)] = index;
      for (const auto &input : op->GetOpInputTensors()) {
         auto &readers = fConsumers[std::string(input)];
         if (readers.empty() || readers.back() != index)
            readers.push_back(index);
      }
   }
}

std::optional<std::size_t> QuantGraphView::Producer(const std::string &tensor) const
{
   auto it = fProducers.find(tensor);
   if (it == fProducers.end())
      return std::nullopt;
   return it->second;
}

const std::vector<std::size_t> &QuantGraphView::Consumers(const std::string &tensor) const
{
   static const std::vector<std::size_t> kNone;
   auto it = fConsumers.find(tensor);
   return it == fConsumers.end() ? kNone : it->second;
}

bool QuantGraphView::IsConstant(const std::string &tensor) const
{
   return fAccess.Model().IsInitializedTensor(tensor);
}

std::vector<std::size_t> QuantGraphView::Shape(const std::string &tensor) const
{
   return fAccess.Model().GetTensorShape(tensor);
}

const TensorQuantState &QuantGraphView::State(const std::string &tensor) const
{
   static const TensorQuantState kFloat;
   if (fStates == nullptr)
      return kFloat;
   auto it = fStates->find(tensor);
   return it == fStates->end() ? kFloat : it->second;
}

// ---- QuantRewriter ----------------------------------------------------------------------

void QuantRewriter::Replace(std::size_t anchor, std::unique_ptr<ROperator> op)
{
   if (fReplacements.count(anchor) != 0)
      throw std::runtime_error("SOFIE quantization: operator " + std::to_string(anchor) + " replaced twice");
   fReplacements[anchor] = std::move(op);
}

void QuantRewriter::Suppress(std::size_t index)
{
   fSuppressed.insert(index);
}

void QuantRewriter::Mutate(std::size_t index, std::function<void(ROperator &)> edit)
{
   fMutations.emplace_back(index, std::move(edit));
}

void QuantRewriter::RetypeTensor(const std::string &tensor, ETensorType type)
{
   fRetypes.emplace_back(tensor, type);
}

void QuantRewriter::Alias(const std::string &name, const std::string &origin)
{
   fAliases.emplace_back(name, origin);
}

void QuantRewriter::AddConstant(const std::string &name, ETensorType type, std::vector<std::size_t> shape,
                                std::shared_ptr<void> data)
{
   fConstants.push_back({name, type, std::move(shape), std::move(data)});
}

void QuantRewriter::AddScratch(const std::string &name, ETensorType type, std::vector<std::size_t> shape)
{
   fScratch.emplace_back(name, type, std::move(shape));
}

std::vector<std::string> QuantRewriter::ConstantNames() const
{
   std::vector<std::string> names;
   for (const auto &constant : fConstants)
      names.push_back(constant.name);
   return names;
}

void QuantRewriter::Commit(QuantModelAccess &access)
{
   auto &model = access.Model();
   for (auto &constant : fConstants) {
      if (model.IsInitializedTensor(constant.name))
         continue;   // Shared by two lowered operators (e.g. one weight read twice).
      AddFoldedConstant(model, constant.name, constant.type, constant.shape, std::move(constant.data));
   }
   for (auto &[name, type, shape] : fScratch)
      model.AddIntermediateTensor(name, type, shape);
   for (auto &[index, edit] : fMutations)
      edit(access.Operator(index));
   for (auto &[anchor, op] : fReplacements)
      access.InstallLoweredOperator(anchor, std::move(op));
   for (auto index : fSuppressed)
      access.SuppressOperator(index);
   for (const auto &[tensor, type] : fRetypes)
      access.SetTensorType(tensor, type);
   for (const auto &[name, origin] : fAliases)
      access.AddAlias(name, origin);
   Discard();
}

void QuantRewriter::Discard()
{
   fReplacements.clear();
   fSuppressed.clear();
   fMutations.clear();
   fRetypes.clear();
   fAliases.clear();
   fConstants.clear();
   fScratch.clear();
}

} // namespace SOFIE
