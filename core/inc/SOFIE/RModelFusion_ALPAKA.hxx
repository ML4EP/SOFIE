#ifndef SOFIE_RMODELFUSION_ALPAKA
#define SOFIE_RMODELFUSION_ALPAKA

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include "SOFIE/SOFIE_common.hxx"

namespace SOFIE {

class RModel;

namespace Fusion {

enum class Access { Elementwise, Scalar, Broadcast };

struct Input {
   std::string tensorName;
   Access access = Access::Elementwise;
   std::vector<Dim> alignedStrides;
   std::string customIndexExpression;
};

struct Group {
   std::vector<size_t> opIndices;
   std::vector<Input> externalInputs;
   std::vector<std::string> outputTensors;
   std::vector<std::string> internalTensors;
   std::string numElements = "0";
   size_t launchOpIndex = 0;
   bool usesIndexedEvaluation = false;

   std::string suffix() const
   {
      std::string s;
      for (auto i : opIndices)
         s += "_" + std::to_string(i);
      return s;
   }
};

struct KernelGroup {
   std::vector<size_t> unitIndices;
   std::vector<Group> branches;
   std::string numElements = "0";
   size_t launchOpIndex = 0;

   std::string suffix() const
   {
      std::string s;
      for (const auto &branch : branches)
         s += branch.suffix();
      return s;
   }
};

struct Plan {
   std::vector<Group> eltwise;
   std::vector<KernelGroup> kernel;
   std::set<size_t> skip;
   std::set<std::string> internal;

   const Group *FindEltwise(size_t opIdx) const
   {
      for (const auto &group : eltwise) {
         if (std::find(group.opIndices.begin(), group.opIndices.end(), opIdx) != group.opIndices.end())
            return &group;
      }
      return nullptr;
   }

   const KernelGroup *FindKernel(size_t opIdx) const
   {
      for (const auto &group : kernel) {
         for (const auto &branch : group.branches) {
            if (std::find(branch.opIndices.begin(), branch.opIndices.end(), opIdx) != branch.opIndices.end())
               return &group;
         }
      }
      return nullptr;
   }
};

Plan Compute(RModel &model);

bool ResolveAccess(const RModel &model, const std::string &tensorName, const std::vector<Dim> &outputShape,
                   Access &access, std::vector<Dim> &alignedStrides);

std::string Launch(const RModel &model, const Group &group);
std::string Launch(const RModel &model, const KernelGroup &group);
std::string Kernel(const RModel &model, const Group &group);
std::string Kernel(const RModel &model, const KernelGroup &group);

} // namespace Fusion

} // namespace SOFIE

#endif
