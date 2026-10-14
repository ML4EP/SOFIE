#ifndef SOFIE_RWORKSPACE_HXX
#define SOFIE_RWORKSPACE_HXX

#include "SOFIE/SOFIE_common.hxx"
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace SOFIE {
inline std::size_t WorkspaceProduct(std::initializer_list<std::size_t> dimensions)
{
   std::size_t result = 1;
   for (auto dimension : dimensions) {
      if (dimension && result > std::numeric_limits<std::size_t>::max() / dimension)
         throw std::overflow_error("SOFIE workspace tensor size overflow");
      result *= dimension;
   }
   return result;
}

inline MemoryResult PlanWorkspace(const std::vector<TensorLifeInfo> &tensors)
{
   std::size_t upperBound = 0;
   for (const auto &tensor : tensors) {
      if (tensor.size > std::numeric_limits<std::size_t>::max() - upperBound)
         throw std::overflow_error("SOFIE workspace size overflow");
      upperBound += tensor.size;
   }
   return OrganizeMemory(tensors);
}
}
#endif
