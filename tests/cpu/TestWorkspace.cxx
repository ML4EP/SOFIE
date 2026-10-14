#include "SOFIE/RWorkspace.hxx"
#include <gtest/gtest.h>

TEST(SOFIEWorkspace, ReusesOnlyDisjointLifetimes)
{
   const std::vector<SOFIE::TensorLifeInfo> tensors{{0, 2, 256}, {1, 3, 512}, {2, 4, 256}};
   const auto plan = SOFIE::PlanWorkspace(tensors);
   ASSERT_EQ(plan.offsets.size(), tensors.size());
   EXPECT_EQ(plan.total_bytes, 768);
   EXPECT_EQ(plan.offsets[0], plan.offsets[2]);
   for (std::size_t first = 0; first < tensors.size(); ++first) {
      EXPECT_EQ(plan.offsets[first] % 256, 0);
      EXPECT_LE(plan.offsets[first] + tensors[first].size, plan.total_bytes);
      for (std::size_t second = first + 1; second < tensors.size(); ++second) {
         if (tensors[first].begin < tensors[second].end && tensors[second].begin < tensors[first].end)
            EXPECT_TRUE(plan.offsets[first] + tensors[first].size <= plan.offsets[second] ||
                        plan.offsets[second] + tensors[second].size <= plan.offsets[first]);
      }
   }
}

TEST(SOFIEWorkspace, RejectsOverflow)
{
   const auto limit = std::numeric_limits<std::size_t>::max();
   EXPECT_THROW(SOFIE::WorkspaceProduct({limit, 2}), std::overflow_error);
   EXPECT_THROW(SOFIE::PlanWorkspace({{0, 2, limit}, {1, 3, 256}}), std::overflow_error);
   EXPECT_EQ(SOFIE::WorkspaceProduct({4, 3, 2}), 24);
   EXPECT_EQ(SOFIE::PlanWorkspace({}).total_bytes, 0);
}
