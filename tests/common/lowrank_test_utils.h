#pragma once

// Dimensions of the Gemm model of the low rank factorization tests and the input they run it on, shared by
// models/LowRankModelGenerator.cxx, cpu/TestLowRankFactorization.cxx and
// alpaka/TestAlpakaLowRank.cxx.

#include <cstddef>
#include <random>
#include <vector>

constexpr size_t kM = 4;     // batch
constexpr size_t kK = 64;    // in features
constexpr size_t kN = 32;    // out features
constexpr size_t kRank = 16; // rank of the factorization

inline std::vector<float> MakeInput()
{
   std::mt19937 rng(7);
   std::normal_distribution<float> dist(0.f, 1.f);
   std::vector<float> x(kM * kK);
   for (auto &v : x) v = dist(rng);
   return x;
}
