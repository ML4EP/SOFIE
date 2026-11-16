#pragma once

#include <chrono>

namespace sofie_bench {

// Default minimum wall-clock warmup duration, applied in addition to
// whatever --warmup iteration count the user passed
constexpr double kMinWarmupDurationMs = 200.0;

// Calls oneCall() repeatedly until at least `minIterations` calls have
// completed AND at least `minDurationMs` of wall-clock time has elapsed
// since the first call. oneCall must itself block until the device work
// it launches has completed otherwise the elapsed time
// measured here would only reflect how fast the host can enqueue work,
// not whether the device itself has finished ramping up.
//
// oneCall returns bool (true = keep going, false = a genuine failure);
// on false, RunWarmup stops immediately and returns false too, so a
// caller that does its own error handling/cleanup on failure 
// can react exactly the way it would have with a
// plain warmup loop.
template <typename Fn>
inline bool RunWarmup(Fn &&oneCall, int minIterations, double minDurationMs = kMinWarmupDurationMs) {
    const auto start = std::chrono::steady_clock::now();
    int count = 0;
    while (true) {
        if (!oneCall())
            return false;
        ++count;
        if (count < minIterations)
            continue;
        const double elapsedMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsedMs >= minDurationMs)
            return true;
    }
}

} // namespace sofie_bench
