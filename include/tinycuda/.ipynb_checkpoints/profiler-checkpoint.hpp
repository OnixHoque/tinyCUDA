#pragma once
#include <cuda_runtime.h>
#include "tinycuda/error.hpp"

namespace tinycuda {

/**
 * @brief KernelProfiler provides untimed warmup followed by repeated timing for CUDA kernel launches.
 *
 * Example usage:
 * tinycuda::KernelProfiler profile(10, 100);
 * float avg_ms = profile([&] {
 * my_kernel<<<grid, block>>>(args...);
 * });
 * printf("Avg kernel time: %.4f ms\n", avg_ms);
 */
class KernelProfiler {
public:
    /**
     * @brief Construct a KernelProfiler
     * @param warmup Number of warmup runs before timing
     * @param repeat Number of timed runs
     */
    KernelProfiler(int warmup = 5, int repeat = 50)
        : warmup_(warmup), repeat_(repeat) {}

    /**
     * @brief Run a kernel launch and measure average execution time in ms
     *
     * Usage:
     * float avg_ms = profile([&] {
     * kernel<<<grid, block>>>(...);
     * });
     */
    template<typename F>
    float operator()(F&& kernel_launch) {
        // Create CUDA events
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // ---------------------
        // Warmup
        // ---------------------
        for (int i = 0; i < warmup_; ++i) {
            kernel_launch();
        }
        CUDA_CHECK(cudaDeviceSynchronize());  // Ensure warmup complete

        // ---------------------
        // Timed runs
        // ---------------------
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeat_; ++i) {
            kernel_launch();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= static_cast<float>(repeat_);

        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return ms;
    }

private:
    int warmup_;
    int repeat_;
};

} // namespace tinycuda