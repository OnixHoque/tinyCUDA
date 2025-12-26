#include <cassert>
#include <cstdio>
#include <vector>
#include "tinycuda/tinycuda.hpp"

/**
 * Simple CUDA kernel: increment each element by 1
 */
__global__ void add_one(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

int main() {
    constexpr size_t N = 1 << 16;

    // Host memory
    std::vector<float> host(N, 0.0f);

    // Buffer
    tinycuda::Buffer<float> buf(host.data(), N);
    buf.to_gpu();

    // Grid and block
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // KernelProfiler: warmup 5, repeat 50
    float* d_ptr = buf.gpu_data(); // capture raw device pointer
    tinycuda::KernelProfiler profile(5, 50);

    float avg_ms = profile([d_ptr, N, grid, block] {
        add_one<<<grid, block>>>(d_ptr, N);
    });

    // Ensure all device writes are visible
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    buf.to_cpu();

    // Verify kernel executed correctly
    // Total increments: 5 warmup + 50 timed = 55
    for (size_t i = 0; i < N; ++i) {
        assert(host[i] == 55.0f);
    }

    // Verify profiler returned positive time
    assert(avg_ms > 0.0f);

    printf("[PASS] KernelProfiler test succeeded, avg_ms = %.4f ms\n", avg_ms);
    return 0;
}