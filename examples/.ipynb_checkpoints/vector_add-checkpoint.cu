#include <cassert>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "tinycuda/tinycuda.hpp"  // Bundles error, memory, profiler

/**
 * Simple CUDA kernel: C = A + B for each element
 */
__global__ void vector_add(float* a, float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    constexpr size_t N = 1 << 16;  // 65K elements for demo

    // Host vectors
    std::vector<float> host_a(N, 1.0f);
    std::vector<float> host_b(N, 2.0f);
    std::vector<float> host_c(N, 0.0f);  // Expected: 3.0f each

    // Buffers for GPU mirroring
    tinycuda::Buffer<float> buf_a(host_a.data(), N);
    tinycuda::Buffer<float> buf_b(host_b.data(), N);
    tinycuda::Buffer<float> buf_c(host_c.data(), N);

    // Transfer to GPU
    buf_a.to_gpu();
    buf_b.to_gpu();
    buf_c.to_gpu();

    // Grid and block dims
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Optional: Time the kernel with profiler (warmup=3, repeat=10 for quick demo)
    tinycuda::KernelProfiler profiler(3, 10);
    float avg_ms = profiler([&] {
        vector_add<<<grid, block>>>(buf_a.gpu_data(), buf_b.gpu_data(), buf_c.gpu_data(), N);
    });
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Average kernel time: %.4f ms\n", avg_ms);

    // Copy result back to host
    buf_c.to_cpu();

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (fabsf(host_c[i] - 3.0f) > 1e-5f) {  // Tolerance for FP
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("[PASS] Vector addition verified: all elements == 3.0f\n");
    } else {
        printf("[FAIL] Verification failed!\n");
        return 1;
    }

    printf("[SUCCESS] Example completed.\n");
    return 0;
}