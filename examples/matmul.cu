#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>  // For fabsf
#include <cuda_runtime.h>
#include "tinycuda/tinycuda.hpp"  // Bundles Buffer, Profiler, error

/**
 * Simple non-tiled matmul kernel: C = A * B (MxK * KxN -> MxN, row-major flat)
 * Each thread computes one output element (tx, ty) via full dot product.
 * For demo; scales OK up to ~1Kx1K before occupancy drops.
 */
__global__ void matmul(float* A, float* B, float* C, int M, int K, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;  // Output row
    int col = blockIdx.x * blockDim.x + tx;  // Output col

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    constexpr int M = 512;  // Rows of A/C
    constexpr int K = 512;  // Cols of A / Rows of B
    constexpr int N = 512;  // Cols of B/C

    // Host matrices (flat, row-major)
    std::vector<float> host_A(M * K, 1.0f);  // All 1s for easy verify
    std::vector<float> host_B(K * N, 2.0f);  // All 2s
    std::vector<float> host_C(M * N, 0.0f);  // Expected: 1024.0f each (512 * 2)

    // Buffers
    tinycuda::Buffer<float> buf_A(host_A.data(), M * K);
    tinycuda::Buffer<float> buf_B(host_B.data(), K * N);
    tinycuda::Buffer<float> buf_C(host_C.data(), M * N);

    // To GPU
    buf_A.to_gpu();
    buf_B.to_gpu();
    buf_C.to_gpu();

    // Grid/block: 2D for parallelism
    constexpr int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Time it
    tinycuda::KernelProfiler profiler(5, 20);  // Warmup + repeats
    float avg_ms = profiler([&] {
        matmul<<<grid, block>>>(buf_A.gpu_data(), buf_B.gpu_data(), buf_C.gpu_data(), M, K, N);
    });
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Rough GFLOPS (2 ops per MAC)
    double gflops = (2.0 * static_cast<double>(M) * K * N / 1e9) / (avg_ms / 1000.0);
    printf("Matmul (512x512) avg time: %.4f ms (GFLOPS: %.2f)\n", avg_ms, gflops);

    // Back to host
    buf_C.to_cpu();

    // Verify (tolerance for FP)
    bool correct = true;
    for (size_t i = 0; i < M * N; ++i) {
        if (fabsf(host_C[i] - 1024.0f) > 1e-3f) {
            correct = false;
            printf("[FAIL] Element %zu: %f (expected 1024.0)\n", i, host_C[i]);
            break;
        }
    }

    printf(correct ? "[PASS] Matmul verified!\n" : "[FAIL] Verification failed.\n");
    return correct ? 0 : 1;
}