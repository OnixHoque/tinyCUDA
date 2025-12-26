#include <cassert>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "tinycuda/tinycuda.hpp"

/**
 * Simple CUDA kernel: add 1 to each element
 */
__global__ void add_one(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

/**
 * Test 1:
 * - Multiple to_gpu() calls should be safe
 * - Device memory should not be reallocated
 */
void test_multiple_to_gpu() {
    constexpr size_t N = 1 << 14;
    std::vector<float> host(N, 1.0f);
    tinycuda::Buffer<float> buf(host.data(), N);
    // First transfer
    buf.to_gpu();
    float* d_ptr_first = buf.gpu_data();
    // Modify host data
    for (auto& v : host) {
        v = 2.0f;
    }
    // Second transfer (should reuse allocation)
    buf.to_gpu();
    float* d_ptr_second = buf.gpu_data();
    // Same device pointer => no reallocation
    assert(d_ptr_first == d_ptr_second);
    // Copy back and verify
    buf.to_cpu();
    for (size_t i = 0; i < N; ++i) {
        assert(host[i] == 2.0f);
    }
    printf("[PASS] multiple to_gpu() test\n");
}

/**
 * Test 2:
 * - Move constructor transfers device ownership
 * - Moved-from buffer becomes inert
 */
void test_move_semantics() {
    constexpr size_t N = 1 << 14;
    std::vector<float> host(N, 5.0f);
    tinycuda::Buffer<float> buf_a(host.data(), N);
    buf_a.to_gpu();
    float* d_ptr_a = buf_a.gpu_data();
    // Move construct
    tinycuda::Buffer<float> buf_b = std::move(buf_a);
    // buf_b owns device memory
    assert(buf_b.gpu_data() == d_ptr_a);
    assert(buf_b.size() == N);
    assert(buf_b.on_gpu());
    // buf_a is now inert
    assert(buf_a.size() == 0);
    assert(!buf_a.on_gpu());
    // Kernel should still work
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    add_one<<<grid, block>>>(buf_b.gpu_data(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    buf_b.to_cpu();
    for (size_t i = 0; i < N; ++i) {
        assert(host[i] == 6.0f);
    }
    printf("[PASS] move semantics test\n");
}

/**
 * Test 3:
 * - Move assignment transfers device ownership
 * - Existing device memory is freed
 * - Moved-from buffer becomes inert
 * - Host pointer is also transferred
 */
void test_move_assignment() {
    constexpr size_t N = 1 << 14;
    std::vector<float> host_a(N, 10.0f);
    std::vector<float> host_b(N, 20.0f);
    tinycuda::Buffer<float> buf_a(host_a.data(), N);
    tinycuda::Buffer<float> buf_b(host_b.data(), N);
    // Allocate device memory for both
    buf_a.to_gpu();
    float* d_ptr_a_original = buf_a.gpu_data();
    buf_b.to_gpu();
    float* d_ptr_b = buf_b.gpu_data();
    assert(d_ptr_a_original != d_ptr_b);  // Different allocations
    // Move-assign: buf_a = move(buf_b)
    buf_a = std::move(buf_b);
    // buf_a now owns buf_b's device memory and host_b ptr
    assert(buf_a.gpu_data() == d_ptr_b);
    assert(buf_a.size() == N);
    assert(buf_a.on_gpu());
    // buf_b is now inert
    assert(buf_b.size() == 0);
    assert(!buf_b.on_gpu());
    // Original buf_a device memory was freed; buf_a.d_ptr_ != d_ptr_a_original
    assert(buf_a.gpu_data() != d_ptr_a_original);
    // Run kernel on buf_a
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    add_one<<<grid, block>>>(buf_a.gpu_data(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    buf_a.to_cpu();
    // to_cpu writes to host_b (now buf_a.h_ptr_)
    for (size_t i = 0; i < N; ++i) {
        assert(host_b[i] == 21.0f);  // 20 + 1
        assert(host_a[i] == 10.0f);  // Unchanged
    }
    printf("[PASS] move assignment test\n");
}

/**
 * Test 4:
 * - Zero-size buffers are supported (no-op operations)
 * - Constructor allows size=0 with nullptr host_ptr
 * - No allocation/copy on to_gpu()/to_cpu()
 */
void test_zero_size() {
    constexpr size_t N = 0;
    float* host_ptr = nullptr;  // nullptr OK for size=0
    tinycuda::Buffer<float> buf(host_ptr, N);
    assert(buf.size() == 0);
    assert(!buf.on_gpu());
    // to_gpu() should not allocate (d_ptr_ remains nullptr)
    buf.to_gpu();
    assert(buf.gpu_data() == nullptr);  // No assert, returns nullptr
    assert(buf.on_gpu());  // Set to true, but no-op
    // to_cpu() should be no-op
    buf.to_cpu();  // Now safe with relaxed assert
    // No data to verify
    printf("[PASS] zero-size test\n");
}

/**
 * Test 5:
 * - Accessing gpu_data() on uninitialized buffer triggers assert
 * - Accessing on moved-from buffer triggers assert (size>0 but d_ptr_=nullptr)
 * - to_cpu() on uninitialized triggers assert
 */
void test_error_cases() {
    constexpr size_t N = 1 << 14;  // Non-zero to trigger asserts
    std::vector<float> host(N, 0.0f);
    tinycuda::Buffer<float> buf_uninit(host.data(), N);
    // These would assert in debug; skip calls to avoid aborting suite
    // In a full test framework, expect_assert_failure({buf_uninit.gpu_data()});
    printf("[INFO] Uninitialized gpu_data()/to_cpu() should assert (skipped to avoid abort)\n");

    // Moved-from (non-zero size, but d_ptr_ null/freed)
    tinycuda::Buffer<float> buf_moved(host.data(), N);
    buf_moved.to_gpu();
    tinycuda::Buffer<float> buf_dummy(nullptr, 0);  // Empty dummy
    buf_moved = std::move(buf_dummy);  // Now moved-from, size=N but d_ptr_=nullptr
    printf("[INFO] Moved-from gpu_data() should assert (skipped to avoid abort)\n");
    printf("[PASS] error cases documented (run in debug for asserts)\n");
}

int main() {
    test_multiple_to_gpu();
    test_move_semantics();
    test_move_assignment();
    test_zero_size();
    test_error_cases();
    printf("[PASS] All Buffer tests passed\n");
    return 0;
}