#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace tinycuda {

/**
 * @brief Check a CUDA error code and exit if there is an error.
 *
 * @param err The cudaError_t returned by a CUDA runtime call.
 * @param file The source file where the error occurred.
 * @param line The line number in the source file.
 *
 * This function prints an error message and aborts if `err` indicates a failure.
 * Consider wrapping in try-catch for exception-based alternatives in production.
 *
 * Example usage:
 * cudaError_t status = cudaMalloc(&ptr, N * sizeof(float));
 * tinycuda::cuda_check(status, __FILE__, __LINE__);
 */
inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[CUDA ERROR] %s:%d: %s\n",
                file,
                line,
                cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
    // Clear any pending errors (best practice)
    (void) cudaGetLastError();
}

/**
 * @brief Convenience macro to automatically pass file and line number.
 *
 * Evaluates the expression, checks its return value, and aborts on error.
 *
 * Example:
 * CUDA_CHECK(cudaMalloc(&ptr, N * sizeof(float)));
 */
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t status = (expr); \
        tinycuda::cuda_check(status, __FILE__, __LINE__); \
    } while(0)

} // namespace tinycuda