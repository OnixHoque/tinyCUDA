#pragma once
#include <cstddef>
#include <cassert>
#include <cuda_runtime.h>
#include "tinycuda/error.hpp"

namespace tinycuda {

/**
 * @brief GPU mirror of user-owned CPU memory.
 *
 * Buffer does NOT allocate or free host memory.
 * It only manages CUDA device allocation and H↔D copies.
 *
 * Typical usage:
 * std::vector<float> h(N);
 * Buffer<float> buf(h.data(), N);
 * buf.to_gpu();
 * kernel<<<grid, block>>>(buf.gpu_data(), N);
 * buf.to_cpu(); // results now in h
 *
 * Supports zero-size buffers (no-op operations).
 * Host memory must outlive the Buffer.
 */
template<typename T>
class Buffer {
public:
    /**
     * @brief Bind to existing CPU memory (non-owning).
     *
     * @param host_ptr Pointer to user-owned CPU memory (nullptr OK if size==0)
     * @param size Number of elements
     *
     * The host memory must remain valid for the lifetime of the Buffer.
     *
     * Example:
     * float* data = (float*)malloc(N * sizeof(float));
     * Buffer<float> buf(data, N);
     */
    Buffer(T* host_ptr, size_t size)
        : h_ptr_(host_ptr), size_(size) {
        assert(size_ == 0 || h_ptr_ != nullptr);  // Relaxed for zero-size
    }

    /**
     * @brief Destructor.
     *
     * Frees CUDA device memory if allocated.
     * Host memory is NEVER freed.
     */
    ~Buffer() {
        if (d_ptr_) {
            CUDA_CHECK(cudaFree(d_ptr_));
        }
    }

    /**
     * @brief Move constructor.
     *
     * Transfers ownership of device memory.
     * The moved-from Buffer becomes empty/invalid.
     *
     * Example:
     * Buffer<float> a(ptr, N);
     * Buffer<float> b = std::move(a);
     */
    Buffer(Buffer&& other) noexcept
        : h_ptr_(other.h_ptr_),
          d_ptr_(other.d_ptr_),
          size_(other.size_),
          on_gpu_(other.on_gpu_) {
        other.d_ptr_ = nullptr;
        other.h_ptr_ = nullptr;  // Make moved-from fully invalid
        other.on_gpu_ = false;
        other.size_ = 0;
    }

    /**
     * @brief Move assignment operator.
     *
     * Frees existing device memory (if any),
     * then takes ownership from the source.
     */
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            if (d_ptr_) {
                CUDA_CHECK(cudaFree(d_ptr_));
            }
            h_ptr_ = other.h_ptr_;
            d_ptr_ = other.d_ptr_;
            size_ = other.size_;
            on_gpu_ = other.on_gpu_;
            other.d_ptr_ = nullptr;
            other.h_ptr_ = nullptr;  // Make moved-from fully invalid
            other.on_gpu_ = false;
            other.size_ = 0;
        }
        return *this;
    }

    // No copying (avoids double-free of device memory)
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    /**
     * @brief Allocate device memory (once) and copy CPU → GPU.
     *
     * Safe to call multiple times (recopies host data).
     *
     * Example:
     * buf.to_gpu();
     */
    void to_gpu() {
        if (!d_ptr_) {
            CUDA_CHECK(cudaMalloc(&d_ptr_, size_ * sizeof(T)));
        }
        CUDA_CHECK(cudaMemcpy(
            d_ptr_,
            h_ptr_,
            size_ * sizeof(T),
            cudaMemcpyHostToDevice
        ));
        on_gpu_ = true;
    }

    /**
     * @brief Copy GPU → CPU.
     *
     * Device memory is retained (data mirrored bidirectionally post-call).
     *
     * Example:
     * kernel<<<...>>>(buf.gpu_data());
     * buf.to_cpu(); // results now in host memory
     */
    void to_cpu() {
        if (size_ == 0) return;  // No-op for zero-size
        assert(d_ptr_ && "Buffer not on GPU (call to_gpu() first)");
        CUDA_CHECK(cudaMemcpy(
            h_ptr_,
            d_ptr_,
            size_ * sizeof(T),
            cudaMemcpyDeviceToHost
        ));
    }

    /**
     * @brief Access CPU pointer.
     *
     * Returns the user-owned host pointer.
     */
    T* cpu_data() const {
        return h_ptr_;
    }

    /**
     * @brief Access GPU pointer.
     *
     * Valid only after to_gpu() has been called.
     * Returns nullptr for zero-size buffers (no allocation).
     *
     * Example:
     * kernel<<<grid, block>>>(buf.gpu_data(), N);
     */
    T* gpu_data() const {
        if (size_ == 0) return nullptr;  // Explicitly allow for zero-size
        assert(d_ptr_ && "GPU data not allocated (call to_gpu() first)");
        return d_ptr_;
    }

    /**
     * @brief Number of elements in the buffer.
     */
    size_t size() const {
        return size_;
    }

    /**
     * @brief Check if data has been transferred to GPU.
     *
     * Remains true after to_cpu() since device copy is retained.
     */
    bool on_gpu() const {
        return on_gpu_;
    }

private:
    T* h_ptr_ = nullptr; // user-owned host memory
    T* d_ptr_ = nullptr; // device memory (owned by Buffer)
    size_t size_ = 0;
    bool on_gpu_ = false;
};

} // namespace tinycuda