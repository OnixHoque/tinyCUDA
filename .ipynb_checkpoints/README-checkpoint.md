# tinyCUDA

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/OnixHoque/tinycuda/actions)  
*A lightweight CUDA wrapper for memory management, kernel timing, and error handling—focus on kernels, not boilerplate.*

tinyCUDA strips away the tedium of CUDA development: no more manual `cudaMalloc`/`cudaMemcpy` juggling, launch errors, or rough timings. It's designed for quick prototyping—perfect for new CUDA users or anyone tired of repetitive setup. Not a full framework (no autograd or N-D tensors); just sane defaults for 1D buffers and kernel launches.

**Built with:** C++17, CUDA 11+. Tested on NVIDIA GPUs (Turing+).

## Features

- **Buffer<T>**: Non-owning GPU mirror of host data. Allocates device mem on first `to_gpu()`, copies H↔D, handles moves. Zero-size support.
- **KernelProfiler**: Warmup + batched timing for accurate kernel execution averages (ms). Untimed warmup avoids JIT overhead.
- **CUDA_CHECK**: Macro for immediate error checking/abort with file:line context.
- **tinycuda.hpp**: One-include aggregator for the full API.

## Quick Start

### Prerequisites
- CUDA Toolkit 11+ (nvcc required).
- C++17 compiler (nvcc handles it).

### Build & Run Examples
```bash
git clone https://github.com/OnixHoque/tinycuda.git
cd tinycuda
./scripts/build_and_run.sh                          # Defaults to vector_add example
./scripts/build_and_run.sh ./examples/matmul.cu     # Matmul example
```

- **Vector Add**: Simple element-wise addition with verification.
- **Matmul**: Basic matrix multiply (512x512) with timing & GFLOPS.

Output example (vector_add):
```
[INFO] Building examples/vector_add.cu → build/vector_add ...
[INFO] Running build/vector_add ...
Average kernel time: 0.0032 ms
[PASS] Vector addition verified: all elements == 3.0f
[SUCCESS] Example completed.
```

### Run Tests
```bash
./scripts/run_tests.sh
```
Runs `test_memory` and `test_profiler`—verifies Buffer moves, zero-size, and profiler accuracy.

## API Overview

### Buffer: GPU-Host Mirroring
```cpp
std::vector<float> host(N, 0.0f);
tinycuda::Buffer<float> buf(host.data(), N);
buf.to_gpu();  // Alloc + H→D copy (safe to repeat)
kernel<<<grid, block>>>(buf.gpu_data(), N);
buf.to_cpu();  // D→H copy (device mem retained)
```

- Move-enabled; no copy ctor.
- `on_gpu()` flag; `size()` query.
- Docs: [memory.hpp](include/tinycuda/memory.hpp).

### KernelProfiler: Timing
```cpp
tinycuda::KernelProfiler prof(5, 50);  // Warmup, repeats
float ms = prof([&] { kernel<<<grid, block>>>(...); });
printf("Avg: %.4f ms\n", ms);
```

- Batched launches (no per-run sync); averages total time.
- Docs: [profiler.hpp](include/tinycuda/profiler.hpp).

### Error Handling
```cpp
#define CUDA_CHECK(expr) /* Auto-checks & aborts on fail */
CUDA_CHECK(cudaMalloc(&ptr, N * sizeof(float)));
```

- Prints file:line + `cudaGetErrorString`.
- Docs: [error.hpp](include/tinycuda/error.hpp).

Include everything: `#include "tinycuda/tinycuda.hpp"`.

## Examples

- **[vector_add.cu](examples/vector_add.cu)**: Element-wise add with Buffer + Profiler + verify.
- **[matmul.cu](examples/matmul.cu)**: Non-tiled matrix multiply (512x512) with GFLOPS calc.

Build any: `./scripts/build_and_run.sh examples/matmul.cu matmul`.

## Project Structure
```
.
├── include/tinycuda/     # Headers
│   ├── error.hpp
│   ├── memory.hpp
│   ├── profiler.hpp
│   └── tinycuda.hpp      # Aggregator
├── examples/             # Demos
├── tests/                # Unit tests
├── scripts/              # Build/run helpers
└── README.md
```

## License
MIT License—see [LICENSE](LICENSE). Free to fork, extend, or teach with.

## Contributing
- Tests: Add to `tests/` + run `./scripts/run_tests.sh`.
- Issues/PRs: Welcome! Focus on simplicity.


---

*Questions? Open an issue or ping on [LinkedIn](https://www.linkedin.com/in/onixhoque/).*