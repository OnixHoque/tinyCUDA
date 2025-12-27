#!/usr/bin/env bash
set -e

# Quick script to compile and run any standalone .cu file that uses tinycuda
# Auto-detects GPU architecture for universal compatibility
#
# Usage:
#   ./scripts/build_and_run.sh                          # defaults to examples/vector_add.cu
#   ./scripts/build_and_run.sh examples/matmul.cu       # specific file
#   ./scripts/build_and_run.sh path/to/my_kernel.cu my_kernel  # custom output name

# Check for nvcc (non-fatal)
if ! command -v nvcc &> /dev/null; then
    echo "[WARN] nvcc not found. Install CUDA Toolkit for compilation."
    exit 1
fi

# Auto-detect compute capability from nvidia-smi
detect_arch() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "[WARN] nvidia-smi not found. Defaulting to sm_75 (Turing/T4)."
        echo "sm_75"
        return
    fi

    local gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    case "$gpu_model" in
        *"T4"*) echo "sm_75" ;;  # Turing
        *"A100"*) echo "sm_80" ;;  # Ampere
        *"RTX 30"*) echo "sm_86" ;;  # Ampere consumer
        *"RTX 40"*) echo "sm_89" ;;  # Ada Lovelace
        *"V100"*) echo "sm_70" ;;  # Volta
        *"P100"*) echo "sm_60" ;;  # Pascal
        *) echo "[WARN] Unknown GPU '$gpu_model'. Defaulting to sm_75."
           echo "sm_75" ;;
    esac
}

ARCH_FLAG="-arch=$(detect_arch)"

# Determine source and output
if [[ $# -eq 0 ]]; then
    SRC="examples/vector_add.cu"
    OUT_NAME="vector_add"
else
    SRC="$1"
    OUT_NAME="${2:-$(basename "$SRC" .cu)}"
fi

# Default to build/ dir
mkdir -p build
OUT="build/$OUT_NAME"

echo "[INFO] Detected arch: $(echo $ARCH_FLAG | cut -d= -f2)"
echo "[INFO] Building $SRC → $OUT ..."
nvcc -std=c++17 $ARCH_FLAG -Iinclude -O2 "$SRC" -o "$OUT"

echo "[INFO] Running $OUT ..."
./"$OUT"