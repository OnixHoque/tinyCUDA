#!/usr/bin/env bash
set -e

# Check for nvcc (non-fatal)
if ! command -v nvcc &> /dev/null; then
    echo "[WARN] nvcc not found. Install CUDA Toolkit for compilation."
fi

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

echo "[INFO] Building $SRC → $OUT ..."
nvcc -std=c++17 -Iinclude -O2 "$SRC" -o "$OUT"

echo "[INFO] Running $OUT ..."
./"$OUT"