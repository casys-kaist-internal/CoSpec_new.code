#!/bin/bash
# Install CoSpec dependencies and build libsmctrl.
# Use this on cloud instances where Docker is unavailable.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Installing Python dependencies ==="
pip3 install Cython UltraDict pytest

echo "=== Installing vLLM (editable) ==="
cd "$PROJECT_ROOT"
pip3 install -e . --no-build-isolation

echo "=== Building libsmctrl ==="
# Adopted from BulletServe (https://github.com/NUS-HPC-AI-Lab/BulletServe)
CSRC_DIR="$PROJECT_ROOT/cospec/csrc"
cd "$CSRC_DIR"
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | sed 's/\.//')
cmake -B build \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --target smctrl -j "$(nproc)"

echo "=== Verifying libsmctrl.so ==="
ls -la "$CSRC_DIR/build/libsmctrl.so"

echo "=== Setup MPS ==="
"$PROJECT_ROOT/cospec/scripts/start_mps.sh"

echo "===== INSTALL COMPLETE ====="
