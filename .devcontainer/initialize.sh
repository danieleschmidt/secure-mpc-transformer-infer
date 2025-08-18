#!/bin/bash
# Initialize script - runs on the host before container creation

echo "🚀 Initializing Secure MPC Transformer development environment..."

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected - GPU acceleration will not be available"
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker is available"
else
    echo "❌ Docker is required but not found"
    exit 1
fi

# Check Git LFS
if command -v git-lfs &> /dev/null; then
    echo "✅ Git LFS is available"
else
    echo "⚠️  Git LFS not found - large model files may not download properly"
fi

echo "🎯 Ready to create development container"
