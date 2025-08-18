#!/bin/bash
# Post-start script - runs every time container starts

echo "🌟 Starting Secure MPC Transformer development session..."

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate mpc-dev

# Display system information
echo "📊 System Information:"
echo "  OS: $(uname -s) $(uname -r)"
echo "  Python: $(python --version)"
echo "  Conda env: $CONDA_DEFAULT_ENV"

# Check GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    echo "  GPU: Not available"
fi

# Check memory
echo "  Memory: $(free -h | grep ^Mem | awk '{print $2}')"

# Start development services
echo ""
echo "🚀 Starting development services..."

# Start Redis if not running
if ! pgrep redis-server > /dev/null; then
    if command -v redis-server &> /dev/null; then
        echo "📡 Starting Redis server..."
        redis-server --daemonize yes --port 6379 --loglevel notice
    fi
fi

# Display active services
echo ""
echo "📡 Active services:"
if pgrep redis-server > /dev/null; then
    echo "  ✅ Redis (port 6379)"
else
    echo "  ❌ Redis"
fi

# Show helpful commands
echo ""
echo "🛠️  Development commands:"
echo "  mpc-validate      - Validate installation"
echo "  mpc-test          - Run all tests"
echo "  mpc-test-fast     - Run fast tests only"
echo "  mpc-benchmark     - Run benchmarks"
echo "  mpc-serve         - Start API server"
echo "  jupyter lab       - Start Jupyter Lab"
echo "  gpu               - Show GPU status"
echo "  gpu-watch         - Monitor GPU usage"
echo ""
echo "📖 Documentation: docs/GETTING_STARTED.md"
echo "🔧 Troubleshooting: docs/TROUBLESHOOTING.md"
echo ""
echo "🎯 Ready for development!"
