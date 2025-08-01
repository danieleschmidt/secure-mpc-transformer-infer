#!/bin/bash
# Post-start script for Secure MPC Transformer development container

set -e

echo "🌟 Starting development services..."

# Ensure conda environment is activated
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc-transformer 2>/dev/null || echo "⚠️  Conda environment not found, using system Python"

# Set working directory
cd /workspace

# Update environment variables
export PYTHONPATH="/workspace/src:$PYTHONPATH"
export MPC_DEBUG=1

# Check GPU availability
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -3
else
    echo "⚠️  No GPU detected"
fi

# Start Jupyter Lab in background (if not already running)
if ! pgrep -f "jupyter-lab" > /dev/null; then
    echo "📊 Starting Jupyter Lab..."
    nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/workspace > /tmp/jupyter.log 2>&1 &
    echo "📊 Jupyter Lab started at http://localhost:8888"
fi

# Check if project dependencies are installed
if ! python -c "import secure_mpc_transformer" 2>/dev/null; then
    echo "📦 Installing project dependencies..."
    if [ -f "pyproject.toml" ]; then
        pip install -e ".[dev,gpu,benchmark]"
    fi
fi

# Ensure pre-commit hooks are installed
if [ -f ".pre-commit-config.yaml" ] && [ ! -f ".git/hooks/pre-commit" ]; then
    echo "🔧 Installing pre-commit hooks..."
    pre-commit install
fi

# Display helpful information
echo ""
echo "✅ Development environment ready!"
echo ""
echo "🔗 Services:"
echo "  📊 Jupyter Lab:  http://localhost:8888"
echo "  📈 Grafana:      http://localhost:3000 (admin/admin)"
echo "  🔍 Prometheus:   http://localhost:9090"
echo "  🏃 MPC API:      http://localhost:8080 (when running)"
echo ""
echo "🛠️  Quick commands:"
echo "  dev-setup       - Initialize environment"
echo "  run-tests       - Run test suite"
echo "  benchmark       - Run benchmarks"
echo "  gpu-monitor     - Monitor GPU usage"
echo ""
echo "📚 Documentation:"
echo "  Architecture:   docs/ARCHITECTURE.md"
echo "  Development:    docs/DEVELOPMENT.md"
echo "  Security:       docs/security/"
echo ""
echo "Happy coding! 🚀"