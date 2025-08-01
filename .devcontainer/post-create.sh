#!/bin/bash
# Post-create script for Secure MPC Transformer development container

set -e

echo "🚀 Setting up Secure MPC Transformer development environment..."

# Ensure we're in the workspace directory
cd /workspace

# Install project dependencies
echo "📦 Installing project dependencies..."
if [ -f "pyproject.toml" ]; then
    pip install -e ".[dev,gpu,benchmark]"
else
    echo "⚠️  pyproject.toml not found, skipping project installation"
fi

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "⚠️  .pre-commit-config.yaml not found, skipping pre-commit setup"
fi

# Set up Jupyter Lab extensions and configuration
echo "📊 Configuring Jupyter Lab..."
jupyter lab --generate-config
cat << 'EOF' >> ~/.jupyter/jupyter_lab_config.py
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
c.LabApp.default_url = '/lab'
EOF

# Install useful Jupyter extensions
pip install --no-cache-dir \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all] \
    jupyter-ai

# Create useful directories
echo "📁 Creating development directories..."
mkdir -p ~/.cache/pip
mkdir -p ~/.cache/pre-commit
mkdir -p ~/workspace/notebooks
mkdir -p ~/workspace/experiments

# Set up GPU monitoring script
echo "🖥️  Setting up GPU monitoring..."
cat << 'EOF' > ~/bin/gpu-monitor
#!/bin/bash
watch -n 1 nvidia-smi
EOF
chmod +x ~/bin/gpu-monitor

# Create useful aliases
echo "🔧 Setting up development aliases..."
cat << 'EOF' >> ~/.bashrc
# Development aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Project-specific aliases
alias pytest-fast='pytest -x -v'
alias pytest-cov='pytest --cov=secure_mpc_transformer --cov-report=html'
alias black-check='black --check --diff .'
alias ruff-check='ruff check .'
alias mypy-check='mypy src/'
alias pre-commit-all='pre-commit run --all-files'

# GPU aliases
alias gpu='nvidia-smi'
alias gpu-watch='watch -n 1 nvidia-smi'
alias gpu-temp='nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits'

# Docker aliases
alias dps='docker ps'
alias dpsa='docker ps -a'
alias di='docker images'
alias dip='docker image prune -f'
alias dvp='docker volume prune -f'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gb='git branch'
alias gco='git checkout'
EOF

# Also add to zsh
cp ~/.bashrc ~/.zshrc

# Set up development environment variables
echo "🌍 Setting up environment variables..."
cat << 'EOF' >> ~/.bashrc
# Development environment
export PYTHONPATH="/workspace/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=all
export NVIDIA_VISIBLE_DEVICES=all

# MPC development settings
export MPC_DEBUG=1
export MPC_LOG_LEVEL=INFO
export MPC_BACKEND=gpu

# Jupyter settings
export JUPYTER_ENABLE_LAB=yes
EOF

# Create a sample notebook
echo "📓 Creating sample notebook..."
cat << 'EOF' > ~/workspace/notebooks/mpc_development.ipynb
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Secure MPC Transformer Development\n",
    "\n",
    "This notebook provides a starting point for developing and testing MPC transformer inference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import torch\n",
    "import numpy as np\n",
    "from transformers import AutoTokenizer, AutoModel\n",
    "\n",
    "# Check GPU availability\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU count: {torch.cuda.device_count()}\")\n",
    "    print(f\"Current GPU: {torch.cuda.current_device()}\")\n",
    "    print(f\"GPU name: {torch.cuda.get_device_name(0)}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Set up shell completion
echo "🔧 Setting up shell completion..."
echo 'eval "$(_PYTEST_COMPLETE=bash_source pytest)"' >> ~/.bashrc

# Create development scripts
echo "📜 Creating development scripts..."
mkdir -p ~/bin

cat << 'EOF' > ~/bin/dev-setup
#!/bin/bash
# Quick development environment setup
cd /workspace
conda activate mpc-transformer
export PYTHONPATH="/workspace/src:$PYTHONPATH"
echo "🚀 Development environment ready!"
echo "📁 Working directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Pip packages: $(pip list | wc -l) installed"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "🖥️  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi
EOF

cat << 'EOF' > ~/bin/run-tests
#!/bin/bash
# Run comprehensive test suite
cd /workspace
echo "🧪 Running test suite..."
pytest tests/ -v --cov=secure_mpc_transformer --cov-report=html --cov-report=term
echo "📊 Coverage report generated in htmlcov/"
EOF

cat << 'EOF' > ~/bin/benchmark
#!/bin/bash
# Run performance benchmarks
cd /workspace
echo "⚡ Running performance benchmarks..."
python benchmarks/run_all.py --gpu --models bert-base
EOF

chmod +x ~/bin/*

# Final setup
echo "✅ Post-create setup complete!"
echo ""
echo "🔧 Available commands:"
echo "  dev-setup     - Initialize development environment"
echo "  run-tests     - Run comprehensive test suite"
echo "  benchmark     - Run performance benchmarks"
echo "  gpu-monitor   - Monitor GPU usage"
echo ""
echo "📊 Access Jupyter Lab at: http://localhost:8888"
echo "📈 Access Grafana at: http://localhost:3000 (admin/admin)"
echo "🔍 Access Prometheus at: http://localhost:9090"
echo ""
echo "🚀 Happy coding!"