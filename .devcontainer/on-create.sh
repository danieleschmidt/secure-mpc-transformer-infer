#!/bin/bash
# On-create script - runs once when container is first created

echo "🏗️  Setting up Secure MPC Transformer development environment..."

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate mpc-dev

# Install the package in development mode
if [ -f pyproject.toml ]; then
    echo "📦 Installing package in development mode..."
    pip install -e ".[dev,gpu,quantum-planning,test]"
fi

# Install pre-commit hooks
if [ -f .pre-commit-config.yaml ]; then
    echo "🔗 Installing pre-commit hooks..."
    pre-commit install
fi

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p logs data models cache results

# Download CUDA samples for reference
echo "🔬 Setting up CUDA development environment..."
if [ -d /usr/local/cuda/samples ]; then
    cp -r /usr/local/cuda/samples ~/cuda-samples
fi

# Initialize Git LFS if available
if command -v git-lfs &> /dev/null; then
    echo "📂 Initializing Git LFS..."
    git lfs install
fi

# Set up shell completions
echo "⚡ Setting up shell completions..."
echo 'eval "$(register-python-argcomplete secure-mpc-transformer)"' >> ~/.bashrc

# Create useful aliases
echo "🔧 Setting up development aliases..."
cat >> ~/.bashrc << 'ALIASES'

# Development aliases
alias mpc-test='pytest tests/ -v'
alias mpc-test-fast='pytest tests/ -v -m "not slow"'
alias mpc-benchmark='python -m secure_mpc_transformer.benchmark'
alias mpc-security-audit='python -m secure_mpc_transformer.security.audit'
alias mpc-validate='python -m secure_mpc_transformer.validate'
alias mpc-serve='python -m secure_mpc_transformer.server'
alias mpc-party='python -m secure_mpc_transformer.party'

# GPU monitoring
alias gpu='nvidia-smi'
alias gpu-watch='watch -n 1 nvidia-smi'
alias gpu-temp='nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits'

# Development shortcuts
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Git shortcuts
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'

ALIASES

echo "✅ Development environment setup complete!"
echo "🎯 You can now start developing with 'conda activate mpc-dev'"
