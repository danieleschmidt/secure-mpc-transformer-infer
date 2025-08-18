#!/bin/bash
# Update-content script - runs when container content is updated

echo "🔄 Updating Secure MPC Transformer development environment..."

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate mpc-dev

# Update pre-commit hooks
if [ -f .pre-commit-config.yaml ]; then
    echo "🔗 Updating pre-commit hooks..."
    pre-commit autoupdate
fi

# Reinstall package if setup files changed
if [ -f pyproject.toml ]; then
    echo "📦 Reinstalling package..."
    pip install -e ".[dev,gpu,quantum-planning,test]" --upgrade
fi

# Update conda environment
echo "🐍 Updating conda environment..."
conda update --all -y

# Clear Python cache
echo "🧹 Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo "✅ Update complete!"
