#!/bin/bash
# Post-create script - runs after container creation and on-create

echo "🔧 Post-creation setup for Secure MPC Transformer..."

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate mpc-dev

# Verify GPU access
echo "🎮 Checking GPU access..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Run quick validation
echo "✅ Running quick validation..."
if [ -f scripts/validate-environment.py ]; then
    python scripts/validate-environment.py
else
    echo "⚠️  Environment validation script not found"
fi

# Start background services if needed
echo "🚀 Starting development services..."

# Start Redis for development (if available)
if command -v redis-server &> /dev/null; then
    echo "📡 Starting Redis server..."
    redis-server --daemonize yes --port 6379 --loglevel notice
fi

# Generate development certificates if they don't exist
if [ ! -f certs/server.crt ]; then
    echo "🔐 Generating development certificates..."
    mkdir -p certs
    openssl req -x509 -newkey rsa:4096 -keyout certs/server.key -out certs/server.crt -days 365 -nodes \
        -subj "/C=US/ST=Development/L=Development/O=SecureMPC/OU=Development/CN=localhost"
    chmod 600 certs/server.key
fi

# Set up Jupyter configuration
echo "📓 Configuring Jupyter..."
jupyter lab --generate-config --allow-root
echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py

# Create sample notebooks
echo "📚 Creating sample notebooks..."
mkdir -p notebooks
cat > notebooks/Getting_Started.ipynb << 'NOTEBOOK'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Secure MPC Transformer - Getting Started\n",
    "\n",
    "This notebook demonstrates basic usage of the Secure MPC Transformer system."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Import required libraries\n",
    "import torch\n",
    "from secure_mpc_transformer import SecureTransformer, SecurityConfig\n",
    "\n",
    "# Check GPU availability\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "print(f\"CUDA devices: {torch.cuda.device_count()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"Current device: {torch.cuda.current_device()}\")\n",
    "    print(f\"Device name: {torch.cuda.get_device_name()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Configure security settings\n",
    "config = SecurityConfig(\n",
    "    protocol=\"3pc_semi_honest\",  # Use semi-honest for development\n",
    "    security_level=128,\n",
    "    gpu_acceleration=True,\n",
    "    debug_mode=True\n",
    ")\n",
    "\n",
    "print(f\"Security configuration: {config}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load a pre-trained model\n",
    "model = SecureTransformer.from_pretrained(\n",
    "    \"bert-base-uncased\",\n",
    "    security_config=config\n",
    ")\n",
    "\n",
    "print(\"Model loaded successfully!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
NOTEBOOK

echo "🎉 Post-creation setup complete!"
echo ""
echo "🚀 Quick start commands:"
echo "  mpc-validate    - Validate environment"
echo "  mpc-test-fast   - Run fast tests"
echo "  mpc-serve       - Start API server"
echo "  jupyter lab     - Start Jupyter Lab"
echo ""
echo "📖 Open notebooks/Getting_Started.ipynb to begin!"
