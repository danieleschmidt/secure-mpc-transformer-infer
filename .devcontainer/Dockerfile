# Development container for Secure MPC Transformer Inference
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.1
ARG PYTHON_VERSION=3.10

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Build arguments
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG PYTHON_VERSION

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic development tools
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    vim \
    nano \
    sudo \
    # Network tools
    net-tools \
    iputils-ping \
    # SSL/TLS support
    ca-certificates \
    # Additional development tools
    gdb \
    valgrind \
    htop \
    tree \
    jq \
    # Docker CLI (for docker-outside-of-docker)
    apt-transport-https \
    gnupg \
    lsb-release \
    # Protocol buffer compiler
    protobuf-compiler \
    libprotobuf-dev \
    # OpenSSL development headers
    libssl-dev \
    libffi-dev \
    # For building Python extensions
    pkg-config \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install Miniconda
USER $USERNAME
WORKDIR /home/$USERNAME

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /home/$USERNAME/miniconda3 \
    && rm miniconda.sh \
    && echo 'export PATH="/home/$USERNAME/miniconda3/bin:$PATH"' >> ~/.bashrc

ENV PATH="/home/$USERNAME/miniconda3/bin:$PATH"

# Create conda environment
RUN conda create -y -n mpc-transformer python=${PYTHON_VERSION} \
    && echo "conda activate mpc-transformer" >> ~/.bashrc

# Switch to the conda environment
SHELL ["conda", "run", "-n", "mpc-transformer", "/bin/bash", "-c"]

# Install Python development dependencies
RUN pip install --no-cache-dir \
    # Core development tools
    ipython \
    jupyter \
    jupyterlab \
    notebook \
    # Code quality
    pre-commit \
    black \
    isort \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    # Security scanning
    bandit \
    safety \
    # Documentation
    sphinx \
    sphinx-rtd-theme \
    mkdocs \
    mkdocs-material \
    # Debugging and profiling
    pdb \
    line_profiler \
    memory_profiler \
    # Additional utilities
    rich \
    typer \
    httpx \
    pydantic

# Install CUDA-specific Python packages
RUN pip install --no-cache-dir \
    cupy-cuda12x \
    numba \
    triton

# Install cryptography and MPC libraries (development versions)
RUN pip install --no-cache-dir \
    cryptography \
    pycryptodome \
    # Note: SEAL and TenSEAL would be installed via project requirements

# Configure Git (will be overridden by mounted config)
RUN git config --global init.defaultBranch main \
    && git config --global core.editor nano \
    && git config --global pull.rebase false

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && echo 'export PATH="/home/$USERNAME/miniconda3/bin:$PATH"' >> ~/.zshrc \
    && echo 'conda activate mpc-transformer' >> ~/.zshrc

# Set up workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Expose common development ports
EXPOSE 8080 8888 9090 3000 50051 6006

# Switch back to root to set up the entrypoint
USER root

# Create startup scripts directory
RUN mkdir -p /usr/local/share/docker-init.d

# Set the default command
CMD ["sleep", "infinity"]