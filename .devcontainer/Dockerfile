# Secure MPC Transformer Development Container
# Based on NVIDIA CUDA with Python development tools

ARG VARIANT=3.10-bullseye
ARG CUDA_VERSION=12.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git openssh-client less iproute2 procps lsb-release \
    #
    # Install C++ tools for CUDA development
    && apt-get -y install build-essential cmake gdb ninja-build \
    #
    # Install Python and pip
    && apt-get -y install python3 python3-pip python3-dev python3-venv \
    #
    # Install additional development tools
    && apt-get -y install curl wget vim nano htop tree jq \
    #
    # Install cryptographic libraries
    && apt-get -y install libssl-dev libffi-dev \
    #
    # Install mathematical libraries
    && apt-get -y install libblas-dev liblapack-dev libatlas-base-dev \
    #
    # Install networking tools
    && apt-get -y install netcat-openbsd telnet \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda clean -tipsy

# Create conda environment for development
RUN conda create -n mpc-dev python=3.10 -y \
    && conda activate mpc-dev \
    && conda install -c conda-forge \
        numpy scipy matplotlib seaborn \
        jupyter jupyterlab \
        pytest pytest-cov pytest-xdist \
        black isort flake8 mypy pylint bandit \
        pre-commit \
        -y

# Install PyTorch with CUDA support
RUN conda activate mpc-dev \
    && conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install additional Python packages for MPC development
RUN conda activate mpc-dev \
    && pip install --no-cache-dir \
        transformers \
        datasets \
        accelerate \
        wandb \
        mlflow \
        prometheus-client \
        redis \
        sqlalchemy \
        alembic \
        fastapi \
        uvicorn \
        pydantic \
        cryptography \
        pycryptodome \
        gmpy2 \
        sympy

# Install development and testing tools
RUN conda activate mpc-dev \
    && pip install --no-cache-dir \
        pytest-benchmark \
        pytest-mock \
        pytest-asyncio \
        coverage \
        codecov \
        memory-profiler \
        line-profiler \
        py-spy \
        scalene

# Install CUDA development tools
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install additional CUDA libraries for MPC
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
        libcudnn8-dev \
        libcublas-dev \
        libcurand-dev \
        libcusparse-dev \
        libcusolver-dev \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to use if preferred
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up shell for the user
RUN usermod --shell /bin/bash $USERNAME

# Configure git for the user
USER $USERNAME
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false

# Set up conda for the user
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate mpc-dev" >> ~/.bashrc

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Expose common ports
EXPOSE 8080 8001 8002 8003 9090 3000 6379 5432 8888 6006

# Set the default command
CMD ["/bin/bash"]
