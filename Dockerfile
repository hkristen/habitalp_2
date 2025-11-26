# HabitAlp 2.0 Docker Image
# Based on NVIDIA CUDA image with conda for reproducible geospatial deep learning

# Use NVIDIA CUDA 12.2 runtime image for wider driver compatibility
# Requires NVIDIA driver >= 525.60.13 (Linux) or >= 527.41 (Windows)
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Add conda to PATH
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (lighter than Anaconda, with conda-forge as default)
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda clean -afy

# Clone the repository from GitHub
WORKDIR /workspace
RUN git clone https://github.com/hkristen/habitalp_2.git && \
    cd habitalp_2

# Set working directory to the cloned repo
WORKDIR /workspace/habitalp_2

# Create conda environment from file
RUN conda env create -f env.yml && \
    conda clean -afy && \
    echo "source activate habitalp_2" > ~/.bashrc

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "habitalp_2", "/bin/bash", "-c"]

# Set the default command to bash with the environment activated
ENV CONDA_DEFAULT_ENV=habitalp_2
ENV PATH=${CONDA_DIR}/envs/habitalp_2/bin:${PATH}

# Create directory for data (can be mounted as volume)
RUN mkdir -p /data

# Expose Jupyter port (in case users want to run notebooks)
EXPOSE 8888

# Set working directory
WORKDIR /workspace/habitalp_2

# Default command - activate environment and start bash
CMD ["/bin/bash"]
