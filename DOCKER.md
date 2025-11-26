# Docker Setup for HabitAlp 2.0

This document explains how to use Docker to reproduce the research results from the HabitAlp 2.0 project.

## Prerequisites

- **Docker**: Install [Docker](https://docs.docker.com/get-docker/)
- **NVIDIA Docker Runtime**: For GPU support, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **NVIDIA GPU**: Required for training and inference
- **NVIDIA Driver**: Minimum version **525.60.13** (Linux) or **527.41** (Windows)
  - This image uses CUDA 12.2, which requires these minimum driver versions
  - Check your driver version: `nvidia-smi` (look for "Driver Version")
  - **Important**: CUDA drivers are NOT forward compatible - your host driver must be at least as recent as the container's CUDA version

## Quick Start - Pull from Docker Hub

The easiest way to get started is to pull the pre-built image from Docker Hub:

```bash
# Pull the image
docker pull hkristen/habitalp_2:v1

# Run the container with GPU support
docker run --gpus all -it hkristen/habitalp_2:v1

# Inside the container, verify the setup
conda activate habitalp_2
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Building the Image Locally

If you want to build the image yourself:

```bash
# Clone the repository
git clone https://github.com/hkristen/habitalp_2.git
cd habitalp_2

# Build the image
docker build -t habitalp_2:v1 .

# This will take 15-30 minutes depending on your internet connection
```

## Using Docker Compose

For easier management with GPU support:

```bash
# Start the container
docker-compose up -d

# Access the container
docker exec -it habitalp_2 /bin/bash

# Stop the container
docker-compose down
```

## Running with Custom Data

Mount your local data directory when running the container:

```bash
docker run --gpus all -it \
  -v /path/to/your/data:/data \
  -v $(pwd)/outputs:/workspace/habitalp_2/outputs \
  hkristen/habitalp_2:v1
```

## Running Jupyter Notebooks

To run the training notebooks:

```bash
# Start container with Jupyter port exposed
docker run --gpus all -it -p 8888:8888 hkristen/habitalp_2:v1

# Inside the container, start Jupyter
conda activate habitalp_2
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then open the URL shown in your browser.

## Running Training Scripts

```bash
# Enter the container
docker run --gpus all -it -v $(pwd)/outputs:/workspace/habitalp_2/outputs hkristen/habitalp_2:v1

# Inside the container
cd /workspace/habitalp_2

# Run a training notebook or script
# Note: Update file paths in notebooks to match your data location
jupyter nbconvert --execute --to notebook notebooks/train-01-unet-post-classification-change.ipynb
```

## Running Inference

```bash
# Run inference with a config file
docker run --gpus all -it \
  -v /path/to/your/data:/data \
  -v $(pwd)/outputs:/workspace/habitalp_2/outputs \
  hkristen/habitalp_2:v1 \
  python src/inference/inference.py --config src/inference/configs/clay_data_2020_full_roi.yaml
```

## Image Details

- **Base Image**: nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
- **CUDA Version**: 12.2 (requires driver >= 525.60.13 on Linux, >= 527.41 on Windows)
- **Python Version**: 3.11
- **PyTorch Version**: 2.6.0 with CUDA 12.2 support
- **Conda Environment**: habitalp_2
- **Working Directory**: /workspace/habitalp_2
- **Code Source**: Cloned from GitHub during build

## Troubleshooting

### GPU Not Available

```bash
# Verify NVIDIA Docker runtime is installed
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi

# If this fails, reinstall NVIDIA Container Toolkit
```

### Out of Memory Errors

Reduce batch size in training scripts or use a GPU with more VRAM.

### Permission Issues with Mounted Volumes

```bash
# Run with user permissions
docker run --gpus all -it --user $(id -u):$(id -g) \
  -v /path/to/data:/data \
  hkristen/habitalp_2:v1
```

## Environment Variables

- `NVIDIA_VISIBLE_DEVICES`: Control which GPUs are visible (default: all)
- `WANDB_API_KEY`: Set your Weights & Biases API key for experiment tracking

## Citation

If you use this Docker image for your research, please cite our paper:

```bibtex
@misc{kristen2025habitatlandcoverchange,
      title={Habitat and Land Cover Change Detection in Alpine Protected Areas: A Comparison of AI Architectures},
      author={Harald Kristen and Daniel Kulmer and Manuela Hirschmugl},
      year={2025},
      eprint={2511.00073},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.00073},
}
```
