# Qwen2.5-VL Fine-Tuning Environment Setup

## System Info

- **GPU:** NVIDIA GH200 480GB
- **Architecture:** aarch64
- **CUDA Toolkit:** 12.8 (nvcc)
- **Driver CUDA:** 13.0

## 1. Install Miniconda

```bash
# Download miniconda for aarch64
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh

# Install to ~/miniconda3
bash /tmp/miniconda.sh -b -p /home/ubuntu/miniconda3

# Add to PATH
export PATH="/home/ubuntu/miniconda3/bin:$PATH"

# Accept TOS (required on first use)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 2. Create Conda Environment

```bash
conda create -n qwen25vl python=3.10 -y
```

## 3. Install PyTorch (CUDA 12.8)

```bash
conda run -n qwen25vl pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Installed versions:
- torch==2.10.0+cu128
- torchvision==0.25.0+cu128

## 4. Install Qwen2.5-VL Dependencies

```bash
conda run -n qwen25vl pip install transformers accelerate peft bitsandbytes qwen-vl-utils huggingface_hub wandb
```

Installed versions:
- transformers==5.3.0
- accelerate==1.13.0
- peft==0.18.1
- bitsandbytes==0.49.2
- qwen-vl-utils==0.0.14
- wandb==0.25.1

## 5. Activate & Verify

```bash
conda activate qwen25vl

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import transformers; print(f'transformers: {transformers.__version__}')
import peft; print(f'peft: {peft.__version__}')
import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')
"
```

## 6. Run Training

```bash
conda activate qwen25vl

# Quick test (500 samples)
python scripts/train_lora.py --mini

# Full training
python scripts/train_lora.py
```
