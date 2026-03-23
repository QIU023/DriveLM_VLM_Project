#!/bin/bash
# One-click setup for GB200 / GH200 server
# Usage: bash setup_gb200.sh
#
# Assumes: fresh Ubuntu with NVIDIA driver installed

set -e

echo "=========================================="
echo "  GB200/GH200 Environment Setup"
echo "=========================================="

# 1. System packages
echo "[1/6] Installing system packages..."
apt-get update && apt-get install -y git wget tmux htop

# 2. Conda
echo "[2/6] Setting up conda..."
if ! command -v conda &>/dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
fi
export PATH="$HOME/miniconda3/bin:$PATH"

# 3. Conda environment
echo "[3/6] Creating conda env (qwen25vl, Python 3.10)..."
conda create -n qwen25vl python=3.10 -y || true
eval "$(conda shell.bash hook)"
conda activate qwen25vl

# 4. Python packages
echo "[4/6] Installing PyTorch + dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q "transformers>=5.3.0" "peft>=0.18.0" accelerate bitsandbytes
pip install -q qwen-vl-utils Pillow tqdm pyyaml wandb scipy

# 5. Pre-download models (cache to ~/.cache/huggingface)
echo "[5/6] Pre-downloading models..."
python -c "
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
print('Downloading Qwen2.5-VL-3B...')
AutoModelForImageTextToText.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.bfloat16)
AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
print('Downloading Qwen2.5-VL-7B...')
AutoModelForImageTextToText.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', torch_dtype=torch.bfloat16)
AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
print('Done!')
"

# 6. Verify
echo "[6/6] Verifying..."
python -c "
import torch, transformers, peft
print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'transformers {transformers.__version__} | peft {peft.__version__}')
"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  Next:"
echo "    1. Upload data/:  scp -r data/ data_processed/ <server>:DriveLM/"
echo "    2. Precompute:    python scripts/precompute_crp_data.py --config configs/gb200.yaml"
echo "    3. Run all:       bash configs/run_all_gb200.sh"
echo "=========================================="
