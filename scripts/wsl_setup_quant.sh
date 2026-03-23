#!/bin/bash
# Setup quantization environment in WSL2
set -e

CONDA=~/miniconda3/bin/conda
PIP=~/miniconda3/envs/quant/bin/pip
PY=~/miniconda3/envs/quant/bin/python

echo "=== Creating conda env 'quant' ==="
$CONDA create -n quant python=3.11 -y

echo "=== Installing PyTorch 2.6 + CUDA 12.4 ==="
$PIP install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing transformers 4.51.3 (compatible with autoawq/auto-gptq) ==="
$PIP install transformers==4.51.3 accelerate

echo "=== Installing autoawq ==="
$PIP install autoawq

echo "=== Installing auto-gptq ==="
$PIP install auto-gptq

echo "=== Verifying ==="
$PY -c "
import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
import transformers; print(f'transformers={transformers.__version__}')
from awq import AutoAWQForCausalLM; print('autoawq OK')
from auto_gptq import AutoGPTQForCausalLM; print('auto-gptq OK')
"

echo "=== Done ==="
