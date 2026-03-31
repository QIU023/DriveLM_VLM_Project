# Environment Snapshot (2026-03-31)

## System
- GPU: NVIDIA B200
- VRAM: 183359 MiB
- Arch: x86_64
- OS: 6.8.0-1046-aws
- Python: Python 3.12.13
- Conda env: main at /venv/main

## Key Packages
```
accelerate              1.13.0
bitsandbytes            0.49.2
huggingface_hub         1.7.1
numpy                   2.4.3
peft                    0.18.1
pillow                  12.1.1
PyYAML                  6.0.3
safetensors             0.7.0
tokenizers              0.22.2
torch                   2.10.0+cu130
torchaudio              2.10.0+cu130
torchcodec              0.10.0+cu130
torchdata               0.10.0
torchtext               0.6.0
torchvision             0.25.0+cu130
tqdm                    4.67.3
transformers            5.3.0
```

## Full pip freeze
```
accelerate==1.13.0
annotated-doc==0.0.4
anyio==4.12.1
asttokens==3.0.1
av==17.0.0
bitsandbytes==0.49.2
certifi==2026.2.25
charset-normalizer==3.4.6
click==8.3.1
comm==0.2.3
cuda-bindings==13.0.3
cuda-pathfinder==1.4.3
debugpy==1.8.20
decorator==5.2.1
executing==2.2.1
filelock==3.25.2
fsspec==2026.2.0
h11==0.16.0
hf-xet==1.4.2
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.7.1
idna==3.11
ipykernel==7.2.0
ipython==9.11.0
ipython_pygments_lexers==1.1.1
ipywidgets==8.1.8
jedi==0.19.2
Jinja2==3.1.6
jupyter_client==8.8.0
jupyter_core==5.9.1
jupyterlab_widgets==3.0.16
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib-inline==0.2.1
mdurl==0.1.2
mpmath==1.3.0
nest-asyncio==1.6.0
networkx==3.6.1
numpy==2.4.3
nvidia-cublas==13.1.0.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.15.1.9
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.0
nvidia-nccl-cu13==2.28.9
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
packaging @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_packaging_1769093650/work
parso==0.8.6
peft==0.18.1
pexpect==4.9.0
pillow==12.1.1
platformdirs==4.9.4
prompt_toolkit==3.0.52
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
Pygments==2.19.2
python-dateutil==2.9.0.post0
PyYAML==6.0.3
pyzmq==27.1.0
qwen-vl-utils==0.0.14
regex==2026.2.28
requests==2.32.5
rich==14.3.3
safetensors==0.7.0
sentencepiece==0.2.1
setuptools==82.0.1
shellingham==1.5.4
six==1.17.0
stack-data==0.6.3
sympy==1.14.0
tokenizers==0.22.2
torch==2.10.0+cu130
torchaudio==2.10.0+cu130
torchcodec==0.10.0+cu130
torchdata==0.10.0
torchtext==0.6.0
torchvision==0.25.0+cu130
tornado==6.5.5
tqdm==4.67.3
traitlets==5.14.3
transformers==5.3.0
triton==3.6.0
typer==0.24.1
typing_extensions==4.15.0
urllib3==2.6.3
wcwidth==0.6.0
wheel==0.46.3
widgetsnbextension==4.0.15
```

## Model Downloads

```bash
# Qwen2.5-VL-3B-Instruct
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir='/root/models/Qwen2.5-VL-3B-Instruct')"

# Qwen2.5-VL-7B-Instruct
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='/root/models/Qwen2.5-VL-7B-Instruct')"

# Qwen2.5-VL-32B-Instruct
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-32B-Instruct', local_dir='/root/models/Qwen2.5-VL-32B-Instruct')"
```

## DriveLM nuScenes Dataset

```bash
# 1. Login to HuggingFace (requires gated access to OpenDriveLab/DriveLM)
huggingface-cli login

# 2. Download DriveLM v1.1 QA dataset
python -c "from huggingface_hub import snapshot_download; snapshot_download('OpenDriveLab/DriveLM', repo_type='dataset', local_dir='data/QA_dataset_nus')"

# 3. Download nuScenes CAM_FRONT images (from nuScenes mini or full)
# Option A: From HuggingFace mirror
python -c "from huggingface_hub import snapshot_download; snapshot_download('OpenDriveLab/DriveLM', repo_type='dataset', allow_patterns='nuscenes/samples/CAM_FRONT/*', local_dir='data')"

# Option B: From nuScenes official site (requires account)
# https://www.nuscenes.org/download → download samples (CAM_FRONT only needed)
# Extract to data/nuscenes/samples/CAM_FRONT/

# 4. Convert to training format
python scripts/convert_data.py
# Output: data_processed/train.json (359k), val.json (19k), train_mini.json (500)

# 5. Create 1/5 subset (for teacher/distillation training)
python -c "
import json, random
random.seed(42)
train = json.load(open('data_processed/train.json'))
random.shuffle(train)
json.dump(train[:len(train)//5], open('data_processed/train_1_5.json','w'))
print(f'Saved {len(train)//5} samples')
"
```
