#!/bin/bash
# WSL2: Re-merge LoRAs + AWQ quantize (all in transformers 4.51.3)
set -e

PY=/home/yiqiao/miniconda3/envs/quant/bin/python
PIP=/home/yiqiao/miniconda3/envs/quant/bin/pip
BASE=/mnt/f/learning/2026Interview+Resume/VLM_projects/DriveLM

echo "=== Installing peft ==="
$PIP install peft -q

echo "=== Step 1: Re-merge LoRAs with transformers 4.51.3 ==="
$PY -c "
import torch, gc, os, time, json
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_DIR = '$BASE'
MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'

jobs = [
    ('baseline', os.path.join(BASE_DIR, 'checkpoints/baseline/checkpoint-46000')),
    ('fastervlm', os.path.join(BASE_DIR, 'checkpoints/fastervlm/final')),
    ('prumerge', os.path.join(BASE_DIR, 'checkpoints/prumerge/final')),
    ('pyramiddrop', os.path.join(BASE_DIR, 'checkpoints/pyramiddrop/final')),
]

for name, lora_path in jobs:
    out = os.path.join(BASE_DIR, 'models', f'qwen25vl-3b-drivelm-{name}-merged')
    print(f'\n==== Merging {name} ====')
    print(f'  LoRA: {lora_path}')
    print(f'  Out:  {out}')

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map='cpu')
    model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)
    model = model.merge_and_unload()

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)

    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    proc.save_pretrained(out)

    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith('.safetensors'))
    print(f'  Done: {size/1024**3:.2f} GB in {time.time()-t0:.0f}s')

    del model; gc.collect()
print('\n==== All merges done ====')
"

echo ""
echo "=== Step 2: AWQ quantize all 4 models ==="
cd $BASE
for name in baseline fastervlm prumerge pyramiddrop; do
    echo ""
    echo "==== AWQ: $name ===="
    $PY scripts/quantize_model.py awq --input models/qwen25vl-3b-drivelm-${name}-merged
done

echo ""
echo "=== All done ==="
