#!/bin/bash
# Batch quantize all merged models in three formats (AWQ / GPTQ / GGUF)
#
# Run on 4070Ti (12GB VRAM) after batch_merge.py has created the merged models.
# Each quantization runs sequentially to avoid OOM.
#
# Usage:
#   bash scripts/batch_quantize.sh           # all 3 models × 3 formats = 9
#   bash scripts/batch_quantize.sh awq       # AWQ only
#   bash scripts/batch_quantize.sh gguf      # GGUF only
#   bash scripts/batch_quantize.sh baseline   # one model, all formats

set -e
cd "$(dirname "$0")/.."

MODELS=(baseline fastervlm prumerge)
METHOD_FILTER="${1:-all}"

echo "============================================================"
echo "Batch Quantize: DriveLM merged models"
echo "Filter: $METHOD_FILTER"
echo "============================================================"

quantize_one() {
    local name=$1
    local method=$2
    local input="models/qwen25vl-3b-drivelm-${name}-merged"

    if [ ! -d "$input" ]; then
        echo "[SKIP] $name: merged model not found at $input"
        return
    fi

    local output="models/qwen25vl-3b-drivelm-${name}-${method}"
    if [ -d "$output" ] && [ "$(ls -A "$output" 2>/dev/null)" ]; then
        echo "[SKIP] $name-$method: already exists at $output"
        return
    fi

    echo ""
    echo ">>> Quantizing: $name -> $method"
    if [ "$method" = "gguf" ]; then
        python scripts/quantize_model.py gguf --input "$input" --output "$output"
    else
        python scripts/quantize_model.py "$method" --input "$input" --output "$output"
    fi
}

for name in "${MODELS[@]}"; do
    # If filter is a model name, only process that model
    if [ "$METHOD_FILTER" != "all" ] && [ "$METHOD_FILTER" != "awq" ] && \
       [ "$METHOD_FILTER" != "gptq" ] && [ "$METHOD_FILTER" != "gguf" ] && \
       [ "$METHOD_FILTER" != "$name" ]; then
        continue
    fi

    echo ""
    echo "============ Model: $name ============"

    if [ "$METHOD_FILTER" = "all" ] || [ "$METHOD_FILTER" = "$name" ] || [ "$METHOD_FILTER" = "awq" ]; then
        quantize_one "$name" "awq"
    fi
    if [ "$METHOD_FILTER" = "all" ] || [ "$METHOD_FILTER" = "$name" ] || [ "$METHOD_FILTER" = "gptq" ]; then
        quantize_one "$name" "gptq"
    fi
    if [ "$METHOD_FILTER" = "all" ] || [ "$METHOD_FILTER" = "$name" ] || [ "$METHOD_FILTER" = "gguf" ]; then
        quantize_one "$name" "gguf"
    fi
done

echo ""
echo "============================================================"
echo "Batch quantize complete!"
echo ""
echo "Model sizes:"
for d in models/qwen25vl-3b-drivelm-*-{awq,gptq,gguf}; do
    if [ -d "$d" ]; then
        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        echo "  $d: $size"
    fi
done
echo "============================================================"
