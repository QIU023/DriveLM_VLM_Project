# TRT-LLM 1.3.0rc15 Native Install — Honest Blocker Report

**Date**: 2026-05-23  
**Status**: ❌ Blocked. Needs ~half-day clean reinstall via NVIDIA Docker image, OR upgrade host to torch 2.10 + cuda 13.1.1 baseline (cascade risk to all SFT pipelines).

## Why we need TRT-LLM 1.3.0rc15

- **TRT-LLM 1.2.1 has NO `modeling_qwen3vl.py`** — Qwen3-VL backbone we use for B.5'' cannot be deployed natively
- **TRT-LLM 1.3+ adds `modeling_qwen3vl.py`** — required for native Qwen3-VL TRT engine build
- See: GitHub issues #10069, #12824 (NVIDIA still hasn't added `modeling_qwen2_5_vl.py`)

## Install attempts (15 iterations, all failed)

| Attempt | Approach | Error |
|---|---|---|
| v1 | `pip install --upgrade tensorrt-llm==1.3.0rc15` (full deps) | `cuda-python 13.0.0` ↔ `cuda-bindings~=13.0.0` ResolutionImpossible |
| v2 | `--no-deps` in existing /opt/trt_venv | undefined symbol `_ZNK3c1010TensorImpl15decref_pyobjectEv` (torch ABI mismatch — needed torch 2.10, had 2.9.1) |
| v3 | Upgrade torch in existing venv 2.9.1→2.10 | `ModuleNotFoundError: AutoProcessor` (transformers broken) |
| v4 | Restore 1.2.1 + create fresh `/opt/trt_venv_13` | fresh torch 2.10 install OK |
| v5 | `--no-deps tensorrt-llm` in fresh venv | `ModuleNotFoundError: transformers` |
| v6 | `transformers==4.57.0` normal deps | `No module named 'httpcore'` |
| v7 | `transformers==4.57.0` then `import tensorrt_llm` | `No module named 'nvtx'` |
| v8 | + `nvtx` | `No module named 'tensorrt'` (the actual TRT lib) |
| v9 | + `tensorrt` | `No module named 'blake3'` |
| v10 | + `blake3 strenum click jsonschema lark openai aiohttp sse-starlette uvicorn fastapi` | `No module named 'pydantic'` |
| v11 | + `pydantic` | `No module named 'soundfile'` |
| v12 | + `soundfile librosa` | `No module named 'torchvision'` |
| v13 | + `torchvision` `--no-deps` | `No module named 'onnx'` |
| v14 | + `onnx onnx_graphsurgeon polygraphy janus etcd3 ray einops sentencepiece scipy matplotlib pandas wandb` | `No module named 'modelopt'` |
| v15 | + `nvidia-modelopt` | `No module named 'h5py'` (Gemma model registration) |
| v16 | + `h5py xgrammar` | `No module named 'llist'` |
| v17 | + `llist` | `ImportError: cannot import name 'FlashInferAttentionMetadata'` |
| v18 | + `flashinfer-python` | **Upgraded torch 2.10 → 2.12 as transitive dep, broke torch ABI again** |
| v19 | Revert torch 2.12 → 2.10 (uninstall flashinfer) | `PackageNotFoundError: nvidia-cuda-tileiras` (another transitive) |

## Pattern

TRT-LLM 1.3.0rc15 has ~50+ transitive deps that:
- Include heavy native cuda libs (flashinfer, cuda-python, cuda-bindings) with **tight torch ABI requirements**
- Each install fixes one ImportError but pulls deps that break another component
- `cuda-python==13.0` vs `cuda-bindings~=13.0.0` is a published conflict in 1.3.0rc15's deps

## Recommended path forward (~half day)

1. **Use NVIDIA's official TRT-LLM Docker image** (likely `nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3` or similar):
   ```
   docker pull nvcr.io/nvidia/tensorrt-llm:1.3-py3
   docker run --gpus all -it -v $PWD:/workspace ...
   ```
   - Docker image has all deps pre-resolved against a known torch ABI
   - Cost: ~5-10GB download + ~30min container setup
2. **Inside Docker**: `trtllm-build --checkpoint_dir ./checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final --output_dir engines/b5pp_qwen3vl --gemm_plugin auto`
3. **Test inference**: `trtllm-serve engines/b5pp_qwen3vl --port 8000`
4. **Bench vs HF baseline**: `deploy/bench_hf_baseline.py` already has the comparison framework

## What works NOW (stable fallback)

- `/opt/trt_venv` with **TRT-LLM 1.2.1** — installs cleanly, `import tensorrt_llm` works
- Cannot build engine for Qwen2.5-VL OR Qwen3-VL (no `modeling_*.py`)
- Useful only for Qwen2-VL (different model) reference benchmarks

## Time accounting (this session)

- TRT install attempts: ~2h consumed across multiple cascading failures
- Verdict: pip-based install is hostile. Docker is the production path.
- Deferred to next session with Docker setup.
