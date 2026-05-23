"""Test TRT-LLM LLM API direct load of B.5'' (Qwen3-VL-4B) ckpt.

If this works, no separate trtllm-build conversion needed - the new pytorch
backend serves HF ckpts directly.
"""

def main():
    ckpt = "/workspace/DriveLM_VLM_Project/checkpoints_qwen25/nusc_planning_b5pp_1cam_qwen3vl_multimodal/final"
    print(f"loading {ckpt} via TRT-LLM LLM API...")

    from tensorrt_llm import LLM
    llm = LLM(model=ckpt, dtype="bfloat16", tensor_parallel_size=1)
    print("OK: LLM instantiated")

    out = llm.generate(["Hello, what is the next driving maneuver?"])
    print(f"Generation: {out[0].outputs[0].text[:200]}")


if __name__ == "__main__":
    main()
