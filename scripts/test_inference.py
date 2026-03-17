"""Verify inference pipeline works with Qwen3.5-4B (natively multimodal)."""
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import torch

model_id = "Qwen/Qwen3.5-4B"

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)
print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Use a sample image from DriveLM repo
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_image = os.path.join(_base, "assets/images/repo/drivelm_teaser.jpg")
image = Image.open(test_image).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the driving scene. What objects are present and what action should the ego vehicle take?"},
        ],
    }
]

# Qwen3.5: use apply_chat_template with tokenize=True for end-to-end processing
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    images=[image],
).to(model.device)

print("Running inference...")
output_ids = model.generate(**inputs, max_new_tokens=256)
output_text = processor.batch_decode(
    output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
)

print("\n" + "=" * 60)
print("MODEL OUTPUT:")
print("=" * 60)
print(output_text[0])
print("=" * 60)
print(f"\nGPU memory after inference: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print("\nInference test PASSED!")
