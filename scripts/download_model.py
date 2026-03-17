"""Download Qwen2.5-VL-3B-Instruct model."""
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

print(f"Downloading model: {model_id}")
print("This will download ~6GB of model weights...")

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

print("Model loaded successfully!")
print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Model device: {model.device}")
