"""Step 5: Convert DriveLM QA data to Qwen-VL conversation format for fine-tuning."""
import json
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = "/root/datasets/DriveLM"
QA_JSON = os.path.join(DATASET_ROOT, "v1_1_train_nus.json")
IMAGE_ROOT = os.path.join(DATASET_ROOT, "nuscenes/samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(QA_JSON, "r") as f:
    data = json.load(f)

training_data = []
skipped = 0

for scene_token, scene in data.items():
    scene_desc = scene.get("scene_description", "")
    for frame_token, frame in scene.get("key_frames", {}).items():
        image_paths = frame.get("image_paths", {})
        # Use CAM_FRONT as primary image for single-image fine-tuning
        cam_front_rel = image_paths.get("CAM_FRONT", "")
        if not cam_front_rel:
            skipped += 1
            continue

        # Convert relative path to absolute
        # Relative path looks like: ../nuscenes/samples/CAM_FRONT/xxx.jpg
        parts = cam_front_rel.replace("\\", "/").split("/")
        cam_dir = parts[-2] if len(parts) >= 2 else "CAM_FRONT"
        filename = parts[-1]
        cam_front_abs = os.path.join(IMAGE_ROOT, cam_dir, filename)
        if not os.path.exists(cam_front_abs):
            skipped += 1
            continue

        qa_data = frame.get("QA", {})
        for category in ["perception", "prediction", "planning", "behavior"]:
            for qa_pair in qa_data.get(category, []):
                question = qa_pair.get("Q", "").strip()
                answer = qa_pair.get("A", "").strip()
                if not question or not answer:
                    continue

                # Add driving context prefix for non-behavior questions
                if category != "behavior":
                    system_context = f"You are an autonomous driving assistant analyzing a driving scene. Category: {category}."
                else:
                    system_context = "You are an autonomous driving assistant. Predict the ego vehicle behavior."

                conversation = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_context,
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"file://{cam_front_abs}"},
                                {"type": "text", "text": question},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": answer,
                        },
                    ],
                    "metadata": {
                        "scene_token": scene_token,
                        "frame_token": frame_token,
                        "category": category,
                        "scene_description": scene_desc,
                    },
                }
                training_data.append(conversation)

print(f"Total training samples: {len(training_data)}")
print(f"Skipped frames (missing images): {skipped}")

# Shuffle and split train/val (95/5)
random.seed(42)
random.shuffle(training_data)
split_idx = int(len(training_data) * 0.95)
train_split = training_data[:split_idx]
val_split = training_data[split_idx:]

# Save full dataset
train_path = os.path.join(OUTPUT_DIR, "train.json")
val_path = os.path.join(OUTPUT_DIR, "val.json")
with open(train_path, "w") as f:
    json.dump(train_split, f, indent=2)
with open(val_path, "w") as f:
    json.dump(val_split, f, indent=2)

# Save a small subset for quick testing (500 samples)
mini_train = train_split[:500]
mini_path = os.path.join(OUTPUT_DIR, "train_mini.json")
with open(mini_path, "w") as f:
    json.dump(mini_train, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}:")
print(f"  train.json: {len(train_split)} samples")
print(f"  val.json: {len(val_split)} samples")
print(f"  train_mini.json: {len(mini_train)} samples (for quick test)")

# Category distribution
from collections import Counter
cat_dist = Counter(s["metadata"]["category"] for s in training_data)
print(f"\nCategory distribution:")
for cat, count in cat_dist.most_common():
    print(f"  {cat}: {count} ({count/len(training_data)*100:.1f}%)")
