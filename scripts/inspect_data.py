"""Inspect DriveLM QA JSON structure."""
import json

import os
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_base, "data/QA_dataset_nus/v1_1_train_nus.json"), "r") as f:
    data = json.load(f)

scenes = list(data.keys())
print(f"Total scenes: {len(scenes)}")
print(f"First scene key: {scenes[0]}")

scene = data[scenes[0]]
print(f"Keys in scene: {list(scene.keys())}")
print(f"Scene description: {scene.get('scene_description', 'N/A')[:120]}")

frames = scene.get("key_frames", {})
print(f"Key frames in this scene: {len(frames)}")

frame_key = list(frames.keys())[0]
frame = frames[frame_key]
print(f"\nFrame keys: {list(frame.keys())}")

qa = frame.get("QA", {})
print(f"QA categories: {list(qa.keys())}")
for cat in qa:
    print(f"  {cat}: {len(qa[cat])} QA pairs")

print(f"\nImage paths: {json.dumps(frame.get('image_paths', {}), indent=2)}")

# Show first QA pair from each category
for cat in qa:
    if qa[cat]:
        first = qa[cat][0]
        print(f"\n--- {cat} sample ---")
        print(f"Q: {first.get('Q', '')}")
        print(f"A: {first.get('A', '')[:200]}")

# Count total QA pairs across all scenes
total_qa = 0
total_frames = 0
for s in data.values():
    for f in s.get("key_frames", {}).values():
        total_frames += 1
        for cat_pairs in f.get("QA", {}).values():
            total_qa += len(cat_pairs)
print(f"\n=== TOTALS ===")
print(f"Scenes: {len(scenes)}, Frames: {total_frames}, QA pairs: {total_qa}")
