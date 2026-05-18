#!/usr/bin/env python3
"""Split DriveLM video-VLA records into train/val by SCENE (80/20).

Scene-level split is the only correct way: splitting by record would leak
multiple QA per scene across train and val, inflating val accuracy.

Deterministic: seeded shuffle of unique scene_tokens, then 80% train / 20% val.

Outputs:
    {input_stem}_train.json
    {input_stem}_val.json

Usage:
    python scripts/split_train_val.py \
        --input data_processed/v1_1_video_n4_FULL_with_traj.json \
        --val-ratio 0.2 --seed 42
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="path to *_with_traj.json")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.input)
    with inp.open() as f:
        data = json.load(f)
    print(f"loaded {len(data)} records")

    # Collect scenes
    scene_records = {}
    for r in data:
        sk = r.get("metadata", {}).get("scene_token")
        if not sk:
            raise ValueError(f"record missing scene_token: {r.get('metadata', {})}")
        scene_records.setdefault(sk, []).append(r)
    print(f"unique scenes: {len(scene_records)}")

    # Shuffle scenes deterministically, split
    scenes = sorted(scene_records.keys())
    rng = random.Random(args.seed)
    rng.shuffle(scenes)
    n_val_scenes = int(round(len(scenes) * args.val_ratio))
    val_scenes = set(scenes[:n_val_scenes])
    train_scenes = set(scenes[n_val_scenes:])

    train_records = [r for s in train_scenes for r in scene_records[s]]
    val_records = [r for s in val_scenes for r in scene_records[s]]

    # Stat: per-source breakdown
    def stat(records, name):
        srcs = Counter(r.get("metadata", {}).get("traj_source", "?") for r in records)
        return f"{name}: {len(records)} records, {len(set(r['metadata']['scene_token'] for r in records))} scenes, traj_source={dict(srcs)}"

    print(stat(train_records, "TRAIN"))
    print(stat(val_records, "VAL"))

    # Write
    train_path = inp.parent / f"{inp.stem}_train.json"
    val_path = inp.parent / f"{inp.stem}_val.json"
    with train_path.open("w") as f:
        json.dump(train_records, f)
    with val_path.open("w") as f:
        json.dump(val_records, f)

    print(f"wrote {train_path} ({train_path.stat().st_size/1e6:.0f} MB)")
    print(f"wrote {val_path} ({val_path.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
