"""Step 5b (video): Convert DriveLM QA data to Qwen2.5-VL *multi-frame video* conversation format.

This mirrors `convert_data.py` but, instead of a single CAM_FRONT image per QA,
collects the **N most recent CAM_FRONT frames** (sorted by filename timestamp)
from the same scene, ending at the current key frame.

Why: Qwen2.5-VL accepts `videos=[ [frame0, frame1, ..., frameN-1] ]` as input,
and produces temporally-aware visual tokens. This is the Tier-1 video data prep.

CLI:
    python scripts/convert_data_video.py --num-frames 4 --frame-stride 1
    python scripts/convert_data_video.py --num-frames 8 --output v1_1_video_n8.json
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "data")
QA_JSON_DEFAULT = os.path.join(DATASET_ROOT, "QA_dataset_nus", "v1_1_train_nus.json")
IMAGE_ROOT_DEFAULT = os.path.join(DATASET_ROOT, "nuscenes", "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_processed")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-frames", type=int, default=4,
                   help="Number of frames per video clip (N). Frames = previous N-1 + current.")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Stride between sampled frames within the sequence. "
                        "1 = consecutive CAM_FRONT samples, 2 = every other, etc.")
    p.add_argument("--qa-json", type=str, default=QA_JSON_DEFAULT,
                   help="Path to DriveLM QA json (v1_1_train_nus.json).")
    p.add_argument("--image-root", type=str, default=IMAGE_ROOT_DEFAULT,
                   help="Root dir holding nuscenes/samples/CAM_FRONT/*.jpg etc.")
    p.add_argument("--output", type=str, default=None,
                   help="Output filename (under data_processed/). Defaults to "
                        "v1_1_video_n{N}.json")
    p.add_argument("--split", action="store_true",
                   help="Also emit train_video_n{N}.json + val_video_n{N}.json (95/5).")
    p.add_argument("--mini-size", type=int, default=500,
                   help="If --split, also write a train_video_mini.json of this size.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only process the first K scenes (debug).")
    return p.parse_args()


def collect_cam_front_index(image_root):
    """Walk image_root/CAM_FRONT and build {scene_prefix: sorted_list_of_paths}."""
    cam_dir = os.path.join(image_root, "CAM_FRONT")
    index = defaultdict(list)
    if not os.path.isdir(cam_dir):
        return index, 0
    files = [f for f in os.listdir(cam_dir) if f.endswith(".jpg")]
    for fn in files:
        # Format: <log_token>__CAM_FRONT__<timestamp>.jpg
        # Group by log_token so that the temporal order is correct *per drive*.
        parts = fn.split("__")
        if len(parts) < 3:
            continue
        log_token = parts[0]
        index[log_token].append(fn)
    for log_token in index:
        # Filenames embed monotonically increasing unix-microseconds — string sort works.
        index[log_token].sort()
    return index, len(files)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(args.qa_json):
        print(f"ERROR: QA JSON not found at {args.qa_json}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.image_root):
        print(f"ERROR: image root not found at {args.image_root}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading QA JSON: {args.qa_json}")
    with open(args.qa_json, "r") as f:
        qa_data = json.load(f)

    print(f"Indexing CAM_FRONT frames under {args.image_root} ...")
    cam_index, total_jpgs = collect_cam_front_index(args.image_root)
    print(f"  -> {total_jpgs} CAM_FRONT jpgs across {len(cam_index)} logs")

    if total_jpgs == 0:
        print("ERROR: no CAM_FRONT jpgs found — nothing to do.", file=sys.stderr)
        sys.exit(2)

    N = args.num_frames
    stride = args.frame_stride
    output_name = args.output or f"v1_1_video_n{N}.json"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    out_records = []
    reasons = Counter()
    n_padded = 0

    scenes_done = 0
    for scene_token, scene in qa_data.items():
        scenes_done += 1
        if args.limit and scenes_done > args.limit:
            break
        scene_desc = scene.get("scene_description", "")
        for frame_token, frame in scene.get("key_frames", {}).items():
            image_paths = frame.get("image_paths", {})
            cam_front_rel = image_paths.get("CAM_FRONT", "")
            if not cam_front_rel:
                reasons["no_cam_front_in_qa"] += 1
                continue

            # Resolve current CAM_FRONT file
            parts = cam_front_rel.replace("\\", "/").split("/")
            cur_fn = parts[-1]
            cur_log = cur_fn.split("__")[0] if "__" in cur_fn else None
            if cur_log is None or cur_log not in cam_index:
                reasons["log_token_not_indexed"] += 1
                continue

            log_files = cam_index[cur_log]
            try:
                cur_idx = log_files.index(cur_fn)
            except ValueError:
                reasons["current_frame_missing_on_disk"] += 1
                continue

            # Walk back N-1 frames (with stride). Pad-replicate the earliest available.
            wanted = []
            padded_this = 0
            for k in range(N - 1, -1, -1):
                target_idx = cur_idx - k * stride
                if target_idx < 0:
                    target_idx = 0
                    padded_this += 1
                wanted.append(log_files[target_idx])

            # Absolute paths + existence check
            frame_paths = []
            missing = False
            for fn in wanted:
                p = os.path.join(args.image_root, "CAM_FRONT", fn)
                if not os.path.exists(p):
                    missing = True
                    break
                frame_paths.append(p)
            if missing:
                reasons["frame_file_missing"] += 1
                continue

            if padded_this > 0:
                n_padded += 1

            qa_dict = frame.get("QA", {})
            for category in ["perception", "prediction", "planning", "behavior"]:
                for qa_pair in qa_dict.get(category, []):
                    question = qa_pair.get("Q", "").strip()
                    answer = qa_pair.get("A", "").strip()
                    if not question or not answer:
                        reasons["empty_qa"] += 1
                        continue

                    if category != "behavior":
                        system_context = (
                            f"You are an autonomous driving assistant analyzing a "
                            f"short driving video. Category: {category}."
                        )
                    else:
                        system_context = (
                            "You are an autonomous driving assistant. Predict the "
                            "ego vehicle behavior from the recent driving video."
                        )

                    conversation = {
                        "messages": [
                            {"role": "system", "content": system_context},
                            {
                                "role": "user",
                                "content": [
                                    # NOTE: the train_lora.py video branch will materialize
                                    # these paths -> PIL frames before calling the processor.
                                    {"type": "video", "video": frame_paths},
                                    {"type": "text", "text": question},
                                ],
                            },
                            {"role": "assistant", "content": answer},
                        ],
                        "metadata": {
                            "scene_token": scene_token,
                            "frame_token": frame_token,
                            "category": category,
                            "scene_description": scene_desc,
                            "num_frames": N,
                            "frame_stride": stride,
                            "padded_frames": padded_this,
                        },
                        # Convenience top-level mirror for any downstream tool that
                        # only wants the frame list.
                        "image_paths": frame_paths,
                        "scene_token": scene_token,
                        "frame_token": frame_token,
                    }
                    out_records.append(conversation)

    kept = len(out_records)
    skipped = sum(reasons.values())
    print(f"\nKept: {kept}")
    print(f"Skipped: {skipped}")
    for r, c in reasons.most_common():
        print(f"  {r}: {c}")
    print(f"Records with >=1 pad-replicated frame: {n_padded}")

    if kept == 0:
        print("ERROR: no records produced — aborting write.", file=sys.stderr)
        sys.exit(3)

    with open(output_path, "w") as f:
        json.dump(out_records, f)
    print(f"\nWrote {output_path} ({kept} samples, {os.path.getsize(output_path)/1024**2:.1f} MB)")

    if args.split:
        random.seed(42)
        random.shuffle(out_records)
        split_idx = int(len(out_records) * 0.95)
        train_split = out_records[:split_idx]
        val_split = out_records[split_idx:]
        train_path = os.path.join(OUTPUT_DIR, f"train_video_n{N}.json")
        val_path = os.path.join(OUTPUT_DIR, f"val_video_n{N}.json")
        with open(train_path, "w") as f:
            json.dump(train_split, f)
        with open(val_path, "w") as f:
            json.dump(val_split, f)
        print(f"Wrote {train_path} ({len(train_split)} samples)")
        print(f"Wrote {val_path} ({len(val_split)} samples)")
        if args.mini_size > 0:
            mini = train_split[: args.mini_size]
            mini_path = os.path.join(OUTPUT_DIR, f"train_video_n{N}_mini.json")
            with open(mini_path, "w") as f:
                json.dump(mini, f)
            print(f"Wrote {mini_path} ({len(mini)} samples)")


if __name__ == "__main__":
    main()
