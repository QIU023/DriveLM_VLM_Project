"""Download only the nuScenes subset images (QA json already downloaded)."""
from huggingface_hub import hf_hub_download

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

print("Downloading nuScenes subset images (~3.5GB)...")
img_path = hf_hub_download(
    repo_id="OpenDriveLab/DriveLM",
    filename="drivelm_nus_imgs_train.zip",
    repo_type="dataset",
    local_dir=DATA_DIR,
)
print(f"Images zip downloaded to: {img_path}")
print("Done!")
