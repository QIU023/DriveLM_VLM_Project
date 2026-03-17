"""Download DriveLM QA JSON and nuScenes subset images from HuggingFace."""
from huggingface_hub import hf_hub_download
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# 1. Download QA json
print("Downloading DriveLM QA json...")
qa_path = hf_hub_download(
    repo_id="OpenDriveLab/DriveLM",
    filename="v1_1_train_nus.json",
    repo_type="dataset",
    local_dir=os.path.join(DATA_DIR, "QA_dataset_nus"),
)
print(f"QA json downloaded to: {qa_path}")

# 2. Download nuScenes subset images (zip)
print("Downloading nuScenes subset images (this may take a while)...")
img_path = hf_hub_download(
    repo_id="OpenDriveLab/DriveLM",
    filename="drivelm_nus_imgs_train.zip",
    repo_type="dataset",
    local_dir=DATA_DIR,
)
print(f"Images zip downloaded to: {img_path}")
print("Done! Next step: unzip the images.")
