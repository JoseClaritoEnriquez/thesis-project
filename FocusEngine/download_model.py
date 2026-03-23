from huggingface_hub import snapshot_download
import os

model_id = "helenai/Qwen3-VL-2B-Instruct-int4"
target_dir = os.path.join(os.getcwd(), "model_qwen3_ov_int4")

print(f"Downloading complete OpenVINO model {model_id}...")
print(f"Target directory: {target_dir}")

os.makedirs(target_dir, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False
)

print("\nDownload complete! You can now run the Focus Engine.")
