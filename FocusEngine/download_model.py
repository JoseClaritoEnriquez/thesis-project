from huggingface_hub import snapshot_download
import os

model_id = "unsloth/Qwen3.5-2B-GGUF"
target_dir = os.path.join(os.getcwd(), "model_qwen35_gguf")

print(f"Downloading Qwen 3.5 2B GGUF model and vision tower...")
print(f"Target directory: {target_dir}")

os.makedirs(target_dir, exist_ok=True)

# We download specific files for efficiency
files_to_download = [
    "Qwen3.5-2B-Q4_K_M.gguf",
    "mmproj-F16.gguf"
]

from huggingface_hub import hf_hub_download

for filename in files_to_download:
    print(f"Fetching {filename}...")
    hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )

print("\nDownload complete! You can now run the Focus Engine.")
