import os
from huggingface_hub import hf_hub_download

MODEL_SPEC = {
    "display_name": "Qwen2.5-VL-3B",
    "repo_id": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
    "llm_filename": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
    "mmproj_filename": "mmproj-F16.gguf",
}
target_dir = os.path.join(os.getcwd(), "model_qwen25_vl_3b_gguf")

print(f"Downloading {MODEL_SPEC['display_name']} GGUF model and vision tower...")
print(f"Target directory: {target_dir}")

os.makedirs(target_dir, exist_ok=True)

# We download specific files for efficiency
files_to_download = [
    MODEL_SPEC["llm_filename"],
    MODEL_SPEC["mmproj_filename"],
]

for filename in files_to_download:
    print(f"Fetching {filename}...")
    hf_hub_download(
        repo_id=MODEL_SPEC["repo_id"],
        filename=filename,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )

print("\nDownload complete! You can now run the Focus Engine.")
