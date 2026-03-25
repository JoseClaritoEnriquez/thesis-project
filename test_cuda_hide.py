import os
import sys
import time

# HIDE THE GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def _ensure_win32_cuda_dlls():
    if sys.platform != "win32": return
    import os
    search_paths = set()
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        bin_path = os.path.join(cuda_path, "bin")
        if os.path.isdir(bin_path): search_paths.add(bin_path)
    base_cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(base_cuda_dir):
        try:
            for version_dir in os.listdir(base_cuda_dir):
                bin_path = os.path.join(base_cuda_dir, version_dir, "bin")
                if os.path.isdir(bin_path): search_paths.add(bin_path)
        except Exception: pass
    for path in search_paths:
        try:
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"): os.add_dll_directory(path)
        except Exception: pass

_ensure_win32_cuda_dlls()

from llama_cpp import Llama

print("--- Testing with CUDA_VISIBLE_DEVICES=-1 ---")
try:
    llm = Llama(
        model_path=r"C:\Users\user\Documents\model\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=0,
        verbose=True
    )
    print("Success: Initialized without CUDA interference.")
except Exception as e:
    print(f"Error: {e}")
