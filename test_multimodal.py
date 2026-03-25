import os
import sys
import time
import base64
from io import BytesIO
from PIL import Image

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
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

model_path = r"C:\Users\user\Documents\model\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
mmproj_path = r"C:\Users\user\Documents\model\mmproj-F16.gguf"

print("--- Starting Multimodal Test ---")
start_time = time.time()

print(f"Loading chat handler with mmproj: {mmproj_path}")
handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=True)

# Using the app's context size
print(f"Initializing Llama on CPU (n_gpu_layers=0, n_ctx=8192, n_threads=6)")
llm = Llama(
    model_path=model_path,
    chat_handler=handler,
    n_gpu_layers=0,
    n_ctx=8192,
    n_threads=6,
    verbose=True
)

print(f"Model loaded in {time.time() - start_time:.2f}s")

# Create a dummy image
img = Image.new('RGB', (960, 540), color = (73, 109, 137))
buffered = BytesIO()
img.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
image_url = f"data:image/jpeg;base64,{img_str}"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the primary activity in this screenshot."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }
]

print("Running multimodal inference...")
inf_start = time.time()
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=20
)
print(f"Inference completed in {time.time() - inf_start:.2f}s")
print("Response:", response["choices"][0]["message"]["content"])
print(f"Total time: {time.time() - start_time:.2f}s")
