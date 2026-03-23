import os
import sys
import json
import re
import threading
import pyautogui
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download

import config_manager

# --- Professional DLL Discovery for OpenVINO ---
if getattr(sys, 'frozen', False):
    ov_libs = os.path.join(sys._MEIPASS, "_internal", "openvino", "libs")
    if os.path.exists(ov_libs):
        try:
            os.add_dll_directory(ov_libs)
        except Exception as e:
            print(f"DLL Discovery Warning: {e}")

# Custom Progress Reporter to pipe Hugging Face status to the UI
class ProgressReporter:
    _callback = None
    _lock = threading.Lock()
    
    @classmethod
    def get_lock(cls): return cls._lock
    @classmethod
    def set_lock(cls, lock): cls._lock = lock

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get('total', 0)
        self.n = 0
        self.desc = kwargs.get('desc', 'Fetching')

    def update(self, n=1):
        self.n += n
        if ProgressReporter._callback:
            if self.total > 0:
                pct = int((self.n / self.total) * 100)
                ProgressReporter._callback(f"{self.desc}: {pct}%")
            else:
                ProgressReporter._callback(f"{self.desc}: {self.n} units")

    def close(self): pass
    def refresh(self): pass
    def clear(self, *args, **kwargs): pass
    def set_description(self, desc): self.desc = desc

class FocusEngineBackend:
    """Base class for vision-language backends."""
    def generate(self, prompt, image=None):
        raise NotImplementedError

class OpenVINOBackend(FocusEngineBackend):
    def __init__(self, model_path, device, report_callback):
        self.model_path = model_path
        self.device = device
        self.report = report_callback
        
        import openvino_genai as ov_genai
        import openvino as ov
        self.ov = ov
        self.ov_genai = ov_genai
        
        self.report(f"Loading OpenVINO GenAI pipeline on {device}...")
        ov_config = {}
        if "GPU" in device.upper():
            ov_config = {"CACHE_DIR": os.path.join(self.model_path, "ov_cache")}
        
        try:
            self.pipeline = ov_genai.VLMPipeline(self.model_path, device, **ov_config)
        except Exception as e:
            if device.upper() != "CPU":
                self.report(f"Warning: Failed to load on {device}: {e}. Falling back to CPU...")
                self.pipeline = ov_genai.VLMPipeline(self.model_path, "CPU")
                config_manager.set_device("CPU")
            else:
                raise e

    def generate(self, prompt, image=None):
        ov_image = None
        if image is not None:
            img_np = np.array(image.convert("RGB")).copy()
            ov_image = self.ov.Tensor(img_np)
        
        config = self.ov_genai.GenerationConfig()
        config.max_new_tokens = 200
        config.repetition_penalty = 1.15
        
        if ov_image is not None:
            raw_output = self.pipeline.generate(prompt, image=ov_image, generation_config=config)
        else:
            raw_output = self.pipeline.generate(prompt, generation_config=config)
        
        if hasattr(raw_output, 'texts') and len(raw_output.texts) > 0:
            return raw_output.texts[0]
        return str(raw_output)

class LlamaCppBackend(FocusEngineBackend):
    def __init__(self, model_path, device, report_callback):
        self.model_path = model_path
        self.device = device
        self.report = report_callback
        
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen2VLChatHandler
        
        # Look for mmproj file in the model directory
        mmproj_path = None
        for f in os.listdir(model_path):
            if "mmproj" in f.lower() and f.endswith(".gguf"):
                mmproj_path = os.path.join(model_path, f)
                break
        
        if not mmproj_path:
            self.report("Warning: No mmproj file found. Vision tasks will fail.")
        
        # Find the main model file
        llm_path = None
        for f in os.listdir(model_path):
            if "qwen" in f.lower() and f.endswith(".gguf") and "mmproj" not in f.lower():
                llm_path = os.path.join(model_path, f)
                break
        
        if not llm_path:
            raise RuntimeError(f"No Qwen GGUF model found in {model_path}")

        self.report(f"Loading LlamaCpp (GGUF) on {device}...")
        
        # Map device to n_gpu_layers
        n_gpu_layers = -1 if "GPU" in device.upper() or "CUDA" in device.upper() else 0
        
        chat_handler = Qwen2VLChatHandler(clip_model_path=mmproj_path) if mmproj_path else None
        
        self.llm = Llama(
            model_path=llm_path,
            chat_handler=chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            verbose=False
        )
        self.report("LlamaCpp backend initialized.")

    def generate(self, prompt, image=None):
        import base64
        from io import BytesIO
        
        if image:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{img_str}"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=200,
            temperature=0.1
        )
        
        return response["choices"][0]["message"]["content"]

class FocusEngine:
    def __init__(self, model_path=None, status_callback=None, device=None, backend=None):
        def _report(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        _report(f"Initializing FocusEngine...")
        
        self.model_path = model_path or config_manager.get_model_path()
        self.device = device or config_manager.get_device()
        self.backend_type = backend or config_manager.get_backend()

        # Handle auto-detection
        if self.backend_type == "auto":
            try:
                import llama_cpp
                _report("llama-cpp-python detected. Using GGUF backend.")
                self.backend_type = "llama-cpp"
            except ImportError:
                _report("LlamaCpp not installed. Falling back to OpenVINO.")
                self.backend_type = "openvino"
            except Exception as e:
                _report(f"Error during backend auto-detection: {e}. Defaulting to OpenVINO.")
                self.backend_type = "openvino"

        if not self.model_path:
            raise RuntimeError("Model path not configured.")

        try:
            if self.backend_type == "openvino":
                # Check if model exists AND weights are present, if not download it
                model_xml = os.path.join(self.model_path, "openvino_language_model.xml")
                model_bin = os.path.join(self.model_path, "openvino_language_model.bin")
                
                if not os.path.exists(model_xml) or not os.path.exists(model_bin):
                    _report(f"Starting Qwen update...")
                    os.makedirs(self.model_path, exist_ok=True)
                    
                    repo_id = "helenai/Qwen2.5-VL-3B-Instruct-ov-int4"
                    
                    large_files = [
                        "openvino_language_model.bin", 
                        "openvino_text_embeddings_model.bin",
                        "openvino_vision_embeddings_merger_model.bin"
                    ]
                    
                    for i, filename in enumerate(large_files):
                        dest = os.path.join(self.model_path, filename)
                        if not os.path.exists(dest):
                            _report(f"Downloading {filename} ({i+1}/{len(large_files)})...")
                            hf_hub_download(
                                repo_id=repo_id,
                                filename=filename,
                                local_dir=self.model_path,
                                local_dir_use_symlinks=False,
                                progress_callback=ProgressReporter().update
                            )
                    
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=self.model_path,
                        local_dir_use_symlinks=False,
                        tqdm_class=ProgressReporter
                    )
                    _report("Download complete.")
                
                self.backend = OpenVINOBackend(self.model_path, self.device, _report)
            elif self.backend_type == "llama-cpp":
                # Check for GGUF files
                gguf_exists = any(f.endswith(".gguf") for f in os.listdir(self.model_path)) if os.path.exists(self.model_path) else False
                if not gguf_exists:
                    _report(f"Starting Qwen 3.5 2B GGUF download (~2GB)...")
                    os.makedirs(self.model_path, exist_ok=True)
                    
                    repo_id = "unsloth/Qwen3.5-2B-GGUF"
                    target_files = [
                        "qwen3.5-2b-instruct-q4_k_m.gguf",
                        "mmproj-F16.gguf" # Changed to F16 as a more common standard, search said F32 but F16 is fine
                    ]
                    
                    for filename in target_files:
                        _report(f"Downloading {filename}...")
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            local_dir=self.model_path,
                            local_dir_use_symlinks=False,
                            progress_callback=ProgressReporter().update
                        )
                    _report("Download complete.")

                self.backend = LlamaCppBackend(self.model_path, self.device, _report)
            else:
                raise ValueError(f"Unknown backend type: {self.backend_type}")

            _report("Focus Engine Initialized successfully.")
        except Exception as e:
            _report(f"Error: Failed to load or download model: {e}")
            raise RuntimeError(f"Failed to initialize Focus Engine: {e}") from e

    def capture_and_process(self, user_task):
        """
        Captures the screen, parses it through the VLM in two passes, 
        and returns the alignment status.
        """
        print("Capturing screenshot...")
        screenshot = pyautogui.screenshot()
        max_side = 960
        w, h = screenshot.size
        scale = min(max_side / w, max_side / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        screenshot = screenshot.resize(new_size, Image.LANCZOS)
        
        try:
            # Pass 1: Vision - Objective Description
            print("Pass 1: Identifying screen content...")
            vision_prompt = (
                "Task: Objectively describe the main activity on this screen. "
                "Output Format: Exactly 2 sentences. "
                "Sentence 1: Active application and main activity. "
                "Sentence 2: Specific visual detail confirming the activity. "
                "Negative Constraint: DO NOT use headers, bullets, or labels like 'key points:'."
            )
            description = self.run_inference(vision_prompt, image=screenshot)
            
            if isinstance(description, dict) and description.get("status") == "ERROR":
                print(f"Pass 1 failed: {description.get('reasoning')}")
                return description

            print(f"Screen Description: {description}")

            # Pass 2: Logic - Verdict based on Description (TEXT ONLY)
            print("Pass 2: Evaluating alignment with task (pure text logic)...")
            verdict_prompt = (
                "Task: Determine if the <ActualActivity> matches the <IntendedTask>.\n\n"
                f"<IntendedTask>{user_task}</IntendedTask>\n"
                f"<ActualActivity>{description}</ActualActivity>\n\n"
                "Instructions:\n"
                "1. Answer with 'Verdict: MATCH' or 'Verdict: MISMATCH'."
            )
            print(f"Pass 2 Input Prompt:\n{verdict_prompt}")
            raw_verdict = self.run_inference(verdict_prompt, image=None)
            print(f"Raw Verdict: {raw_verdict}")

            # 3. Clean up RAM
            import gc
            del screenshot
            gc.collect()

            return self._parse_response(raw_verdict, user_task)
        except Exception as e:
            print(f"Error in two-pass processing: {e}")
            return {"status": "ERROR", "reasoning": str(e)}

    def run_inference(self, prompt_text, image=None):
        try:
            print(f"Running {self.backend_type} inference (image={'PRESENT' if image else 'NONE'})...")
            return self.backend.generate(prompt_text, image)
        except Exception as e:
            print(f"Inference error: {e}")
            return {"status": "ERROR", "reasoning": str(e)}

    def _parse_response(self, raw_output, user_task=None):
        """Extract status and reasoning from model output."""
        status = "ALIGNED"  # default
        reasoning = raw_output 

        matches = list(re.finditer(r"(?:Verdict|Analysis|Answer|Final Verdict):\s*(MATCH|MISMATCH|ALIGNED|UNALIGNED)", raw_output, re.IGNORECASE))
        if matches:
            verdict = matches[-1].group(1).upper()
            if verdict in ["MATCH", "ALIGNED"]:
                status = "ALIGNED"
            else:
                status = "UNALIGNED"
            return {"status": status, "reasoning": reasoning}

        if any(kw in raw_output.upper() for kw in ["MISMATCH", "UNALIGNED", "OFF-TASK", "DISTRACT"]):
            status = "UNALIGNED"
        
        return {"status": status, "reasoning": reasoning}

if __name__ == "__main__":
    # Quick test function
    print("Testing Focus Engine Initialization...")
    engine = FocusEngine()
    print("Testing Capture and Process...")
    res = engine.capture_and_process("Writing Python code")
    print(f"Result: {res}")
