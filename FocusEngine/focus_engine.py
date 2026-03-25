import os
import re
import subprocess
import threading
import inspect
import importlib

import pyautogui
from PIL import Image
from huggingface_hub import hf_hub_download

import config_manager

# Qwen-VL allocates a large image token block (often >=1024 tokens). n_ctx=2048 cannot
# fit image + prompt + generation and triggers decode failures and poisoned handler state.
LLAMA_VL_N_CTX = 8192


def _sanitize_model_text(text):
    """Remove internal reasoning tags and normalize whitespace."""
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # Some runtimes emit an opening <think> tag without closing it.
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

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

def _normalize_runtime(value):
    normalized = str(value or "auto").strip().lower()
    mapping = {
        "auto": "auto",
        "gpu": "auto",
        "cuda": "cuda",
        "nvidia": "cuda",
        "vulkan": "vulkan",
        "amd": "vulkan",
        "cpu": "cpu",
    }
    return mapping.get(normalized, "auto")


def detect_gpu_vendor():
    """
    Detect the primary GPU vendor on Windows.
    Returns: "nvidia", "amd", or None when unknown/not detected.
    """
    commands = [
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ],
        ["wmic", "path", "win32_VideoController", "get", "name"],
    ]

    output = ""
    for command in commands:
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=5)
            if output.strip():
                break
        except Exception:
            continue

    if not output:
        return None

    text = output.lower()
    if "nvidia" in text:
        return "nvidia"
    if "amd" in text or "radeon" in text:
        return "amd"
    return None


def resolve_runtime(configured_runtime, report_callback):
    runtime = _normalize_runtime(configured_runtime)
    if runtime != "auto":
        return runtime

    vendor = detect_gpu_vendor()
    if vendor == "nvidia":
        report_callback("Detected NVIDIA GPU. Selecting CUDA runtime.")
        return "cuda"
    if vendor == "amd":
        report_callback("Detected AMD GPU. Selecting Vulkan runtime.")
        return "vulkan"

    report_callback("No NVIDIA/AMD GPU detected. Falling back to CPU runtime.")
    return "cpu"


def find_qwen_vl_assets(model_path):
    if not os.path.exists(model_path):
        return None, None

    gguf_files = [f for f in os.listdir(model_path) if f.lower().endswith(".gguf")]
    mmproj_candidates = [f for f in gguf_files if "mmproj" in f.lower()]
    model_candidates = [
        f for f in gguf_files if "mmproj" not in f.lower() and "qwen3.5-2b" in f.lower()
    ]

    mmproj_path = os.path.join(model_path, mmproj_candidates[0]) if mmproj_candidates else None
    llm_path = os.path.join(model_path, model_candidates[0]) if model_candidates else None
    return llm_path, mmproj_path


def build_qwen_vl_chat_handler(mmproj_path, runtime):
    import llama_cpp

    chat_module = importlib.import_module("llama_cpp.llama_chat_format")
    # Prefer handlers known to work with the split GGUF+mmproj layout.
    candidate_names = ("Qwen25VLChatHandler", "Qwen2VLChatHandler", "Qwen3VLChatHandler")
    available = [name for name in candidate_names if hasattr(chat_module, name)]

    for name in available:
        handler_cls = getattr(chat_module, name)
        init_params = inspect.signature(handler_cls.__init__).parameters
        kwargs = {}
        if "clip_model_path" in init_params:
            kwargs["clip_model_path"] = mmproj_path
        if "use_gpu" in init_params:
            kwargs["use_gpu"] = runtime in ("cuda", "vulkan")
        if "verbose" in init_params:
            kwargs["verbose"] = False
        try:
            return handler_cls(**kwargs), name
        except TypeError:
            continue

    exported_qwen_vl = [
        symbol
        for symbol in dir(chat_module)
        if "qwen" in symbol.lower() and "vlchathandler" in symbol.lower()
    ]
    version = getattr(llama_cpp, "__version__", "unknown")
    available_text = ", ".join(exported_qwen_vl) if exported_qwen_vl else "none"
    raise RuntimeError(
        "Installed llama-cpp-python does not expose a supported Qwen VL chat handler "
        f"(detected version: {version}; available handlers: {available_text}). "
        "Run: python install_runtime.py --profile jamepeng"
    )

class LlamaCppBackend(FocusEngineBackend):
    def __init__(self, model_path, runtime, report_callback):
        self.model_path = model_path
        self.runtime = runtime
        self.report = report_callback

        from llama_cpp import Llama

        llm_path, mmproj_path = find_qwen_vl_assets(model_path)
        if not llm_path or not mmproj_path:
            raise RuntimeError(
                "Missing Qwen 3.5 2B VL assets. Required files: "
                "Qwen3.5-2B-Q4_K_M.gguf and mmproj-F16.gguf."
            )

        requested_gpu = runtime in ("cuda", "vulkan")
        n_gpu_layers = -1 if requested_gpu else 0
        chat_handler, handler_name = build_qwen_vl_chat_handler(mmproj_path, runtime)
        self._llm_path = llm_path
        self._mmproj_path = mmproj_path
        self._chat_handler = chat_handler
        self._active_n_gpu_layers = n_gpu_layers

        self.report(
            f"Loading llama.cpp backend (runtime={runtime}, n_gpu_layers={n_gpu_layers}, "
            f"chat_handler={handler_name})..."
        )
        try:
            self.llm = self._build_llm(n_gpu_layers)
            self.effective_runtime = runtime
        except Exception as e:
            if requested_gpu:
                self.report(
                    f"GPU runtime '{runtime}' failed to initialize ({e}). "
                    "Retrying with CPU runtime."
                )
                self.llm = self._build_llm(0)
                self.effective_runtime = "cpu"
            else:
                raise

        self.report(f"LlamaCpp backend initialized on runtime={self.effective_runtime}.")

    def _build_llm(self, n_gpu_layers):
        from llama_cpp import Llama

        self._active_n_gpu_layers = n_gpu_layers
        return Llama(
            model_path=self._llm_path,
            chat_handler=self._chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=LLAMA_VL_N_CTX,
            verbose=False,
        )

    def _inference_error_is_retryable(self, exc):
        return self._is_recoverable_decode_error(exc) or self._is_template_render_error(
            exc
        )

    def _reinitialize_backend(self):
        """
        Recreate the VL chat handler and Llama instance. Reusing only Llama after decode
        errors can leave the multimodal handler/template in a bad state and break later passes.
        """
        chat_handler, handler_name = build_qwen_vl_chat_handler(
            self._mmproj_path, self.effective_runtime
        )
        self._chat_handler = chat_handler
        self.report(
            f"Rebuilt VL chat handler ({handler_name}) and Llama after inference failure."
        )
        self.llm = self._build_llm(self._active_n_gpu_layers)

    @staticmethod
    def _is_recoverable_decode_error(exc):
        text = str(exc).lower()
        return (
            "fatal decode error" in text
            or "llama_decode failed" in text
            or "invalid input batch" in text
        )

    @staticmethod
    def _is_template_render_error(exc):
        return "nonetype" in str(exc).lower() and "render" in str(exc).lower()

    @staticmethod
    def _plain_completion_prompt(prompt):
        return (
            "You are a precise and helpful assistant.\n\n"
            f"User:\n{prompt}\n\n"
            "Assistant:\n"
        )

    def _generate_text_only(self, prompt):
        # Text-only path: avoid multimodal handler fragility when no media is present.
        if hasattr(self.llm, "reset"):
            try:
                self.llm.reset()
            except Exception:
                pass

        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )
            return _sanitize_model_text(response["choices"][0]["message"]["content"])
        except Exception as e:
            if not (self._is_template_render_error(e) or self._is_recoverable_decode_error(e)):
                raise
            self.report(
                "Text-only chat path failed; switching to plain completion fallback..."
            )
            if hasattr(self.llm, "reset"):
                try:
                    self.llm.reset()
                except Exception:
                    pass
            completion = self.llm.create_completion(
                prompt=self._plain_completion_prompt(prompt),
                max_tokens=200,
                temperature=0.1,
                stop=["<|im_end|>", "<|im_start|>"],
            )
            return _sanitize_model_text(completion["choices"][0]["text"])

    def generate(self, prompt, image=None):
        import base64
        from io import BytesIO

        if image is None:
            for attempt in range(2):
                try:
                    return self._generate_text_only(prompt)
                except Exception as e:
                    if attempt == 0 and self._inference_error_is_retryable(e):
                        self.report(
                            "Recoverable llama error detected. Reinitializing backend and retrying once..."
                        )
                        self._reinitialize_backend()
                        continue
                    raise

        # Avoid stale KV/cache state across alternating image/text turns.
        if hasattr(self.llm, "reset"):
            try:
                self.llm.reset()
            except Exception:
                pass

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{img_str}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        for attempt in range(2):
            try:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=200,
                    temperature=0.1,
                )
                raw_content = response["choices"][0]["message"]["content"]
                return _sanitize_model_text(raw_content)
            except Exception as e:
                if attempt == 0 and self._inference_error_is_retryable(e):
                    self.report(
                        "Recoverable llama error detected. Reinitializing backend and retrying once..."
                    )
                    self._reinitialize_backend()
                    continue
                raise

class FocusEngine:
    def __init__(self, model_path=None, status_callback=None, device=None, backend=None):
        def _report(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        _report(f"Initializing FocusEngine...")

        self.model_path = model_path or config_manager.get_model_path()
        configured_runtime = device or config_manager.get_device()
        self.runtime = resolve_runtime(configured_runtime, _report)
        self.backend_type = "llama-cpp"

        if not self.model_path:
            raise RuntimeError("Model path not configured.")

        try:
            llm_path, mmproj_path = find_qwen_vl_assets(self.model_path)
            if not llm_path or not mmproj_path:
                _report("Qwen 3.5 2B VL assets missing. Starting download...")
                os.makedirs(self.model_path, exist_ok=True)

                repo_id = "unsloth/Qwen3.5-2B-GGUF"
                target_files = [
                    "Qwen3.5-2B-Q4_K_M.gguf",
                    "mmproj-F16.gguf",
                ]

                for filename in target_files:
                    _report(f"Downloading {filename}...")
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=self.model_path,
                    )
                _report("Download complete.")

            self.backend = LlamaCppBackend(self.model_path, self.runtime, _report)
            config_manager.set_device(self.backend.effective_runtime.upper())

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
            description = _sanitize_model_text(description)

            print(f"Screen Description: {description}")

            # Pass 2: Logic - Verdict based on Description (TEXT ONLY)
            print("Pass 2: Evaluating alignment with task (pure text logic)...")
            verdict_prompt = (
                "Task: Decide if <ActualActivity> is the same kind of work as <IntendedTask>.\n"
                "Use a broad, normal definition: e.g. 'programming' includes writing, editing, "
                "debugging, running, or configuring software; developer terminals, IDEs, "
                "stack traces, build logs, and model/runtime setup for an app you are building "
                "still count as programming. MISMATCH only when the activity is clearly a "
                "different category (e.g. gaming, social media, video, unrelated reading).\n\n"
                f"<IntendedTask>{user_task}</IntendedTask>\n"
                f"<ActualActivity>{description}</ActualActivity>\n\n"
                "Instructions:\n"
                "1. Output exactly one line.\n"
                "2. The only allowed output is: 'Verdict: MATCH' or 'Verdict: MISMATCH'.\n"
                "3. Do not include analysis, thinking tags, or any extra text."
            )
            print(f"Pass 2 Input Prompt:\n{verdict_prompt}")
            raw_verdict = self.run_inference(verdict_prompt, image=None)
            if isinstance(raw_verdict, dict) and raw_verdict.get("status") == "ERROR":
                return raw_verdict
            raw_verdict = _sanitize_model_text(raw_verdict)
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
            print(
                f"Running {self.backend_type} inference "
                f"(runtime={self.backend.effective_runtime}, image={'PRESENT' if image else 'NONE'})..."
            )
            return self.backend.generate(prompt_text, image)
        except Exception as e:
            print(f"Inference error: {e}")
            return {"status": "ERROR", "reasoning": str(e)}

    def _parse_response(self, raw_output, user_task=None):
        """Extract status and reasoning from model output."""
        if isinstance(raw_output, dict):
            return raw_output

        raw_output = _sanitize_model_text(raw_output)
        status = "ALIGNED"  # default
        reasoning = raw_output 

        matches = list(
            re.finditer(
                r"(?:Verdict|Analysis|Answer|Final Verdict):\s*(MATCH|MISMATCH|ALIGNED|UNALIGNED)\b",
                raw_output,
                re.IGNORECASE,
            )
        )
        if matches:
            verdict = matches[-1].group(1).upper()
            if verdict in ["MATCH", "ALIGNED"]:
                status = "ALIGNED"
            else:
                status = "UNALIGNED"
            return {"status": status, "reasoning": reasoning}

        # Fallback: inspect only the tail to avoid misclassifying echoed prompt text.
        tail = raw_output[-160:].upper()
        if re.search(r"\b(MISMATCH|UNALIGNED|OFF-TASK|DISTRACT)\b", tail):
            status = "UNALIGNED"
        elif re.search(r"\b(MATCH|ALIGNED)\b", tail):
            status = "ALIGNED"
        else:
            status = "UNALIGNED"
        
        return {"status": status, "reasoning": reasoning}

if __name__ == "__main__":
    # Quick test function
    print("Testing Focus Engine Initialization...")
    engine = FocusEngine()
    print("Testing Capture and Process...")
    res = engine.capture_and_process("Writing Python code")
    print(f"Result: {res}")
