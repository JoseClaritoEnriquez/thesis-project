import os
import re
import subprocess
import threading
import inspect
import importlib
import ctypes

import pyautogui
from PIL import Image
from huggingface_hub import hf_hub_download

import config_manager

# Qwen-VL allocates a large image token block (often >=1024 tokens). n_ctx=2048 cannot
# fit image + prompt + generation and triggers decode failures and poisoned handler state.
LLAMA_VL_N_CTX = 8192
DEFAULT_GENERATION_OPTIONS = {
    "max_tokens": 200,
    "temperature": 0.1,
}
VERDICT_GENERATION_OPTIONS = {
    "max_tokens": 220,
    "temperature": 0.0,
}

_llama_noop_log_callback = None


def _silence_llama_cpp_logs():
    """Drop llama.cpp ggml_log lines (e.g. llama_context, load_hparams) that still print with verbose=False."""
    global _llama_noop_log_callback
    if _llama_noop_log_callback is not None:
        return
    import llama_cpp.llama_cpp as _lc

    @_lc.llama_log_callback
    def _noop(_level, _text, _user_data):
        pass

    _llama_noop_log_callback = _noop
    _lc.llama_log_set(_noop, ctypes.c_void_p(0))


def _suppress_llama_native_io():
    """Context manager: silence OS-level stderr/stdout from llama.cpp during native calls."""
    from llama_cpp._utils import suppress_stdout_stderr

    return suppress_stdout_stderr(disable=False)


def _sanitize_model_text(text):
    """Remove internal reasoning tags and normalize whitespace."""
    if text is None:
        return ""
    cleaned = str(text)
    # Preserve the content inside <think> blocks (strip tags only).
    # Some models wrap the whole response in <think>, and we still want to keep
    # structured fields like `ObservedActivity:` / `Reasoning:` for parsing.
    cleaned = re.sub(
        r"<think>\s*(.*?)\s*</think>",
        lambda m: m.group(1),
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Some runtimes emit an opening <think> tag without closing it; drop only the tag.
    cleaned = re.sub(r"<think>", "", cleaned, flags=re.IGNORECASE)
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

        _silence_llama_cpp_logs()

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
        self._chat_handler_name = handler_name
        self._active_n_gpu_layers = n_gpu_layers
        self._llama_cpp_version = getattr(importlib.import_module("llama_cpp"), "__version__", "unknown")

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
        with _suppress_llama_native_io():
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
        self._chat_handler_name = handler_name
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
    def _qwen_completion_prompt(prompt):
        return (
            "<|im_start|>system\n"
            "You are a precise and helpful assistant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _normalize_generation_options(self, options):
        merged = dict(DEFAULT_GENERATION_OPTIONS)
        if options:
            merged.update(options)
        return merged

    def _generate_text_only(self, prompt, options=None):
        # Text-only path: avoid multimodal handler fragility when no media is present.
        gen_options = self._normalize_generation_options(options)
        if hasattr(self.llm, "reset"):
            try:
                self.llm.reset()
            except Exception:
                pass

        try:
            with _suppress_llama_native_io():
                response = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    **gen_options,
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
            completion_options = dict(gen_options)
            completion_options.setdefault("stop", ["<|im_end|>", "<|im_start|>"])
            with _suppress_llama_native_io():
                completion = self.llm.create_completion(
                    prompt=self._qwen_completion_prompt(prompt),
                    **completion_options,
                )
            return _sanitize_model_text(completion["choices"][0]["text"])

    def get_stack_fingerprint(self):
        return {
            "runtime": self.effective_runtime,
            "llama_cpp_version": self._llama_cpp_version,
            "chat_handler": self._chat_handler_name,
            "n_ctx": LLAMA_VL_N_CTX,
            "model_file": os.path.basename(self._llm_path),
            "mmproj_file": os.path.basename(self._mmproj_path),
        }

    def generate(self, prompt, image=None, options=None):
        import base64
        from io import BytesIO

        gen_options = self._normalize_generation_options(options)
        if image is None:
            for attempt in range(2):
                try:
                    return self._generate_text_only(prompt, options=gen_options)
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
                with _suppress_llama_native_io():
                    response = self.llm.create_chat_completion(
                        messages=messages,
                        **gen_options,
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
    def __init__(self, model_path=None, status_callback=None, device=None, backend=None, trace_callback=None):
        self.trace_callback = trace_callback

        def _report(msg):
            print(msg)
            self._emit_trace("INFO", "ENGINE", msg)
            if status_callback:
                status_callback(msg)

        _report(f"Initializing FocusEngine...")
        _report(f"Engine module: {os.path.abspath(__file__)}")

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
            stack = self.backend.get_stack_fingerprint()
            _report(
                "Stack fingerprint: "
                f"runtime={stack['runtime']}, "
                f"llama_cpp={stack['llama_cpp_version']}, "
                f"chat_handler={stack['chat_handler']}, "
                f"n_ctx={stack['n_ctx']}, "
                f"model={stack['model_file']}, "
                f"mmproj={stack['mmproj_file']}"
            )

            _report("Focus Engine Initialized successfully.")
        except Exception as e:
            _report(f"Error: Failed to load or download model: {e}")
            self._emit_trace("ERROR", "ENGINE", f"Failed to initialize model backend: {e}")
            raise RuntimeError(f"Failed to initialize Focus Engine: {e}") from e

    def _emit_trace(self, level, source, message):
        if self.trace_callback:
            self.trace_callback(message, level=level, source=source)

    def capture_and_process(self, user_task):
        """
        Captures the screen, runs a single-pass MATCH/MISMATCH check,
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
            # One-pass multimodal verdict directly from screenshot + intended task.
            print("One Pass: Evaluating screenshot alignment with intended task...")
            verdict_prompt = (
                "Task: Decide if the current screenshot activity is the same work category as <IntendedTask>.\n"
                "Rule: Use broad category matching. Example: for <IntendedTask>programming</IntendedTask>, "
                "coding, debugging, terminal runs, build logs, and runtime setup for the app all count as MATCH. "
                "Use MISMATCH only when activity is clearly a different category with no immediate contribution.\n\n"
                "Example Output:\n"
                "ObservedActivity: Developer debugging Python errors in terminal and IDE.\n"
                "Reasoning: This is directly part of programming work and supports completion of the intended task.\n"
                "Verdict: MATCH\n\n"
                f"<IntendedTask>{user_task}</IntendedTask>\n\n"
                "Instructions:\n"
                "1. Analyze the screenshot and compare by category fit.\n"
                "2. Output exactly three lines in this exact schema:\n"
                "ObservedActivity: <what you see on screen>\n"
                "Reasoning: <short reason supporting your judgement>\n"
                "Verdict: MATCH or Verdict: MISMATCH\n"
                "3. Do not include thinking tags or extra sections."
            )
            print(f"One Pass Input Prompt:\n{verdict_prompt}")
            self._emit_trace("LLM_INPUT", "ENGINE", verdict_prompt)
            raw_verdict = self.run_inference(
                verdict_prompt,
                image=screenshot,
                options=VERDICT_GENERATION_OPTIONS,
            )
            if isinstance(raw_verdict, dict) and raw_verdict.get("status") == "ERROR":
                return raw_verdict
            raw_verdict = _sanitize_model_text(raw_verdict)
            raw_verdict_str = raw_verdict if isinstance(raw_verdict, str) else str(raw_verdict)
            print(f"Raw Verdict: {raw_verdict_str!r} (len={len(raw_verdict_str)})")
            self._emit_trace("LLM_OUTPUT", "ENGINE", raw_verdict_str)

            # 3. Clean up RAM
            import gc
            del screenshot
            gc.collect()

            return self._parse_response(raw_verdict, user_task)
        except Exception as e:
            print(f"Error in one-pass processing: {e}")
            self._emit_trace("ERROR", "ENGINE", f"Error in one-pass processing: {e}")
            return {"status": "ERROR", "reasoning": str(e)}

    def run_inference(self, prompt_text, image=None, options=None):
        try:
            options_tag = ""
            if options:
                options_tag = (
                    f", temperature={options.get('temperature')}, "
                    f"max_tokens={options.get('max_tokens')}"
                )
            print(
                f"Running {self.backend_type} inference "
                f"(runtime={self.backend.effective_runtime}, image={'PRESENT' if image else 'NONE'}"
                f"{options_tag})..."
            )
            return self.backend.generate(prompt_text, image, options=options)
        except Exception as e:
            print(f"Inference error: {e}")
            self._emit_trace("ERROR", "ENGINE", f"Inference error: {e}")
            return {"status": "ERROR", "reasoning": str(e)}

    def _parse_response(self, raw_output, user_task=None):
        """Extract status, observed activity, and reasoning from model output."""
        if isinstance(raw_output, dict):
            return raw_output

        raw_output = _sanitize_model_text(raw_output)
        status = "ALIGNED"  # default
        reasoning = raw_output
        observed_activity = ""

        observed_matches = list(
            re.finditer(
                r"ObservedActivity:\s*(.+)",
                raw_output,
                re.IGNORECASE,
            )
        )
        if observed_matches:
            observed_activity = observed_matches[-1].group(1).strip()

        reasoning_matches = list(
            re.finditer(
                r"Reasoning:\s*(.+)",
                raw_output,
                re.IGNORECASE,
            )
        )
        if reasoning_matches:
            reasoning = reasoning_matches[-1].group(1).strip()

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
            return {
                "status": status,
                "reasoning": reasoning,
                "observed_activity": observed_activity,
            }

        # Fallback: inspect only the tail to avoid misclassifying echoed prompt text.
        tail = raw_output[-160:].upper()
        if re.search(r"\b(MISMATCH|UNALIGNED|OFF-TASK|DISTRACT)\b", tail):
            status = "UNALIGNED"
        elif re.search(r"\b(MATCH|ALIGNED)\b", tail):
            status = "ALIGNED"
        else:
            status = "UNALIGNED"
        
        return {
            "status": status,
            "reasoning": reasoning,
            "observed_activity": observed_activity,
        }

if __name__ == "__main__":
    # Quick test function
    print("Testing Focus Engine Initialization...")
    engine = FocusEngine()
    print("Testing Capture and Process...")
    res = engine.capture_and_process("Writing Python code")
    print(f"Result: {res}")
