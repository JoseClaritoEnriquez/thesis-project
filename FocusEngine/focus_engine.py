import os
import re
import subprocess
import threading
import inspect
import importlib
import ctypes
import sys
import hashlib

def _ensure_win32_cuda_dlls():
    """
    On Windows, ensures that CUDA runtime DLLs are in the search path.
    This is required for llama-cpp-python CUDA wheels to load correctly.
    """
    if sys.platform != "win32":
        return

    search_paths = set()
    
    # 1. Check CUDA_PATH environment variable
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        bin_path = os.path.join(cuda_path, "bin")
        if os.path.isdir(bin_path):
            search_paths.add(bin_path)

    # 2. Check standard installation directory
    base_cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(base_cuda_dir):
        try:
            for version_dir in os.listdir(base_cuda_dir):
                bin_path = os.path.join(base_cuda_dir, version_dir, "bin")
                if os.path.isdir(bin_path):
                    search_paths.add(bin_path)
        except Exception:
            pass

    # 3. Add found paths to DLL search path
    for path in search_paths:
        try:
            # Prepend to PATH for libraries that don't use add_dll_directory
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
            # Use add_dll_directory for Python 3.8+ compatibility
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(path)
        except Exception as e:
            print(f"Failed to add DLL directory {path}: {e}")
            pass

# Ensure DLLs are available before any import of llama_cpp
_ensure_win32_cuda_dlls()

import pyautogui
from PIL import Image
from huggingface_hub import hf_hub_download

import config_manager

# Qwen-VL allocates a large image token block (often >=1024 tokens). n_ctx=2048 can be tight,
# but 8192 consumes significant RAM on CPU. 4096 is a healthy middle ground.
LLAMA_VL_N_CTX = 4096
MODEL_SPEC = {
    "display_name": "Qwen2.5-VL-3B",
    "repo_id": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
    "llm_filename": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
    "mmproj_filename": "mmproj-F16.gguf",
    "llm_name_tokens": ("qwen2.5-vl-3b", "qwen2_5-vl-3b", "qwen2.5", "qwen2_5"),
}
DEFAULT_GENERATION_OPTIONS = {
    "max_tokens": 200,
    "temperature": 0.1,
}
OBSERVATION_GENERATION_OPTIONS = {
    "max_tokens": 80,
    "temperature": 0.0,
}
VERDICT_GENERATION_OPTIONS = {
    "max_tokens": 120,
    "temperature": 0.0,
    "stop": ["Thinking Process:", "<think>", "</think>"],
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


# ---------------------------------------------------------------------------
# Screenshot window management: hide the app's own windows so the VL model
# doesn't observe them and produce self-referential "ALIGNED" verdicts.
# ---------------------------------------------------------------------------

def _hide_app_windows():
    """Minimize FocusEngine windows and the console before screenshot capture.
    Returns a list of (hwnd, was_zoomed) tuples to restore later."""
    if sys.platform != "win32":
        return []

    import time
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    SW_MINIMIZE = 6
    hidden = []
    seen_hwnds = set()

    # 1. Minimize the console window (the terminal running python main.py)
    try:
        console_hwnd = kernel32.GetConsoleWindow()
        if (
            console_hwnd
            and user32.IsWindowVisible(console_hwnd)
            and not user32.IsIconic(console_hwnd)
        ):
            was_zoomed = bool(user32.IsZoomed(console_hwnd))
            user32.ShowWindow(console_hwnd, SW_MINIMIZE)
            hidden.append((console_hwnd, was_zoomed))
            seen_hwnds.add(int(console_hwnd))
    except Exception:
        pass

    # 2. Minimize all windows with "focus engine" in their title
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _enum_cb(hwnd, _lparam):
        try:
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    if "focus engine" in buf.value.lower():
                        hwnd_int = int(hwnd)
                        if hwnd_int not in seen_hwnds and not user32.IsIconic(hwnd):
                            was_zoomed = bool(user32.IsZoomed(hwnd))
                            user32.ShowWindow(hwnd, SW_MINIMIZE)
                            hidden.append((hwnd, was_zoomed))
                            seen_hwnds.add(hwnd_int)
        except Exception:
            pass
        return True

    cb = WNDENUMPROC(_enum_cb)
    user32.EnumWindows(cb, 0)

    if hidden:
        time.sleep(0.4)

    return hidden


def _restore_app_windows(handles):
    """Restore previously minimized windows."""
    if sys.platform != "win32" or not handles:
        return

    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    SW_MAXIMIZE = 3

    for hwnd, was_zoomed in handles:
        try:
            user32.ShowWindow(hwnd, SW_MAXIMIZE if was_zoomed else SW_RESTORE)
        except Exception:
            pass

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
    preferred_llm_filename = MODEL_SPEC["llm_filename"].lower()
    preferred_mmproj_filename = MODEL_SPEC["mmproj_filename"].lower()
    llm_name_tokens = tuple(token.lower() for token in MODEL_SPEC["llm_name_tokens"])
    model_candidates = [f for f in gguf_files if "mmproj" not in f.lower()]

    preferred_mmproj = next(
        (f for f in mmproj_candidates if f.lower() == preferred_mmproj_filename),
        None,
    )
    preferred_llm = next(
        (f for f in model_candidates if f.lower() == preferred_llm_filename),
        None,
    )
    token_matched_llm = next(
        (f for f in model_candidates if any(token in f.lower() for token in llm_name_tokens)),
        None,
    )

    mmproj_file = preferred_mmproj or (mmproj_candidates[0] if mmproj_candidates else None)
    llm_file = preferred_llm or token_matched_llm or (model_candidates[0] if model_candidates else None)

    mmproj_path = os.path.join(model_path, mmproj_file) if mmproj_file else None
    llm_path = os.path.join(model_path, llm_file) if llm_file else None
    return llm_path, mmproj_path


def _is_target_llm_file(llm_path):
    if not llm_path:
        return False
    name = os.path.basename(str(llm_path)).lower()
    if name == MODEL_SPEC["llm_filename"].lower():
        return True
    tokens = tuple(token.lower() for token in MODEL_SPEC["llm_name_tokens"])
    return any(token in name for token in tokens)


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

        import psutil

        if runtime == "cpu":
            # Prevent the CUDA-enabled library from chatter/looping with the driver 
            # while running on CPU. This MUST be set before llama_cpp is imported.
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            # Prevent OpenMP from fighting with llama-cpp's internal threading
            os.environ["OMP_NUM_THREADS"] = "1"

        from llama_cpp import Llama

        _silence_llama_cpp_logs()

        llm_path, mmproj_path = find_qwen_vl_assets(model_path)
        if not llm_path or not mmproj_path:
            raise RuntimeError(
                f"Missing {MODEL_SPEC['display_name']} assets. Required files: "
                f"{MODEL_SPEC['llm_filename']} and {MODEL_SPEC['mmproj_filename']}."
            )

        requested_gpu = runtime in ("cuda", "vulkan")
        n_gpu_layers = -1 if requested_gpu else 0
        chat_handler, handler_name = build_qwen_vl_chat_handler(mmproj_path, runtime)
        self._llm_path = llm_path
        self._mmproj_path = mmproj_path
        self._chat_handler = chat_handler
        self._chat_handler_name = handler_name
        self._active_n_gpu_layers = n_gpu_layers
        self._n_threads = psutil.cpu_count(logical=False) or 4
        self._llama_cpp_version = getattr(importlib.import_module("llama_cpp"), "__version__", "unknown")
        self._kv_clear_warning_emitted = False

        self.report(
            f"Loading llama.cpp backend (runtime={runtime}, n_gpu_layers={n_gpu_layers}, "
            f"n_threads={self._n_threads}, chat_handler={handler_name})..."
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
        # Temporarily disabled for debugging
        # with _suppress_llama_native_io():
        return Llama(
            model_path=self._llm_path,
            chat_handler=self._chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_ctx=LLAMA_VL_N_CTX,
            n_threads=getattr(self, "_n_threads", None),
            n_threads_batch=getattr(self, "_n_threads", None),
            verbose=False,
        )

    def _clear_kv_cache(self):
        """Aggressively clear KV cache to prevent cross-call state contamination
        between create_completion (text-only) and create_chat_completion (VL)."""
        cleared_anything = False
        try:
            if hasattr(self.llm, "_ctx") and self.llm._ctx is not None:
                self.llm._ctx.kv_cache_clear()
                cleared_anything = True
        except Exception:
            pass
        try:
            if hasattr(self.llm, "reset"):
                self.llm.reset()
                cleared_anything = True
        except Exception:
            pass
        if not cleared_anything and not self._kv_clear_warning_emitted:
            self.report(
                "Warning: KV cache clear appears unsupported on this llama-cpp build. "
                "Cross-call state contamination may still occur."
            )
            self._kv_clear_warning_emitted = True

    def _inference_error_is_retryable(self, exc):
        return self._is_recoverable_decode_error(exc) or self._is_template_render_error(
            exc
        )

    def _reinitialize_backend(self):
        """
        Recreate the VL chat handler and Llama instance. Reusing only Llama after decode
        errors can leave the multimodal handler/template in a bad state and break later passes.
        """
        import gc

        # Force cleanup of old native objects before re-allocating
        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "_chat_handler"):
            del self._chat_handler
        gc.collect()

        chat_handler, handler_name = build_qwen_vl_chat_handler(
            self._mmproj_path, self.effective_runtime
        )
        self._chat_handler = chat_handler
        self._chat_handler_name = handler_name
        self.report(
            f"Rebuilt VL chat handler ({handler_name}) and Llama after inference failure."
        )
        self.llm = self._build_llm(self._active_n_gpu_layers)
        gc.collect()

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
        # Text-only path: use raw completion with Qwen chat template to bypass
        # the VL chat handler, which requires image content to render its template.
        gen_options = self._normalize_generation_options(options)
        self._clear_kv_cache()
        completion_options = dict(gen_options)
        completion_options.setdefault("stop", ["<|im_end|>", "<|im_start|>"])
        with _suppress_llama_native_io():
            completion = self.llm.create_completion(
                prompt=self._qwen_completion_prompt(prompt),
                **completion_options,
            )
        result = _sanitize_model_text(completion["choices"][0]["text"])
        # Clear cache after text-only completion so the next VL chat call
        # doesn't hit a stale KV state from the raw completion API.
        self._clear_kv_cache()
        return result

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
                        self.report(f"Inference attempt failed: {type(e).__name__}: {e}")
                        self.report(
                            "Recoverable llama error detected. Reinitializing backend and retrying once..."
                        )
                        self._reinitialize_backend()
                        self._clear_kv_cache()
                        continue
                    raise

        self._clear_kv_cache()

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
                    self.report(f"Inference attempt failed: {type(e).__name__}: {e}")
                    self.report(
                        "Recoverable llama error detected. Reinitializing backend and retrying once..."
                    )
                    self._reinitialize_backend()
                    self._clear_kv_cache()
                    continue
                raise

class FocusEngine:
    def __init__(self, model_path=None, status_callback=None, device=None, backend=None, trace_callback=None):
        self.trace_callback = trace_callback
        self._last_screenshot_hash = None
        self._same_screenshot_streak = 0

        def _report(msg):
            print(msg)
            # self._emit_trace("INFO", "ENGINE", msg) # Redundant with StreamTee(STDOUT)

        # _report(f"Initializing FocusEngine...")
        # _report(f"Engine module: {os.path.abspath(__file__)}")

        self.model_path = model_path or config_manager.get_model_path()
        configured_runtime = device or config_manager.get_device()
        self.runtime = resolve_runtime(configured_runtime, _report)
        self.backend_type = "llama-cpp"

        if not self.model_path:
            raise RuntimeError("Model path not configured.")

        try:
            llm_path, mmproj_path = find_qwen_vl_assets(self.model_path)
            needs_target_download = (
                not llm_path
                or not mmproj_path
                or not _is_target_llm_file(llm_path)
            )
            if needs_target_download:
                if llm_path and not _is_target_llm_file(llm_path):
                    _report(
                        "Detected existing non-target VL model in selected folder "
                        f"({os.path.basename(llm_path)}). Downloading {MODEL_SPEC['display_name']}..."
                    )
                else:
                    _report(f"{MODEL_SPEC['display_name']} assets missing. Starting download...")
                os.makedirs(self.model_path, exist_ok=True)
                repo_id = MODEL_SPEC["repo_id"]
                target_files = [
                    MODEL_SPEC["llm_filename"],
                    MODEL_SPEC["mmproj_filename"],
                ]

                for filename in target_files:
                    _report(f"Downloading {filename}...")
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=self.model_path,
                    )
                _report("Download complete.")
                llm_path, mmproj_path = find_qwen_vl_assets(self.model_path)
                if not llm_path or not mmproj_path or not _is_target_llm_file(llm_path):
                    raise RuntimeError(
                        f"Failed to resolve {MODEL_SPEC['display_name']} assets after download."
                    )

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

    @staticmethod
    def _heuristic_verdict_from_text(user_task, observed_activity):
        """
        Deterministic fallback when pass-2 generation fails.
        Uses task/activity token overlap plus lightweight domain cues.
        """
        task_text = str(user_task or "").strip().lower()
        observed_text = str(observed_activity or "").strip().lower()

        if not observed_text:
            return "MISMATCH", "No observed activity was extracted from the screenshot."

        distraction_cues = {
            "youtube", "netflix", "tiktok", "instagram", "facebook",
            "x", "twitter", "game", "gaming", "reddit",
        }
        if any(cue in observed_text for cue in distraction_cues):
            return "MISMATCH", "Observed activity appears to be entertainment/social media."

        token_pattern = r"[a-z0-9_]+"
        stop_words = {
            "the", "a", "an", "to", "for", "of", "and", "or", "on", "in", "is", "are",
            "this", "that", "with", "by", "from", "as", "at", "be", "it", "user",
        }
        task_tokens = {
            t for t in re.findall(token_pattern, task_text)
            if len(t) > 2 and t not in stop_words
        }
        observed_tokens = {
            t for t in re.findall(token_pattern, observed_text)
            if len(t) > 2 and t not in stop_words
        }
        overlap = task_tokens.intersection(observed_tokens)
        if overlap:
            return "MATCH", "Observed activity shares key terms with the intended task."

        if any(k in task_text for k in ("program", "coding", "code", "debug", "python", "software")):
            dev_cues = ("ide", "editor", "terminal", "script", "debug", "build", "repo", "commit", "python", "code")
            if any(cue in observed_text for cue in dev_cues):
                return "MATCH", "Observed activity shows software development workflow."

        return "MISMATCH", "Observed activity does not clearly match the intended task."

    def _track_screenshot_fingerprint(self, screenshot):
        """
        Debug aid for suspected 'stuck frame' captures.
        We hash the resized frame and log when many consecutive captures are identical.
        """
        try:
            frame_hash = hashlib.sha1(screenshot.tobytes()).hexdigest()[:12]
        except Exception:
            return

        if frame_hash == self._last_screenshot_hash:
            self._same_screenshot_streak += 1
        else:
            self._same_screenshot_streak = 0
            self._last_screenshot_hash = frame_hash

        repeated_frames = self._same_screenshot_streak + 1
        if repeated_frames in (4, 8, 16):
            print(
                "Notice: consecutive screenshots are byte-identical "
                f"for {repeated_frames} captures (frame_hash={frame_hash})."
            )

    def capture_and_process(self, user_task):
        """
        Captures the screen, runs a two-pass MATCH/MISMATCH check,
        and returns the alignment status.
        """
        screenshot = None
        try:
            _hide_app_windows()
            screenshot = pyautogui.screenshot()

            max_side = 960
            w, h = screenshot.size
            scale = min(max_side / w, max_side / h, 1.0)
            new_size = (int(w * scale), int(h * scale))
            screenshot = screenshot.resize(new_size, Image.LANCZOS)
            self._track_screenshot_fingerprint(screenshot)

            # Pass 1: extract observed activity directly from screenshot.
            print("Pass 1: Identifying current activity...")
            observation_prompt = (
                "Context: You are a desktop activity monitor.\n"
                "Task: Describe the primary activity in this screenshot. IMPORTANT: If you see a terminal, command prompt, or UI showing the 'Intended Goal', ignore it and describe the other visible windows. Do not simply read the Intended Goal text from the screen.\n\n"
                f"Intended Goal: {user_task}\n\n"
                "Format:\n"
                "ObservedActivity: [One concise sentence describing the foreground app and action]"
            )
            # print(f"Pass 1 Input Prompt:\n{observation_prompt}")
            self._emit_trace("LLM_INPUT", "ENGINE", observation_prompt)
            raw_observation = self.run_inference(
                observation_prompt,
                image=screenshot,
                options=OBSERVATION_GENERATION_OPTIONS,
            )
            if isinstance(raw_observation, dict) and raw_observation.get("status") == "ERROR":
                return raw_observation
            raw_observation = _sanitize_model_text(raw_observation)
            raw_observation_str = (
                raw_observation if isinstance(raw_observation, str) else str(raw_observation)
            )
            # print(f"Raw Observation: {raw_observation_str!r} (len={len(raw_observation_str)})")
            self._emit_trace("LLM_OUTPUT", "ENGINE", raw_observation_str)

            observed_activity = raw_observation_str
            observed_match = re.search(
                r"ObservedActivity:\s*(.+)",
                raw_observation_str,
                re.IGNORECASE,
            )
            if observed_match:
                observed_activity = observed_match.group(1).strip()
            observed_activity = observed_activity.splitlines()[0].strip() if observed_activity else ""

            # Pass 2: strict text-only verdict against intended task.
            print("Pass 2: Checking alignment with task...")
            verdict_prompt = (
                "Compare the ObservedActivity to the IntendedTask. Be strict.\n\n"
                "Example:\n"
                "IntendedTask: writing code\n"
                "ObservedActivity: User is watching a full-screen YouTube video of a cat.\n"
                "Reasoning: YouTube is entertainment, not a coding environment.\n"
                "Verdict: MISMATCH\n\n"
                "Data to Evaluate:\n"
                f"IntendedTask: {user_task}\n"
                f"ObservedActivity: {observed_activity}\n\n"
                "Final Output:\n"
                f"ObservedActivity: {observed_activity}\n"
                "Reasoning: \n"
                "After your reasoning, add a new line as exactly: Verdict: MATCH or Verdict: MISMATCH"
            )
            # print(f"Pass 2 Input Prompt:\n{verdict_prompt}")
            self._emit_trace("LLM_INPUT", "ENGINE", verdict_prompt)
            raw_verdict = self.run_inference(
                verdict_prompt,
                image=None,
                options=VERDICT_GENERATION_OPTIONS,
            )
            if isinstance(raw_verdict, dict) and raw_verdict.get("status") == "ERROR":
                return raw_verdict
            raw_verdict = _sanitize_model_text(raw_verdict)
            raw_verdict_str = raw_verdict if isinstance(raw_verdict, str) else str(raw_verdict)
            if not raw_verdict_str.strip():
                retry_prompt = (
                    "Output one label only.\n"
                    "Allowed labels: MATCH or MISMATCH.\n\n"
                    f"IntendedTask: {user_task}\n"
                    f"ObservedActivity: {observed_activity}\n"
                    "Verdict: "
                )
                print("Pass 2 Retry: Empty verdict output; requesting verdict-only label...")
                self._emit_trace("LLM_INPUT", "ENGINE", retry_prompt)
                retry_verdict = self.run_inference(
                    retry_prompt,
                    image=None,
                    options={"max_tokens": 6, "temperature": 0.0, "stop": ["\n", ".", ","]},
                )
                if not (isinstance(retry_verdict, dict) and retry_verdict.get("status") == "ERROR"):
                    retry_text = _sanitize_model_text(retry_verdict)
                    retry_label_match = re.search(
                        r"\b(MATCH|MISMATCH|ALIGNED|UNALIGNED)\b",
                        str(retry_text),
                        re.IGNORECASE,
                    )
                    if retry_label_match:
                        retry_label = retry_label_match.group(1).upper()
                        normalized = "MATCH" if retry_label in ("MATCH", "ALIGNED") else "MISMATCH"
                        raw_verdict_str = (
                            f"ObservedActivity: {observed_activity}\n"
                            "Reasoning: Verdict recovered via strict label retry.\n"
                            f"Verdict: {normalized}"
                        )
            if not raw_verdict_str.strip():
                heuristic_verdict, heuristic_reason = self._heuristic_verdict_from_text(
                    user_task,
                    observed_activity,
                )
                raw_verdict_str = (
                    f"ObservedActivity: {observed_activity}\n"
                    f"Reasoning: {heuristic_reason}\n"
                    f"Verdict: {heuristic_verdict}"
                )
                print("Pass 2 Fallback: Using deterministic heuristic verdict.")
            # print(f"Raw Verdict: {raw_verdict_str!r} (len={len(raw_verdict_str)})")
            self._emit_trace("LLM_OUTPUT", "ENGINE", raw_verdict_str)

            if "ObservedActivity:" not in raw_verdict_str and observed_activity:
                raw_verdict_str = f"ObservedActivity: {observed_activity}\n{raw_verdict_str}"

            return self._parse_response(raw_verdict_str, user_task)
        except Exception as e:
            print(f"Error in one-pass processing: {e}")
            self._emit_trace("ERROR", "ENGINE", f"Error in one-pass processing: {e}")
            return {"status": "ERROR", "reasoning": str(e)}
        finally:
            # Clean up frame buffer even when any pass fails.
            if screenshot is not None:
                try:
                    del screenshot
                except Exception:
                    pass
            import gc
            gc.collect()

    def run_inference(self, prompt_text, image=None, options=None):
        try:
            options_tag = ""
            if options:
                options_tag = (
                    f", temperature={options.get('temperature')}, "
                    f"max_tokens={options.get('max_tokens')}"
                )
            # print(
            #     f"Running {self.backend_type} inference "
            #     f"(runtime={self.backend.effective_runtime}, image={'PRESENT' if image else 'NONE'}"
            #     f"{options_tag})..."
            # )
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
