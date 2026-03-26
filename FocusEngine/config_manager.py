import os
import json
import sys

def get_app_data_dir():
    if getattr(sys, 'frozen', False):
        app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        dir_path = os.path.join(app_data, 'FocusEngine')
    else:
        # In dev, store in the project directory
        dir_path = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_config_path():
    return os.path.join(get_app_data_dir(), "config.json")

def load_config():
    path = get_config_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # Default config
    return {
        "device": "AUTO",
        "runtime": "auto",
        "model_path": None,
        "backend": "llama-cpp",
    }

def save_config(config):
    path = get_config_path()
    try:
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

def get_device():
    config = load_config()
    runtime = str(config.get("runtime", "")).strip().lower()
    if runtime == "cuda":
        return "CUDA"
    if runtime == "vulkan":
        return "VULKAN"
    if runtime == "cpu":
        return "CPU"

    legacy = str(config.get("device", "AUTO")).strip().upper()
    if legacy in {"CUDA", "VULKAN", "CPU", "AUTO"}:
        return legacy
    if legacy in {"NVIDIA", "GPU"}:
        return "AUTO"
    if legacy == "AMD":
        return "VULKAN"
    return "AUTO"

def set_device(device):
    config = load_config()
    normalized = str(device or "AUTO").strip().upper()
    if normalized in {"NVIDIA", "GPU"}:
        normalized = "AUTO"
    if normalized == "AMD":
        normalized = "VULKAN"
    if normalized not in {"AUTO", "CUDA", "VULKAN", "CPU"}:
        normalized = "AUTO"

    config["device"] = normalized
    config["runtime"] = normalized.lower()
    save_config(config)


def get_runtime():
    return get_device().lower()


def set_runtime(runtime):
    set_device(runtime)

def get_model_path():
    path = load_config().get("model_path")
    if path and not os.path.exists(path):
        print(f"Configured model path does not exist: {path}")
        return None
    return path

def set_model_path(path):
    config = load_config()
    config["model_path"] = path
    save_config(config)

def get_one_minute_mode():
    return load_config().get("one_minute_mode", False)

def set_one_minute_mode(enabled):
    config = load_config()
    config["one_minute_mode"] = bool(enabled)
    save_config(config)

def get_backend():
    # Backward-compatible getter. Backend is fixed to llama-cpp.
    return "llama-cpp"

def set_backend(backend):
    config = load_config()
    # Keep key for backward compatibility, but enforce llama-cpp-only architecture.
    config["backend"] = "llama-cpp"
    save_config(config)
