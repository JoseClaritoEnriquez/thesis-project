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
        "device": "CPU", 
        "model_path": None,
        "backend": "auto" # "auto", "openvino", or "llama-cpp" (replaces transformers)
    }

def save_config(config):
    path = get_config_path()
    try:
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

def get_device():
    return load_config().get("device", "CPU")

def set_device(device):
    config = load_config()
    config["device"] = device
    save_config(config)

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

def get_backend():
    return load_config().get("backend", "auto")

def set_backend(backend):
    config = load_config()
    config["backend"] = backend
    save_config(config)
