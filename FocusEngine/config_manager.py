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
    return {"device": "CPU"}

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
