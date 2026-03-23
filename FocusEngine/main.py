import sys
import os
import subprocess
import tkinter as tk
import config_manager

def install_dependencies():
    """Automatically install dependencies from requirements.txt if running from source."""
    if getattr(sys, 'frozen', False):
        return # Never run pip in a packaged executable
        
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if os.path.exists(req_file):
        print("Checking/Installing dependencies from requirements.txt...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        except Exception as e:
            print(f"Auto-install failed: {e}")

# Run dependency check only if not frozen
if not getattr(sys, 'frozen', False):
    try:
        import dotenv
        import openvino_genai
        import huggingface_hub
        import PIL
        import numpy
    except ImportError:
        install_dependencies()

from dotenv import load_dotenv, find_dotenv

# PyInstaller creates a temp folder and stores path in _MEIPASS
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    # Use LOCALAPPDATA for logs to avoid permission issues in Program Files
    app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
    log_dir = os.path.join(app_data, 'FocusEngine')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "focus_engine.log")
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(bundle_dir, "focus_engine.log")

# When running in Pyinstaller with console=False, stdout and stderr are None.
# Regular file objects are buffered, so prints won't show up immediately.
# We use a custom Logger to force a flush after every write.
class AutoFlushLogger:
    def __init__(self, filename):
        self.terminal = None # Store existing if we want to tee, but here they are None
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.log.write(message)
        self.log.flush() # Force write to disk immediately

    def flush(self):
        self.log.flush()

if sys.stdout is None or sys.stderr is None:
    logger = AutoFlushLogger(log_file)
    if sys.stdout is None:
        sys.stdout = logger
    if sys.stderr is None:
        sys.stderr = logger
        
# Load environment variables as early as possible
env_path = os.path.join(bundle_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv(find_dotenv())


def main():
    # Local imports inside main() to prevent child processes from 
    # re-importing heavy ML libraries and potentially spawning more sub-processes
    from ui_manager import UIManager, SplashScreen
    import threading
    import queue
    
    # First Run / Model Path Check
    model_path = config_manager.get_model_path()
    if not model_path:
        print("Model path not configured. Prompting user...")
        # Local import inside main to avoid circular dependencies or early UI init
        from ui_manager import UIManager
        model_path = UIManager.prompt_for_model_dir()
        if model_path:
            config_manager.set_model_path(model_path)
            print(f"Model path saved: {model_path}")
        else:
            print("Model selection cancelled. Exiting.")
            return

    # Initialize splash screen immediately
    splash = SplashScreen()
    
    from focus_engine import FocusEngine
    from monitor_loop import MonitorLoop

    print("Initializing Application...")
    
    # Result queue for the background thread
    result_queue = queue.Queue()

    def _init_engine_thread():
        try:
            device = config_manager.get_device()
            # Pass configured model_path to FocusEngine
            engine = FocusEngine(
                model_path=model_path,
                status_callback=lambda msg: splash.status_queue.put(msg), 
                device=device
            )
            result_queue.put(("SUCCESS", engine))
        except Exception as e:
            result_queue.put(("ERROR", e))

    # Start initialization in background
    thread = threading.Thread(target=_init_engine_thread, daemon=True)
    thread.start()

    engine_container = {"engine": None}

    def _check_engine_ready():
        try:
            if not splash.root.winfo_exists() or splash.is_closing:
                return

            try:
                status, value = result_queue.get_nowait()
                if status == "SUCCESS":
                    engine_container["engine"] = value
                    splash.close()
                    return
                else:
                    print(f"Failed to initialize Focus Engine: {value}")
                    if splash.root.winfo_exists() and not splash.is_closing:
                        splash.status_label.config(text=f"Error: {value}\nClosing in 5 seconds...")
                        splash.root.after(5000, splash.close)
            except queue.Empty:
                # Poll again
                if splash.root.winfo_exists() and not splash.is_closing:
                    splash.root.after(200, _check_engine_ready)
        except (tk.TclError, AttributeError):
            # Window likely already gone
            pass

    # Start polling for the engine
    try:
        splash.root.after(200, _check_engine_ready)
    except tk.TclError:
        pass
    
    # Keep the splash screen's event loop running (this is the key for responsiveness)
    try:
        splash.root.mainloop()
    except (tk.TclError, KeyboardInterrupt):
        pass

    # Once splash.root.mainloop() returns (because splash.close() was called),
    # check if we actually got an engine.
    engine = engine_container.get("engine")
    if not engine:
        return

    print("Engine Initialized.")

    ui = UIManager(
        start_callback=lambda task: _start_monitoring(task),
        stop_callback=lambda: _stop_monitoring(),
        reset_callback=lambda: _reset_monitoring()
    )

    monitor = MonitorLoop(engine, ui.status_queue, ui.event_queue)

    def _start_monitoring(task):
        monitor.start(task)

    def _stop_monitoring():
        monitor.stop()
        
    def _reset_monitoring():
        monitor.reset_counter()

    # Start the Tkinter main loop for the main UI
    ui.run()

if __name__ == "__main__":
    import multiprocessing
    # Force 'spawn' method for consistency on Windows
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    # Crucial for PyTorch / Accelerate / Multiprocessing in a PyInstaller exe
    # Must be called before any other code that might spawn processes
    multiprocessing.freeze_support()
    
    # Optional: ensure we only run main in the original parent process
    # Children created by multiprocessing libraries should be caught by freeze_support()
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"CRITICAL CRASH:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        
        # Try to write to life-saving crash log
        try:
            with open("crash_report.txt", "w") as f:
                f.write(error_msg)
            print("Crash report written to crash_report.txt")
        except:
            pass
        
        # Keep console open if not frozen to allow user to see error
        if not getattr(sys, 'frozen', False):
            print("\nPress Enter to exit...")
            input()
