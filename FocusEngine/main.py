import sys
import os
import subprocess
import tkinter as tk
import queue
from datetime import datetime
import config_manager

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
        except Exception:
            pass

def install_dependencies():
    """Automatically install dependencies from requirements.txt if running from source."""
    if getattr(sys, 'frozen', False):
        return # Never run pip in a packaged executable

    script_dir = os.path.dirname(os.path.abspath(__file__))
    installer = os.path.join(script_dir, "install_runtime.py")
    
    if not os.path.exists(installer):
        return

    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Hardware Setup")
    root.geometry("480x320")
    root.resizable(False, False)
    root.eval('tk::PlaceWindow . center')
    root.configure(bg="#fdf6e3")

    tk.Label(
        root, text="Select your hardware for AI Inference:",
        font=("Helvetica", 14, "bold"), bg="#fdf6e3", fg="#b58900"
    ).pack(pady=(25, 5))

    tk.Label(
        root, text="This determines which version of the inference engine is installed.",
        font=("Helvetica", 10), bg="#fdf6e3", fg="#657b83"
    ).pack(pady=(0, 25))

    selected_profile = tk.StringVar(value="jamepeng-cpu")

    options = [
        ("CPU (Slow but highly compatible)", "jamepeng-cpu"),
        ("NVIDIA GPU (CUDA - Fast)", "jamepeng-nvidia"),
        ("AMD GPU (Vulkan - Fast)", "jamepeng-amd")
    ]

    for text, value in options:
        rb = tk.Radiobutton(
            root, text=text, variable=selected_profile, value=value,
            bg="#fdf6e3", fg="#2aa198", font=("Helvetica", 11, "bold"),
            activebackground="#fdf6e3", activeforeground="#2aa198", selectcolor="#eee8d5"
        )
        rb.pack(anchor="w", padx=80, pady=5)

    def on_submit():
        root.quit()

    tk.Button(
        root, text="Install Dependencies", command=on_submit,
        bg="#2aa198", fg="white", font=("Helvetica", 11, "bold"), padx=20
    ).pack(pady=30)

    root.mainloop()
    profile = selected_profile.get()
    root.destroy()

    print(f"Checking/Installing dependencies (profile={profile})...")
    try:
        subprocess.check_call([sys.executable, installer, "--profile", profile])
    except Exception as e:
        print(f"Auto-install failed: {e}")

# Run dependency check only if not frozen
if not getattr(sys, 'frozen', False):
    _ensure_win32_cuda_dlls()
    try:
        import dotenv
        import llama_cpp
        import huggingface_hub
        import PIL
        import numpy
    except (ImportError, RuntimeError):
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


class StreamTee:
    """Duplicate stream output to a sink callback while preserving original behavior."""

    def __init__(self, stream, sink=None, stream_name="stdout"):
        self.stream = stream
        self.sink = sink
        self.stream_name = stream_name

    def write(self, message):
        if self.stream:
            self.stream.write(message)
        if self.sink:
            self.sink(message, source=self.stream_name)

    def flush(self):
        if self.stream:
            self.stream.flush()


def _build_log_sink(target_queue):
    """Return a function that writes structured log events into a queue."""

    def _sink(message, level="INFO", source="SYSTEM"):
        text = str(message or "")
        if not text.strip():
            return
        target_queue.put(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "source": source,
                "message": text.rstrip("\n"),
            }
        )

    return _sink

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
    
    app_log_queue = queue.Queue()
    emit_log = _build_log_sink(app_log_queue)

    # Tee stdout/stderr to the UI terminal stream while preserving current behavior.
    if not isinstance(sys.stdout, StreamTee):
        sys.stdout = StreamTee(sys.stdout, sink=emit_log, stream_name="STDOUT")
    if not isinstance(sys.stderr, StreamTee):
        sys.stderr = StreamTee(sys.stderr, sink=lambda msg, source="STDERR": emit_log(msg, level="ERROR", source=source), stream_name="STDERR")

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
                trace_callback=emit_log,
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

    def _handle_settings_changed():
        """Called when device is changed in settings."""
        nonlocal engine
        print("Settings changed. Reloading engine...")
        
        # Stop monitoring if active
        monitor.stop()
        
        # Re-initialize engine with splash screen
        new_splash = SplashScreen(parent=ui.root)
        
        new_device = config_manager.get_device()
        
        def _reinit_thread():
            try:
                new_engine = FocusEngine(
                    model_path=model_path,
                    status_callback=lambda msg: new_splash.status_queue.put(msg),
                    trace_callback=emit_log,
                    device=new_device
                )
                result_queue.put(("SUCCESS", new_engine))
            except Exception as e:
                result_queue.put(("ERROR", e))

        threading.Thread(target=_reinit_thread, daemon=True).start()

        def _check_reinit_ready():
            try:
                status, value = result_queue.get_nowait()
                if status == "SUCCESS":
                    nonlocal engine
                    engine = value
                    monitor.engine = engine # Update monitor with new engine
                    monitor.device = new_device # Update monitor with new device string
                    new_splash.close()
                    ui._build_setup_view() # Return to main view
                else:
                    print(f"Failed to reload engine: {value}")
                    new_splash.status_label.config(text=f"Error: {value}")
                    ui.root.after(3000, new_splash.close)
            except queue.Empty:
                ui.root.after(200, _check_reinit_ready)

        ui.root.after(200, _check_reinit_ready)

    ui = UIManager(
        start_callback=lambda task: _start_monitoring(task),
        stop_callback=lambda: _stop_monitoring(),
        reset_callback=lambda: _reset_monitoring(),
        settings_changed_callback=_handle_settings_changed,
        log_queue=app_log_queue,
    )

    monitor = MonitorLoop(
        engine,
        ui.status_queue,
        ui.event_queue,
        device=config_manager.get_device(),
        log_callback=emit_log,
    )

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
