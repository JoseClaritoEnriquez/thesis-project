import sys
import os

from dotenv import load_dotenv, find_dotenv

# PyInstaller creates a temp folder and stores path in _MEIPASS
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    log_file = os.path.join(os.path.dirname(sys.executable), "focus_engine.log")
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = "focus_engine.log"

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
    
    # Initialize splash screen immediately
    splash = SplashScreen()
    
    from focus_engine import FocusEngine
    from monitor_loop import MonitorLoop

    print("Initializing Application...")
    
    try:
        engine = FocusEngine()
    except Exception as e:
        print(f"Failed to initialize Focus Engine: {e}")
        splash.close()
        return

    print("Engine Initialized.")
    splash.close()

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

    # Start the Tkinter main loop on the primary thread
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
    main()
