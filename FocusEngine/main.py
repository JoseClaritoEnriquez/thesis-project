import threading
from focus_engine import FocusEngine
from monitor_loop import MonitorLoop
from ui_manager import UIManager

def main():
    print("Initializing Application...")
    
    try:
        engine = FocusEngine()
    except Exception as e:
        print(f"Failed to initialize Focus Engine: {e}")
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

    # Start the Tkinter main loop on the primary thread
    ui.run()

if __name__ == "__main__":
    main()
