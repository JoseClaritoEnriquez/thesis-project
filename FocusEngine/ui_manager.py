import tkinter as tk
from tkinter import ttk
import queue
import threading
import sys
import pystray
from PIL import Image, ImageDraw

def create_tray_icon(ui_manager):
    """
    Creates a simple system tray icon using pystray.
    ui_manager must provide a method to show the window.
    """
    # Create a 64x64 simple colored square for the tray icon
    image = Image.new('RGB', (64, 64), color=(0, 128, 255))
    dc = ImageDraw.Draw(image)
    dc.rectangle((16, 16, 48, 48), fill=(255, 255, 255))
    
    def on_show_window(icon, item):
        icon.stop()
        ui_manager.restore_window()

    def on_exit(icon, item):
        icon.stop()
        ui_manager.exit_app()

    menu = pystray.Menu(
        pystray.MenuItem('Show', on_show_window),
        pystray.MenuItem('Exit', on_exit)
    )

    return pystray.Icon("FocusEngine", image, "Focus Engine", menu)

class UIManager:
    def __init__(self, start_callback, stop_callback, reset_callback):
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.reset_callback = reset_callback
        
        self.status_queue = queue.Queue()
        self.event_queue = queue.Queue()
        
        self.root = tk.Tk()
        self.root.title("Focus Engine (High Performance)")
        self.root.geometry("450x350")
        self.root.resizable(False, False)
        
        # Simple Solarized-esque styling for baseline
        self.bg_color = "#fdf6e3"
        self.fg_color = "#657b83"
        self.root.configure(bg=self.bg_color)
        
        self.tray_icon = None
        self.frame = None
        self.timer_id = None
        self.remaining_seconds = 0
        self.session_task = ""
        self.unaligned_count = 0
        
        self._build_setup_view()
        
        # Periodic UI update to check queues without blocking Tk mainloop
        self.root.after(100, self._process_queues)
        
        # Handle window deletion (closing redirects to tray)
        self.root.protocol("WM_DELETE_WINDOW", self.hide_to_tray)

    def _clear_frame(self):
        """Destroys the current frame to prepare for a new view."""
        if self.frame:
            self.frame.destroy()
            self.frame = None

    def _build_setup_view(self):
        """Builds the initial configuration screen"""
        self._clear_frame()
        self.frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        self.frame.pack(expand=True, fill="both")

        tk.Label(
            self.frame, text="Focus Engine", font=("Helvetica", 18, "bold"), 
            bg=self.bg_color, fg="#b58900"
        ).pack(pady=(0, 20))

        tk.Label(
            self.frame, text="Current Task:", bg=self.bg_color, fg=self.fg_color
        ).pack(anchor="w")

        self.task_entry = tk.Entry(self.frame, width=50, font=("Helvetica", 12))
        self.task_entry.pack(pady=5)
        self.task_entry.insert(0, "Refactoring the database module")

        tk.Label(
            self.frame, text="Session Length (Minutes):", bg=self.bg_color, fg=self.fg_color
        ).pack(anchor="w", pady=(10, 0))

        self.time_entry = tk.Entry(self.frame, width=20, font=("Helvetica", 12))
        self.time_entry.pack(pady=5, anchor="w")
        self.time_entry.insert(0, "60")

        self.start_button = tk.Button(
            self.frame, text="Start Focus Session", bg="#2aa198", fg="black", 
            font=("Helvetica", 12, "bold"), command=self._on_start
        )
        self.start_button.pack(pady=20, fill="x")

        self.status_label = tk.Label(
            self.frame, text="Status: IDLE", bg=self.bg_color, fg="#859900"
        )
        self.status_label.pack(side="bottom")

    def _on_start(self):
        task = self.task_entry.get().strip()
        if not task:
            return
        
        try:
            minutes = int(self.time_entry.get().strip())
        except ValueError:
            minutes = 60

        self.session_task = task
        self.remaining_seconds = minutes * 60
        self.unaligned_count = 0
        
        # Start AI worker
        self.start_callback(task)
        
        # Transition to running session view (stay visible, don't hide to tray)
        self._build_running_view()

    def _build_running_view(self):
        """Builds the active session screen with timer and stop button."""
        self._clear_frame()
        self.frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        self.frame.pack(expand=True, fill="both")

        tk.Label(
            self.frame, text="Session Active", font=("Helvetica", 18, "bold"),
            bg=self.bg_color, fg="#859900"
        ).pack(pady=(0, 10))

        # Task display
        tk.Label(
            self.frame, text="Task:", bg=self.bg_color, fg=self.fg_color,
            font=("Helvetica", 10)
        ).pack(anchor="w")
        tk.Label(
            self.frame, text=self.session_task, bg=self.bg_color, fg="#268bd2",
            font=("Helvetica", 12, "bold"), wraplength=400, justify="left"
        ).pack(anchor="w", pady=(0, 15))

        # Timer display
        self.timer_label = tk.Label(
            self.frame, text=self._format_time(self.remaining_seconds),
            font=("Helvetica", 48, "bold"), bg=self.bg_color, fg="#b58900"
        )
        self.timer_label.pack(pady=(0, 10))

        # Status info row
        info_frame = tk.Frame(self.frame, bg=self.bg_color)
        info_frame.pack(fill="x", pady=(0, 10))

        self.status_label = tk.Label(
            info_frame, text="Status: Monitoring...", bg=self.bg_color, fg="#859900",
            font=("Helvetica", 10)
        )
        self.status_label.pack(side="left")

        self.counter_label = tk.Label(
            info_frame, text="Distractions: 0", bg=self.bg_color, fg=self.fg_color,
            font=("Helvetica", 10)
        )
        self.counter_label.pack(side="right")

        # Stop button
        self.stop_button = tk.Button(
            self.frame, text="Stop Session", bg="#dc322f", fg="white",
            font=("Helvetica", 12, "bold"), command=self._on_stop
        )
        self.stop_button.pack(pady=10, fill="x")

        # Start the countdown timer
        self._tick_timer()

    def _format_time(self, seconds):
        """Formats seconds into MM:SS string."""
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"

    def _tick_timer(self):
        """Called every second to update the countdown timer."""
        if self.remaining_seconds <= 0:
            # Session complete
            self._on_stop()
            return

        self.remaining_seconds -= 1
        if hasattr(self, 'timer_label') and self.timer_label.winfo_exists():
            self.timer_label.config(text=self._format_time(self.remaining_seconds))

        # Schedule next tick
        self.timer_id = self.root.after(1000, self._tick_timer)

    def _on_stop(self):
        """Stops the session and shows the results view."""
        # Cancel the timer
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None

        # Stop the AI monitoring
        self.stop_callback()

        # Show results
        self._build_results_view()

    def _build_results_view(self):
        """Shows a summary after the session ends."""
        self._clear_frame()
        self.frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        self.frame.pack(expand=True, fill="both")

        tk.Label(
            self.frame, text="Session Complete", font=("Helvetica", 18, "bold"),
            bg=self.bg_color, fg="#b58900"
        ).pack(pady=(0, 20))

        tk.Label(
            self.frame, text=f"Task: {self.session_task}",
            font=("Helvetica", 12), bg=self.bg_color, fg=self.fg_color,
            wraplength=400, justify="left"
        ).pack(anchor="w", pady=5)

        tk.Label(
            self.frame, text=f"Distractions detected: {self.unaligned_count}",
            font=("Helvetica", 12), bg=self.bg_color, fg=self.fg_color
        ).pack(anchor="w", pady=5)

        tk.Label(
            self.frame, text="Great work! Keep up the focus. 💪",
            font=("Helvetica", 14), bg=self.bg_color, fg="#859900"
        ).pack(pady=20)

        tk.Button(
            self.frame, text="New Session", bg="#2aa198", fg="black",
            font=("Helvetica", 12, "bold"), command=self._build_setup_view
        ).pack(pady=10, fill="x")

    def hide_to_tray(self):
        self.root.withdraw() # Hide window
        self.tray_icon = create_tray_icon(self)
        
        # Run tray in separate thread otherwise it blocks Tk
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def restore_window(self):
        self.root.deiconify()
        if self.tray_icon:
            self.tray_icon.stop()
            self.tray_icon = None

    def exit_app(self):
        self.stop_callback()
        if self.tray_icon:
            self.tray_icon.stop()
        self.root.quit()
        sys.exit(0)

    def trigger_intervention(self, task, reasoning):
        """Displays the 'Nudge' full-screen topmost modal."""
        # Solarized Dark / High contrast Red
        modal = tk.Toplevel(self.root)
        modal.attributes("-topmost", True)
        modal.attributes("-fullscreen", True)
        modal.configure(bg="#dc322f") # Red background
        
        modal.focus_force()

        frame = tk.Frame(modal, bg="#dc322f")
        frame.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            frame, text=f"Are you still focusing on:\n'{task}'?",
            font=("Helvetica", 32, "bold"), fg="#fdf6e3", bg="#dc322f", justify="center"
        ).pack(pady=20)
        
        tk.Label(
            frame, text=f"Reason: {reasoning}",
            font=("Helvetica", 18), fg="#eee8d5", bg="#dc322f", justify="center", wraplength=800
        ).pack(pady=10)

        def on_return():
            self.reset_callback()
            modal.destroy()

        btn = tk.Button(
            frame, text="✅ I'm back on track!", font=("Helvetica", 24, "bold"),
            bg="#859900", fg="black", command=on_return, padx=20, pady=10
        )
        btn.pack(pady=40)

    def _process_queues(self):
        """Reads from queues to update UI based on AI worker thread state."""
        try:
            while True:
                event = self.event_queue.get_nowait()
                event_type = event.get("type")
                
                if event_type == "STATE":
                    if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                        self.status_label.config(text=f"Status: {event.get('value')}")
                elif event_type == "INTERVENTION":
                    self.trigger_intervention(event.get("task"), event.get("reason"))
        except queue.Empty:
            pass

        # Also check status_queue for counter updates
        try:
            while True:
                status = self.status_queue.get_nowait()
                self.unaligned_count = status.get("counter", self.unaligned_count)
                if hasattr(self, 'counter_label') and self.counter_label.winfo_exists():
                    self.counter_label.config(text=f"Distractions: {self.unaligned_count}")
        except queue.Empty:
            pass

        self.root.after(100, self._process_queues)

    def run(self):
        self.root.mainloop()
