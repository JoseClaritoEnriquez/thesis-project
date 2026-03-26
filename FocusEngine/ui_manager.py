import os
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText
import queue
import threading
import pystray
import config_manager
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

class SplashScreen:
    """A minimal loading window to show while the heavy ML model is loading."""
    def __init__(self, parent=None):
        if parent:
            self.root = tk.Toplevel(parent)
            self.is_toplevel = True
        else:
            self.root = tk.Tk()
            self.is_toplevel = False
            
        self.root.title("Focus Engine")
        self.status_queue = queue.Queue()
        self.after_id = None
        self.is_closing = False
        
        # Solarized-esque colors to match main UI
        bg_color = "#fdf6e3"
        self.root.configure(bg=bg_color)
        
        # Center the window on screen
        window_width = 400
        window_height = 200
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        self.root.overrideredirect(True) # Remove window decorations for splash feel
        self.root.attributes("-topmost", True)
        
        # Make the window draggable
        self._drag_data = {"x": 0, "y": 0}
        self.root.bind("<Button-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._do_drag)

        main_frame = tk.Frame(self.root, bg=bg_color, highlightbackground="#b58900", highlightthickness=2)
        main_frame.pack(expand=True, fill="both")

        # Minimize button
        min_btn = tk.Button(
            main_frame, text="—", font=("Helvetica", 10, "bold"),
            bg=bg_color, fg="#657b83", relief="flat", cursor="hand2",
            command=self._minimize, padx=5
        )
        min_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-5, y=5)

        tk.Label(
            main_frame, text="Focus Engine", font=("Helvetica", 18, "bold"),
            bg=bg_color, fg="#b58900"
        ).pack(pady=(30, 10))

        self.status_label = tk.Label(
            main_frame, text="Loading AI Model...\n(This may take up to a minute)", 
            font=("Helvetica", 10), bg=bg_color, fg="#657b83", wraplength=350
        )
        self.status_label.pack(pady=10)
        
        try:
            self.after_id = self.root.after(100, self._check_status)
            self.root.update()
        except tk.TclError:
            pass

    def _check_status(self):
        """Periodically check the queue for status updates."""
        if self.is_closing:
            return
            
        try:
            if not self.root.winfo_exists():
                return
            
            while True:
                msg = self.status_queue.get_nowait()
                if msg == "CLOSE":
                    self.close()
                    return
                self.status_label.config(text=msg)
        except (queue.Empty, tk.TclError, AttributeError):
            pass
            
        try:
            if not self.is_closing and self.root.winfo_exists():
                self.after_id = self.root.after(100, self._check_status)
        except (tk.TclError, AttributeError):
            pass

    def update_status(self, msg):
        """Thread-safe status update."""
        self.status_queue.put(msg)

    def _start_drag(self, event):
        """Remembers the starting point of a drag."""
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _do_drag(self, event):
        """Moves the window based on mouse motion."""
        x = self.root.winfo_x() - self._drag_data["x"] + event.x
        y = self.root.winfo_y() - self._drag_data["y"] + event.y
        self.root.geometry(f"+{x}+{y}")

    def _minimize(self):
        """Minimizes the splash screen and ensures it shows in the taskbar."""
        self.root.update_idletasks()
        self.root.withdraw()
        self.root.overrideredirect(False)
        self.root.state('iconic')
        self.root.deiconify()
        self.root.bind("<Map>", self._on_map)

    def _on_map(self, event):
        """Restores overrideredirect when coming back from minimized state."""
        self.root.overrideredirect(True)
        self.root.unbind("<Map>")

    def close(self):
        """Safely closes the splash screen."""
        if self.is_closing:
            return
        self.is_closing = True
        
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except:
                pass
            self.after_id = None
            
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except:
            pass

class UIManager:
    @staticmethod
    def prompt_for_model_dir():
        """Shows a folder selection dialog for the model directory."""
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Select Folder to Store AI Model (1.5GB)")
        root.destroy()
        return path

    def __init__(self, start_callback, stop_callback, reset_callback, settings_changed_callback, log_queue=None):
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.reset_callback = reset_callback
        self.settings_changed_callback = settings_changed_callback
        self.is_closing = False
        
        self.status_queue = queue.Queue()
        self.event_queue = queue.Queue()
        self.log_queue = log_queue if log_queue is not None else queue.Queue()
        self.log_history = []
        self.max_log_history = 1000
        self.output_window: tk.Toplevel = None  # type: ignore
        self.output_text: ScrolledText = None  # type: ignore
        
        self.root = tk.Tk()
        self.root.title("Focus Engine (High Performance)")
        self.root.geometry("450x350")
        self.root.resizable(False, False)
        
        # Simple Solarized-esque styling for baseline
        self.bg_color = "#fdf6e3"
        self.fg_color = "#657b83"
        self.root.configure(bg=self.bg_color)
        
        self.tray_icon: pystray.Icon = None  # type: ignore
        self.frame: tk.Frame = None  # type: ignore
        self.timer_id: str = None  # type: ignore
        self.remaining_seconds = 0
        self.session_task = ""
        self.unaligned_count = 0
        self.total_unaligned_count = 0
        
        # UI Elements defined in views
        self.status_label: tk.Label = None  # type: ignore
        self.counter_label: tk.Label = None  # type: ignore
        self.total_counter_label: tk.Label = None  # type: ignore
        self.timer_label: tk.Label = None  # type: ignore
        self.stop_button: tk.Button = None  # type: ignore
        
        self._build_setup_view()
        
        # Periodic UI update to check queues without blocking Tk mainloop
        try:
            self.root.after(100, self._process_queues)
        except tk.TclError:
            pass
        
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
        self.start_button.pack(pady=(20, 10), fill="x")

        # Settings button
        self.settings_button = tk.Button(
            self.frame, text="⚙️ Settings", bg=self.bg_color, fg=self.fg_color,
            font=("Helvetica", 10), command=self._build_settings_view,
            relief="flat", cursor="hand2"
        )
        self.settings_button.pack(pady=5)

        self.status_label = tk.Label(
            self.frame, text=f"Status: IDLE ({config_manager.get_device()})", bg=self.bg_color, fg="#859900"
        )
        self.status_label.pack(side="bottom")

    def _build_settings_view(self):
        """Builds the settings screen."""
        self._clear_frame()
        self.frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        self.frame.pack(expand=True, fill="both")

        tk.Label(
            self.frame, text="Settings", font=("Helvetica", 18, "bold"), 
            bg=self.bg_color, fg="#b58900"
        ).pack(pady=(0, 20))

        tk.Button(
            self.frame,
            text="View Output Terminal",
            bg="#268bd2",
            fg="white",
            font=("Helvetica", 10, "bold"),
            command=self.open_output_terminal,
        ).pack(fill="x", pady=(0, 15))

        self.one_min_var = tk.BooleanVar(value=config_manager.get_one_minute_mode())
        tk.Checkbutton(
            self.frame, text="1-Minute Inference Cycle (Wait if <1m)", variable=self.one_min_var,
            bg=self.bg_color, fg=self.fg_color, font=("Helvetica", 10),
            selectcolor=self.bg_color, activebackground=self.bg_color
        ).pack(anchor="w", pady=(0, 15))

        # Button row
        btn_frame = tk.Frame(self.frame, bg=self.bg_color)
        btn_frame.pack(fill="x", side="bottom")

        tk.Button(
            btn_frame, text="Cancel", bg="#93a1a1", fg="white",
            font=("Helvetica", 10, "bold"), command=self._build_setup_view,
            padx=10
        ).pack(side="left")

        tk.Button(
            btn_frame, text="Save & Apply", bg="#2aa198", fg="black",
            font=("Helvetica", 12, "bold"), command=self._on_save_settings,
            padx=20
        ).pack(side="right")

    def _format_log_line(self, entry):
        timestamp = entry.get("timestamp", "")
        level = entry.get("level", "INFO")
        source = entry.get("source", "APP")
        message = str(entry.get("message", ""))
        
        # Simplify common levels/sources to reduce clutter
        prefix = ""
        if level != "INFO" or source not in ("STDOUT", "APP", "ENGINE", "MONITOR"):
            prefix = f" [{level}]"
        
        return f"[{timestamp}]{prefix} {message}"

    def _append_log_entry(self, entry):
        line = self._format_log_line(entry)
        self.log_history.append(line)
        if len(self.log_history) > self.max_log_history:
            self.log_history = self.log_history[-self.max_log_history :]

        if self.output_text and self.output_text.winfo_exists():
            self.output_text.config(state="normal")
            self.output_text.insert("end", line + "\n")
            self.output_text.see("end")
            self.output_text.config(state="disabled")

    def open_output_terminal(self):
        if self.output_window and self.output_window.winfo_exists():
            self.output_window.deiconify()
            self.output_window.lift()
            self.output_window.focus_force()
            return

        self.output_window = tk.Toplevel(self.root)
        self.output_window.title("Focus Engine - Live Output")
        self.output_window.geometry("900x520")
        self.output_window.configure(bg=self.bg_color)

        header = tk.Label(
            self.output_window,
            text="Live Input / Output / Errors",
            bg=self.bg_color,
            fg="#b58900",
            font=("Helvetica", 12, "bold"),
        )
        header.pack(anchor="w", padx=12, pady=(10, 6))

        self.output_text = ScrolledText(
            self.output_window,
            wrap="word",
            font=("Consolas", 9),
            bg="#002b36",
            fg="#93a1a1",
            insertbackground="#93a1a1",
        )
        self.output_text.pack(expand=True, fill="both", padx=12, pady=(0, 10))
        self.output_text.config(state="normal")
        for line in self.log_history:
            self.output_text.insert("end", line + "\n")
        self.output_text.see("end")
        self.output_text.config(state="disabled")

        def _on_close():
            if self.output_window and self.output_window.winfo_exists():
                self.output_window.destroy()
            self.output_window = None
            self.output_text = None

        self.output_window.protocol("WM_DELETE_WINDOW", _on_close)

    def _on_save_settings(self):
        if hasattr(self, 'one_min_var'):
            config_manager.set_one_minute_mode(self.one_min_var.get())
        self._build_setup_view()

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
        self.total_unaligned_count = 0
        
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
            info_frame, text="Nudge Counter: 0/6", bg=self.bg_color, fg=self.fg_color,
            font=("Helvetica", 10)
        )
        self.counter_label.pack(side="left", padx=(20, 0))

        self.total_counter_label = tk.Label(
            info_frame, text="Total Distractions: 0", bg=self.bg_color, fg=self.fg_color,
            font=("Helvetica", 10)
        )
        self.total_counter_label.pack(side="right")

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

    def _on_error(self, message):
        """Stops the session and shows an error message."""
        if self.timer_id:
            try:
                self.root.after_cancel(self.timer_id)
            except:
                pass
            self.timer_id = None
        self.stop_callback()
        self._build_error_view(message)

    def _build_error_view(self, message):
        """Shows a failure summary when monitoring crashes."""
        self._clear_frame()
        self.frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        self.frame.pack(expand=True, fill="both")

        tk.Label(
            self.frame, text="Monitoring Failed", font=("Helvetica", 18, "bold"),
            bg=self.bg_color, fg="#dc322f"
        ).pack(pady=(0, 20))

        tk.Label(
            self.frame, text=message,
            font=("Helvetica", 11), bg=self.bg_color, fg=self.fg_color,
            wraplength=400, justify="center"
        ).pack(pady=20)

        tk.Button(
            self.frame, text="Back to Setup", bg="#2aa198", fg="black",
            font=("Helvetica", 12, "bold"), command=self._build_setup_view
        ).pack(pady=10, fill="x")

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
            self.frame, text=f"Total distractions in session: {self.total_unaligned_count}",
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
        self.is_closing = True
        self.stop_callback()
        if self.tray_icon:
            self.tray_icon.stop()
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
        
        # Hard exit to ensure no background worker processes keep the app alive on Windows.
        os._exit(0)

    def trigger_intervention(self, task, reasoning, observed_activity=None):
        """Displays the 'Nudge' full-screen topmost modal."""
        if self.is_closing:
            return
            
        # Solarized Dark / High contrast Red
        try:
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

            if observed_activity:
                tk.Label(
                    frame,
                    text=f"Observed: {observed_activity}",
                    font=("Helvetica", 16),
                    fg="#eee8d5",
                    bg="#dc322f",
                    justify="center",
                    wraplength=800,
                ).pack(pady=10)

            def on_return():
                self.reset_callback()
                modal.destroy()

            btn = tk.Button(
                frame, text="✅ I'm back on track!", font=("Helvetica", 24, "bold"),
                bg="#859900", fg="black", command=on_return, padx=20, pady=10
            )
            btn.pack(pady=40)
        except tk.TclError:
            pass

    def _process_queues(self):
        """Reads from queues to update UI based on AI worker thread state."""
        if self.is_closing:
            return
            
        try:
            if not self.root.winfo_exists():
                return
                
            while True:
                event = self.event_queue.get_nowait()
                event_type = event.get("type")
                
                if event_type == "STATE":
                    if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                        self.status_label.config(text=f"Status: {event.get('value')}")
                elif event_type == "INTERVENTION":
                    self.trigger_intervention(
                        event.get("task"),
                        event.get("reason"),
                        event.get("observed_activity"),
                    )
                elif event_type == "ERROR":
                    self._on_error(event.get("value"))
        except queue.Empty:
            pass
        except (tk.TclError, AttributeError):
            return

        # Process terminal logs from background workers.
        try:
            while True:
                entry = self.log_queue.get_nowait()
                if isinstance(entry, dict):
                    self._append_log_entry(entry)
                else:
                    self._append_log_entry(
                        {
                            "timestamp": "",
                            "level": "INFO",
                            "source": "APP",
                            "message": str(entry),
                        }
                    )
        except queue.Empty:
            pass
        except (tk.TclError, AttributeError):
            return

        # Also check status_queue for counter updates
        try:
            while True:
                status = self.status_queue.get_nowait()
                self.unaligned_count = status.get("counter", self.unaligned_count)
                self.total_unaligned_count = status.get("total_counter", self.total_unaligned_count)
                
                if hasattr(self, 'counter_label') and self.counter_label.winfo_exists():
                    self.counter_label.config(text=f"Nudge Counter: {self.unaligned_count}/6")
                
                if hasattr(self, 'total_counter_label') and self.total_counter_label.winfo_exists():
                    self.total_counter_label.config(text=f"Total Distractions: {self.total_unaligned_count}")
        except queue.Empty:
            pass
        except (tk.TclError, AttributeError):
            return

        try:
            if not self.is_closing and self.root.winfo_exists():
                self.root.after(100, self._process_queues)
        except (tk.TclError, AttributeError):
            pass

    def run(self):
        try:
            self.root.mainloop()
        except (tk.TclError, KeyboardInterrupt):
            pass
