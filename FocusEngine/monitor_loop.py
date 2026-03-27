import time
import queue
import threading
from datetime import datetime
import json
from session_logger import SessionLogger

class MonitorLoop:
    def __init__(self, engine, status_queue, event_queue, device="CPU", log_callback=None):
        self.engine = engine
        self.device = device
        self.status_queue = status_queue
        self.event_queue = event_queue
        self.log_callback = log_callback
        self.unaligned_counter = 0
        self.total_distractions = 0
        self.running = False
        self._stop_event = threading.Event()
        self.user_task = ""
        self.thread = None
        self.session_logger = None
        self.interval = 10 # 10s interval for i7-12700 spec

    def _emit_log(self, message, level="INFO", source="MONITOR"):
        if self.log_callback:
            self.log_callback(message, level=level, source=source)

    def start(self, user_task):
        """Starts the background monitoring loop in a new thread."""
        # Ensure any previous thread is fully stopped before starting a new one
        if self.thread is not None and self.thread.is_alive():
            print("Warning: previous monitor thread still alive, stopping it first...")
            self.stop()

        self._stop_event.clear()
        self.running = True
        self.user_task = user_task
        self.unaligned_counter = 0
        self.total_distractions = 0
        self.session_logger = SessionLogger(user_task)

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print(f"Monitoring started for task: {user_task}")

    def stop(self):
        """Signals the background thread to stop and waits for it to finish."""
        self.running = False
        self._stop_event.set()  # Wake up the sleeping thread immediately
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=15)  # Wait long enough for any in-flight inference
            if self.thread.is_alive():
                print("Warning: monitor thread did not stop within timeout.")
                self._emit_log("Warning: monitor thread did not stop within timeout.", level="ERROR")
        # Write session-end summary to the log file
        if self.session_logger:
            self.session_logger.log_session_end(total_distractions=self.total_distractions)
            self.session_logger = None
        print("Monitoring stopped.")

    def reset_counter(self):
        self.unaligned_counter = 0
        print("Unaligned counter manually reset by UI.")

    def _log_status(self, status, latency_ms):
        """Writes to a local audit log (JSON schema)."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": status,
            "task_context": self.user_task,
            "inference_latency_ms": latency_ms
        }
        
        # In a real app we might write this to a SQLite DB or local JSONL file
        # Keeping it console-only for testing, unless File I/O needed.
        # print(f"Audit Log: {json.dumps(log_entry)}")

    def _run_loop(self):
        print(f"Monitoring started. Task: '{self.user_task}'")
        
        import config_manager
        one_minute_mode = config_manager.get_one_minute_mode()
        next_wait = self.interval
        
        while self.running:
            # IDLE state — use Event.wait() so stop() can wake us immediately
            self._stop_event.wait(timeout=next_wait if one_minute_mode else self.interval)

            if not self.running:
                break

            # CAPTURE & INFER state
            start_time = time.time()
            try:
                # Post capture request to queue to notify UI (optional, for debug)
                self.event_queue.put({"type": "STATE", "value": f"Monitoring ({self.device})"})
                
                result = self.engine.capture_and_process(self.user_task)
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                if one_minute_mode:
                    latency_sec = latency_ms / 1000.0
                    next_wait = max(0.0, 60.0 - latency_sec)
                
                status = result.get("status", "UNKNOWN")
                reasoning = result.get("reasoning", "")
                observed_activity = result.get("observed_activity", "")
                
                if status == "ERROR":
                    print(f"Critical error in inference: {reasoning}")
                    self.running = False
                    self.event_queue.put({"type": "ERROR", "value": f"Monitoring Failed: {reasoning}"})
                    break

                self._log_status(status, latency_ms)

                # Log to session text file
                if self.session_logger:
                    self.session_logger.log_inference(
                        verdict=status,
                        inference_time_ms=latency_ms,
                        observed_activity=observed_activity,
                        reasoning=reasoning,
                    )
                
                if status == "UNALIGNED":
                    self.unaligned_counter += 1
                    self.total_distractions += 1
                    print(
                        f"[UNALIGNED] Observed: {observed_activity} | Reason: {reasoning}"
                    )
                else:
                    self.unaligned_counter = 0
                    print(f"[ALIGNED] Observed: {observed_activity}")

                # Always update the status queue for metrics
                self.status_queue.put({
                    "status": status,
                    "counter": self.unaligned_counter,
                    "total_counter": self.total_distractions,
                    "latency": latency_ms
                })

                # Check if we need to trigger an intervention (>= 6)
                if self.unaligned_counter >= 6:
                    intervention_enabled = config_manager.get_intervention_enabled()
                    if intervention_enabled:
                        print(
                            f"Intervention triggered! Counter at {self.unaligned_counter}. "
                            f"Observed: {observed_activity} | Reason: {reasoning}"
                        )
                        self._emit_log(
                            f"Intervention triggered at {self.unaligned_counter}. Observed: {observed_activity} | Reason: {reasoning}",
                            level="ERROR",
                        )
                        self.event_queue.put({
                            "type": "INTERVENTION", 
                            "task": self.user_task,
                            "reason": reasoning,
                            "observed_activity": observed_activity,
                        })
                        if self.session_logger:
                            self.session_logger.log_intervention(
                                observed_activity=observed_activity,
                                reasoning=reasoning,
                                counter=self.unaligned_counter,
                            )
                    else:
                        self._emit_log(
                            f"Intervention suppressed by settings at {self.unaligned_counter}. Observed: {observed_activity} | Reason: {reasoning}",
                            source="MONITOR",
                        )
                else:
                    self.event_queue.put({"type": "STATE", "value": f"IDLE ({self.device})"})

            except Exception as e:
                print(f"Error in monitor loop: {e}")
                self._emit_log(f"Error in monitor loop: {e}", level="ERROR")
                self.event_queue.put({"type": "ERROR", "value": str(e)})

        print("Monitoring loop stopped.")
