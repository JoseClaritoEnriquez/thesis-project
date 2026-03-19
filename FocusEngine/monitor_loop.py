import time
import queue
import threading
from datetime import datetime
import json

class MonitorLoop:
    def __init__(self, engine, status_queue, event_queue):
        self.engine = engine
        self.status_queue = status_queue
        self.event_queue = event_queue
        self.unaligned_counter = 0
        self.running = False
        self._stop_event = threading.Event()
        self.user_task = ""
        self.thread = None
        self.interval = 10 # 10s interval for i7-12700 spec

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

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Signals the background thread to stop and waits for it to finish."""
        self.running = False
        self._stop_event.set()  # Wake up the sleeping thread immediately
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=15)  # Wait long enough for any in-flight inference
            if self.thread.is_alive():
                print("Warning: monitor thread did not stop within timeout.")

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
        print(f"Audit Log: {json.dumps(log_entry)}")

    def _run_loop(self):
        print(f"Monitoring loop started. Task: '{self.user_task}'")
        
        while self.running:
            # IDLE state — use Event.wait() so stop() can wake us immediately
            self._stop_event.wait(timeout=self.interval)

            if not self.running:
                break

            # CAPTURE & INFER state
            start_time = time.time()
            try:
                # Post capture request to queue to notify UI (optional, for debug)
                self.event_queue.put({"type": "STATE", "value": "Capturing & Inferencing"})
                
                result = self.engine.capture_and_process(self.user_task)
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                status = result.get("status", "UNKNOWN")
                reasoning = result.get("reasoning", "")
                
                self._log_status(status, latency_ms)
                
                if status == "UNALIGNED":
                    self.unaligned_counter += 1
                    print(f"[UNALIGNED] Count: {self.unaligned_counter}. Reason: {reasoning}")
                else:
                    self.unaligned_counter = 0
                    print(f"[ALIGNED] Resetting counter. Reason: {reasoning}")

                # Check if we need to trigger an intervention (>= 6)
                if self.unaligned_counter >= 6:
                    print(f"Intervention triggered! Counter at {self.unaligned_counter}. Reason: {reasoning}")
                    self.event_queue.put({
                        "type": "INTERVENTION", 
                        "task": self.user_task,
                        "reason": reasoning
                    })
                    # We might pause here until the UI resets us, 
                    # but we can rely on the UI calling self.reset_counter()
                else:
                    self.event_queue.put({"type": "STATE", "value": "IDLE"})
                    self.status_queue.put({
                        "status": status,
                        "counter": self.unaligned_counter,
                        "latency": latency_ms
                    })

            except Exception as e:
                print(f"Error in monitor loop: {e}")
                self.event_queue.put({"type": "ERROR", "value": str(e)})

        print("Monitoring loop stopped.")
