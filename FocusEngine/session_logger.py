"""
Session Logger for FocusEngine
Logs each inference pass to a timestamped JSON file per monitoring session.
"""

import os
import json
import threading
from datetime import datetime


class SessionLogger:
    """Writes inference results to a per-session JSON log file."""

    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_logs")

    def __init__(self, intended_task: str):
        self.intended_task = intended_task
        self._lock = threading.Lock()

        os.makedirs(self.LOG_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_task = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in intended_task)[:50].strip()
        filename = f"session_{timestamp}_{safe_task}.json"
        self.filepath = os.path.join(self.LOG_DIR, filename)

        # Initialize the JSON structure
        self._data = {
            "intended_task": intended_task,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "finish_time": None,
            "total_distractions": 0,
            "inferences": [],
            "interventions": [],
        }
        self._flush()

    def _flush(self):
        """Write the current data to disk."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def log_inference(self, verdict: str, inference_time_ms: int, observed_activity: str, reasoning: str):
        """Append one inference result to the session log."""
        entry = {
            "verdict": verdict,
            "inference_time_ms": inference_time_ms,
            "observed_activity": observed_activity,
            "reasoning": reasoning,
        }
        with self._lock:
            self._data["inferences"].append(entry)
            self._flush()

    def log_intervention(self, observed_activity: str, reasoning: str, counter: int):
        """Log when an intervention is triggered."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "observed_activity": observed_activity,
            "reasoning": reasoning,
            "consecutive_unaligned": counter,
        }
        with self._lock:
            self._data["interventions"].append(entry)
            self._flush()

    def log_session_end(self, total_distractions: int = 0):
        """Write session-end metadata."""
        with self._lock:
            self._data["finish_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._data["total_distractions"] = total_distractions
            self._flush()
