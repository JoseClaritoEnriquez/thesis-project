import time
import random
from datetime import datetime

# We speed up the simulation so you don't have to wait 5 minutes to see it work
TIME_SCALE = 10.0

class MockEngine:
    def capture_and_process(self, task):
        # Simulate inference taking a random time between 15 to 45 seconds
        latency = random.uniform(15.0, 45.0)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   -> Mock inference working (simulated {latency:.1f}s computation)...")
        time.sleep(latency / TIME_SCALE)
        return {"status": "ALIGNED", "latency_sec": latency}

def simulate_loop():
    engine = MockEngine()
    interval = 10
    one_minute_mode = True
    next_wait = interval
    
    print(f"--- Starting 1-Minute Cycle Simulation (Running {TIME_SCALE}x faster) ---")
    
    for i in range(1, 6): # Simulate 5 cycles
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cycle {i}: Idle wait for {next_wait:.1f}s...")
        time.sleep(next_wait / TIME_SCALE)
        
        # In the real app, this is (time.time() - start_time)
        result = engine.capture_and_process("Test Task")
        latency_sec = result["latency_sec"]
        
        # Core logic we added
        if one_minute_mode:
            next_wait = max(0.0, 60.0 - latency_sec)
            
        cycle_total = latency_sec + next_wait
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   -> Inference took {latency_sec:.1f}s.")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   -> Remaining time in minute: {next_wait:.1f}s. Total cycle length: {cycle_total:.1f}s.")

if __name__ == "__main__":
    simulate_loop()
