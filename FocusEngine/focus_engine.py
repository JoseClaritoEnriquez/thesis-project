import os
import sys
import json
import re
import pyautogui
import openvino_genai as ov_genai
from PIL import Image
from huggingface_hub import snapshot_download

# Focus Engine Class (Torch-Free / OpenVINO Native)
class FocusEngine:
    # Pre-exported OpenVINO model directory (relative to bundle_dir)
    MODEL_DIR = "model_openvino"

    def __init__(self, model_id=None, status_callback=None):
        def _report(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        _report(f"Initializing FocusEngine (OpenVINO GenAI)...")
        
        # Determine the absolute path to the model directory
        if getattr(sys, 'frozen', False):
            # In packaged app, model is in _internal/model_openvino
            self.model_path = os.path.join(sys._MEIPASS, "_internal", self.MODEL_DIR)
        else:
            # In dev, use the current folder or a specified path
            self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.MODEL_DIR)

        try:
            # Check if model exists AND weights are present, if not download it
            model_xml = os.path.join(self.model_path, "openvino_language_model.xml")
            model_bin = os.path.join(self.model_path, "openvino_language_model.bin")
            
            if not os.path.exists(model_xml) or not os.path.exists(model_bin):
                _report(f"Model files not found. Starting download...")
                os.makedirs(self.model_path, exist_ok=True)
                snapshot_download(
                    repo_id="helenai/Qwen2-VL-2B-Instruct-ov-int4",
                    local_dir=self.model_path,
                    local_dir_use_symlinks=False
                )
                _report("Download complete.")

            _report(f"Loading OpenVINO GenAI pipeline...")
            # VLMPipeline handles preprocessing, tokenization, and generation natively
            self.pipeline = ov_genai.VLMPipeline(self.model_path, "CPU")
            _report("Focus Engine Initialized successfully.")
        except Exception as e:
            _report(f"Error: Failed to load or download model: {e}")
            raise RuntimeError(f"Failed to load VLM Pipeline: {e}") from e

    def capture_and_process(self, user_task):
        """
        Captures the screen, parses it through the VLM, 
        and returns the alignment status.
        """
        print("Capturing screenshot...")
        # Capture and resize while preserving aspect ratio (keeps text readable)
        screenshot = pyautogui.screenshot()
        # Scale so the longest side is at most 960px to balance RAM usage and text readability
        max_side = 960
        w, h = screenshot.size
        scale = min(max_side / w, max_side / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        screenshot = screenshot.resize(new_size, Image.LANCZOS)
        
        # 2. Run Inference
        result = self.run_inference(screenshot, user_task)
        
        # 3. Clean up RAM natively
        import gc
        del screenshot
        gc.collect()
        
        return result

    def get_prompt(self, user_task):
        return (
            f"The user's current task is: \"{user_task}\"\n"
            f"Analyze the screenshot for evidence of this specific task. Avoid generic assumptions.\n\n"
            f"Rules:\n"
            f"- ALIGNED: The screen shows clear evidence of tools, documents, or content directly supporting the task \"{user_task}\".\n"
            f"- UNALIGNED: The screen shows content unrelated to the specific task (e.g., entertainment, social media, or a completely different work task).\n"
            f"- Be critical. Do not assume a connection exists (e.g., 'programming' is not 'chatting' unless you see chat-specific code).\n"
            f"- If there is no clear evidence of the specific task, choose UNALIGNED.\n\n"
            f"Reply in this format:\n"
            f"Visible Evidence: <list specific UI elements, text, or apps seen>\n"
            f"Reasoning: <explain why the evidence supports or contradicts the task \"{user_task}\">\n"
            f"Answer: ALIGNED or UNALIGNED"
        )

    def run_inference(self, image, task):
        prompt_text = self.get_prompt(task)
        
        try:
            # 2. Run Inference
            # Convert PIL image to openvino.Tensor (v2026.0+ removes ov_genai.Image)
            import numpy as np
            import openvino as ov
            
            # Ensure RGB and convert to numpy
            ov_image = ov.Tensor(np.array(image.convert("RGB")))
            
            print("Running native OpenVINO inference...")
            # Simple greedy generation
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = 200
            
            # Run native OpenVINO inference
            raw_output = self.pipeline.generate(
                prompt_text,
                image=ov_image,
                generation_config=config
            )
            
            # Extract text from result object (v2026.0 returns VLMDecodedResults)
            if hasattr(raw_output, 'texts') and len(raw_output.texts) > 0:
                raw_output_text = raw_output.texts[0]
            else:
                raw_output_text = str(raw_output)

            print(f"Raw model output (text): {raw_output_text}")
            
            # Cleanup
            import gc
            del ov_image
            gc.collect()
            
            return self._parse_response(raw_output_text)
                
        except Exception as e:
            print(f"Inference error: {e}")
            return {"status": "ERROR", "reasoning": str(e)}

    def _parse_response(self, raw_output):
        """Extract status and reasoning from model output."""
        status = "ALIGNED"  # default
        reasoning = raw_output 

        # 1. Primary Check: Look for the 'Answer:' field explicitly
        match = re.search(r"Answer:\s*(ALIGNED|UNALIGNED)", raw_output, re.IGNORECASE)
        if match:
            status = match.group(1).upper()
            return {"status": status, "reasoning": reasoning}

        # 2. Fallback: Determine verdict via keyword and negation analysis
        lower = raw_output.lower()
        
        negation_patterns = [
            r"not\s+(?:\w+\s+)*aligned",
            r"isn'?t\s+(?:\w+\s+)*aligned",
            r"not\s+(?:\w+\s+)*related",
            r"isn'?t\s+(?:\w+\s+)*related",
            r"not\s+(?:\w+\s+)*focused",
            r"off[\s-]?task", r"distract",
        ]
        
        if any(re.search(p, lower) for p in negation_patterns):
            status = "UNALIGNED"
        elif "UNALIGNED" in raw_output.upper():
            status = "UNALIGNED"
        elif "ALIGNED" in raw_output.upper():
            status = "ALIGNED"
        else:
            # Keyword fallback for distraction detection
            distraction_keywords = ["social media", "youtube", "gaming",
                                    "entertainment", "news feed", "video game"]
            if any(kw in lower for kw in distraction_keywords):
                status = "UNALIGNED"

        return {"status": status, "reasoning": reasoning}

if __name__ == "__main__":
    # Quick test function
    print("Testing Focus Engine Initialization...")
    engine = FocusEngine()
    print("Testing Capture and Process...")
    res = engine.capture_and_process("Writing Python code for Focus Engine")
    print(f"Result: {res}")
