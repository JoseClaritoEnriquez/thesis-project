import json
import re
import pyautogui
from PIL import Image
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
import torch

class FocusEngine:
    # Pre-exported OpenVINO model (no PyTorch loading, no on-the-fly export needed)
    OV_MODEL_ID = "echarlaix/SmolVLM2-2.2B-Instruct-openvino"
    # Original model for the processor/tokenizer
    PROCESSOR_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    def __init__(self, model_id=None):
        print(f"Initializing FocusEngine...")
        
        self.processor = AutoProcessor.from_pretrained(self.PROCESSOR_ID)

        try:
            print(f"Loading pre-exported OpenVINO model: {self.OV_MODEL_ID}")
            self.model = OVModelForVisualCausalLM.from_pretrained(
                self.OV_MODEL_ID,
                device="CPU",
                ov_config={"PERFORMANCE_HINT": "LATENCY"}
            )
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def capture_and_process(self, user_task):
        """
        Captures the screen, parses it through the VLM using a CoT prompt, 
        and returns the alignment status.
        """
        print("Capturing screenshot...")
        # Capture and resize while preserving aspect ratio (keeps text readable)
        screenshot = pyautogui.screenshot()
        # Scale so the longest side is at most 1280px
        max_side = 1280
        w, h = screenshot.size
        scale = min(max_side / w, max_side / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        screenshot = screenshot.resize(new_size, Image.LANCZOS)
        
        # 2. Run Inference
        result = self.run_inference(screenshot, user_task)
        
        # 3. Clean up RAM natively
        # With 16GB, we rely on python's GC and Windows natural caching, no explicit windll calls.
        del screenshot
        
        return result

    def get_prompt(self, user_task):
        return (
            f"The user's current task is: \"{user_task}\"\n"
            f"Look at this screenshot and describe the main application and content visible.\n"
            f"Then decide if the user is working on their task or is distracted.\n"
            f"ALIGNED means the screen shows a tool or content useful for the task "
            f"(e.g., Word/Docs for writing, IDE for coding, browser with research for studying).\n"
            f"UNALIGNED means the screen shows something clearly unrelated "
            f"(e.g., games, social media, entertainment videos).\n"
            f"Reply in this format:\n"
            f"Screen: [describe what app and content you see]\n"
            f"Answer: ALIGNED or UNALIGNED"
        )

    def run_inference(self, image, task):
        prompt_text = self.get_prompt(task)
        
        # Format the messages following the processor's chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        try:
            # Process inputs
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")

            # Generate
            print("Running inference...")
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                repetition_penalty=1.2,
            )
            
            # Decode only the NEW tokens (skip the input prompt tokens)
            input_len = inputs["input_ids"].shape[-1]
            generated_texts = self.processor.batch_decode(
                generated_ids[:, input_len:], skip_special_tokens=True
            )
            
            raw_output = generated_texts[0].strip()
            print(f"Raw model output: {raw_output}")
            
            return self._parse_response(raw_output)
                
        except Exception as e:
            print(f"Inference error: {e}")
            return {"status": "ERROR", "reasoning": str(e)}

    def _parse_response(self, raw_output):
        """Extract status and reasoning from model output."""
        status = "ALIGNED"  # default
        reasoning = raw_output[:300]

        # Try to extract the Screen: description
        screen_match = re.search(r'Screen:\s*(.+?)(?:\n|Verdict|$)', raw_output, re.IGNORECASE | re.DOTALL)
        if screen_match:
            reasoning = screen_match.group(1).strip()

        # Determine verdict — check negations BEFORE positive match
        lower = raw_output.lower()
        
        # Check for negated alignment first (e.g., "not aligned", "not directly related")
        negation_patterns = [
            r"not\s+(?:\w+\s+)*aligned",   # "not aligned", "not directly aligned"
            r"isn'?t\s+(?:\w+\s+)*aligned",
            r"not\s+(?:\w+\s+)*related",    # "not related", "not directly related"
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
