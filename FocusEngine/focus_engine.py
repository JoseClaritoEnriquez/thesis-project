import json
import pyautogui
from PIL import Image
import openvino as ov
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
import platform

class FocusEngine:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-500M-Instruct"):
        print(f"Initializing FocusEngine with model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)

        # OS check to ensure we only try loading OpenVINO GPU optimizations on Windows/Intel setups
        current_os = platform.system()
        
        try:
            if current_os == "Windows":
                # High-performance target for i7-12700 + Dual GPU setup
                print("Windows detected. Attempting to load OpenVINO model on GPU.1...")
                self.model = OVModelForVisualCausalLM.from_pretrained(
                    model_id, 
                    export=True, # Set to True if model isn't pre-exported, otherwise False
                    device="GPU.1", 
                    ov_config={"PERFORMANCE_HINT": "LATENCY"}
                )
            else:
                print(f"Non-Windows OS ({current_os}) detected. Falling back to CPU for compatibility testing.")
                # Fallback for Mac (will use CPU instead)
                self.model = OVModelForVisualCausalLM.from_pretrained(
                    model_id, 
                    export=True, 
                    device="CPU"
                )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model on desired device: {e}")
            print("Falling back to standard CPU execution...")
            self.model = OVModelForVisualCausalLM.from_pretrained(
                model_id, 
                export=True, 
                device="CPU"
            )

    def capture_and_process(self, user_task):
        """
        Captures the screen, parses it through the VLM using a CoT prompt, 
        and returns the alignment status.
        Uses 768x768 resolution for better text clarity reading code/docs.
        """
        print("Capturing screenshot...")
        # 1. Capture & Immediate Resize (Memory Efficiency) -> 768x768 for i7-12700
        screenshot = pyautogui.screenshot().resize((768, 768), Image.LANCZOS)
        
        # 2. Run Inference
        result = self.run_inference(screenshot, user_task)
        
        # 3. Clean up RAM natively
        # With 16GB, we rely on python's GC and Windows natural caching, no explicit windll calls.
        del screenshot
        
        return result

    def get_prompt(self, user_task):
        return f"""Goal: {user_task}
Image: Desktop Screenshot
Instructions:
1. Identify the primary active window.
2. Determine if the content is a tool (IDE, Terminal), research (Docs, Stack Overflow), or distraction (Social Media, YouTube, Gaming, News).
3. If the user is in a 'Neutral' app (Browser), look at the URL or page title.
Return purely JSON format: {{"status": "ALIGNED"|"UNALIGNED", "reasoning": "Reason here"}}"""

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
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)
            
            # Decode output
            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            # Output format looks like: "user\n[IMAGE]\n{prompt}\nassistant\n{response}"
            # Extract just the assistant response
            full_response = generated_texts[0]
            assistant_response = full_response.split("assistant\n")[-1].strip()
            
            # Attempt to parse json
            # First, clean up any markdown formatting if the model wrapped it in ```json
            clean_json = assistant_response.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(clean_json)
                return result
            except json.JSONDecodeError:
                print(f"Failed to parse JSON. Raw output: {assistant_response}")
                # Fallback heuristic
                if "UNALIGNED" in assistant_response.upper():
                    return {"status": "UNALIGNED", "reasoning": "Failed to parse JSON. Fallback to unaligned."}
                return {"status": "ALIGNED", "reasoning": "Failed to parse JSON. Assuming aligned."}
                
        except Exception as e:
            print(f"Inference error: {e}")
            return {"status": "ERROR", "reasoning": str(e)}

if __name__ == "__main__":
    # Quick test function
    print("Testing Focus Engine Initialization...")
    engine = FocusEngine()
    print("Testing Capture and Process...")
    res = engine.capture_and_process("Writing Python code for Focus Engine")
    print(f"Result: {res}")
