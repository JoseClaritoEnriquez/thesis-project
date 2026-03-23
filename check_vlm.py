import openvino_genai as ov_genai
import os

# Create a dummy image
from PIL import Image
import numpy as np

img = Image.new('RGB', (100, 100))

print("Checking VLMPipeline help...")
import openvino_genai as ov_genai
print(f"Class doc: {ov_genai.VLMPipeline.__doc__}")
print(f"Init doc: {ov_genai.VLMPipeline.__init__.__doc__}")
print(f"Generate doc: {ov_genai.VLMPipeline.generate.__doc__}")
