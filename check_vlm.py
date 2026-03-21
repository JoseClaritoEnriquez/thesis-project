import openvino_genai as ov_genai
import os

# Create a dummy image
from PIL import Image
import numpy as np

img = Image.new('RGB', (100, 100))

print("Checking VLMPipeline help...")
# Use pydoc or just inspect
import inspect
print(f"Generate signature: {inspect.signature(ov_genai.VLMPipeline.generate)}")

# Or just try to see if it accepts a list of PIL images or numpy arrays
# We can't actually run it without a model, but we can look at the docstring
print(f"Docstring: {ov_genai.VLMPipeline.generate.__doc__}")
