import openvino_genai as ov_genai
import openvino as ov
import numpy as np
from PIL import Image

def find_image_class(obj, name_prefix="ov_genai"):
    for attr in dir(obj):
        if "Image" in attr:
            print(f"Found {name_prefix}.{attr}")
        sub = getattr(obj, attr)
        if hasattr(sub, "__name__") and sub.__name__ == "openvino_genai":
             continue # Avoid infinite loop
             
find_image_class(ov_genai)
try:
    import openvino_genai.py_openvino_genai as py_ov
    find_image_class(py_ov, "py_ov")
except:
    pass

# Try to see if VLMPipeline has any nested classes?
print(f"VLMPipeline attributes: {dir(ov_genai.VLMPipeline)}")
