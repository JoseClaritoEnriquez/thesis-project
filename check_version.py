import openvino_genai as ov_genai
print(f"Version: {ov_genai.get_version()}")
try:
    print(f"Full version: {ov_genai.__version__}")
except:
    pass

import openvino as ov
print(f"OpenVINO Version: {ov.__version__}")
