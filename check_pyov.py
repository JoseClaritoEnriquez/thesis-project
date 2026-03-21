import openvino_genai as ov_genai
try:
    print(f"Attributes of py_openvino_genai:")
    import openvino_genai.py_openvino_genai as py_ov
    # Print all that have 'Image' in them
    for attr in dir(py_ov):
        if "Image" in attr:
            print(f"py_ov.{attr}")
except Exception as e:
    print(f"Error: {e}")
