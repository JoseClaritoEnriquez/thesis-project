import openvino_genai as ov_genai
print(f"VLMDecodedResults attributes: {dir(ov_genai.DecodedResults)}")
# Wait, let's check what VLMPipeline.generate returns exactly
# It's usually a subclass of DecodedResults or similar.
