# I can't easily get a VLMDecodedResults instance without a model, 
# but I'll check its type if I just print it.
# Usually it's a list.
import openvino_genai as ov_genai
print(f"DecodedResults type: {type(ov_genai.DecodedResults)}")
# Actually, the user's error says 'object has no attribute lower'.
# This proves it's an instance of VLMDecodedResults.
