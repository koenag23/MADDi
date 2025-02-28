import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# ANOTHER VERSION OF STARTER CODE
# Load model directly
# from transformers import AutoProcessor, AutoModelForImageTextToText

# processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
# model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

model_id = "meta-llama/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
