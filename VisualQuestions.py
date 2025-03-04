import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Image URL
img_url = 'https://st.focusedcollection.com/9264440/i/650/focused_193439804-stock-photo-scenic-view-city-skyline-zurich.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Specify the question you want to ask about the image
question = "What is in the image?"

# Use the processor to prepare inputs for VQA (image + question)
inputs = processor(raw_image, question, return_tensors="pt")

# Generate the answer from the model
out = model.generate(**inputs)

# Decode and print the answer to the question
answer = processor.decode(out[0], skip_special_tokens=True)
print(f"Answer: {answer}")