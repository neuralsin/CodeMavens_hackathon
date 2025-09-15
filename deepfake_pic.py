import tkinter as tk
from tkinter import filedialog
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch

# model
model_name = "prithivMLmods/deepfake-detector-model-v1"
print("Loading model...... (might take a minute)")
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"    #gpu if laptop supports it
model.to(device)
print(f"Model loaded ✅ (Using device: {device})")

def detect(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    label = "real" if probs[0] >= probs[1] else "fake"
    fake_prob = probs[1].item() * 100  # convert
    real_prob = probs[0].item() * 100

    # result
    
    print("Result saved to reality_check.txt")
    print(f"{label} → Fake: {fake_prob:.1f}%, Real: {real_prob:.1f}%")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # hide root window
    file_path = filedialog.askopenfilename(
        title="Select an Image to Check",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if file_path:
        detect(file_path)
    else:
        print("No file selected. Exiting.")
