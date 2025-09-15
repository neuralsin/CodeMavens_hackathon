import tkinter as tk
from tkinter import filedialog
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch, cv2, numpy as np

# model
model_name = "prithivMLmods/deepfake-detector-model-v1"
print("Loading model... (first run might take a minute)")
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

#gpu preference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded ✅ (Using device: {device})")


def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_probs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 30 == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                frame_probs.append(float(probs[1]))  # Class 1 = fake
                print(f"Frame {frame_idx}: Fake probability = {probs[1]*100:.1f}%")
        frame_idx += 1
    cap.release()

    avg_fake = np.mean(frame_probs) if frame_probs else 0.0
    verdict = "fake" if avg_fake > 0.5 else "real"

    
    
    print(f"\n--- Deepfake Detector Result ---")
    print(f"Video: {video_path}")
    print(f"Average Fake Probability: {avg_fake*100:.1f}%")
    print(f"Verdict saved to reality_check.txt → {verdict}")
    print("--------------------------------")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        title="Select a Video to Check",
        filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")]
    )
    if file_path:
        detect_video(file_path)
    else:
        print("No file selected. Exiting.")
