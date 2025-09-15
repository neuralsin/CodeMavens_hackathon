import torch
import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()
audio_file = filedialog.askopenfilename(
    title="Select an audio file",
    filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
)
if not audio_file:
    raise ValueError("⚠️ No file selected. Exiting...")

print(f"✅ Selected file: {audio_file}")


y, sr = sf.read(audio_file)
if y.ndim > 1:               # mono aud
    y = y.mean(axis=1)

#Resample to 16k 
if sr != 16000:
    print(f"Resampling from {sr} Hz to 16000 Hz...")
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sr = 16000


model_name = "MelodyMachine/Deepfake-audio-detection-V2"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


inputs = extractor(y, sampling_rate=sr, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}         #inputs for the Mahcine learning model


with torch.no_grad():
    outputs = model(**inputs)                   #outputs from the model after using tensor input from above
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

fake_prob = probs[0][1].item()
real_prob = probs[0][0].item()


print(f"\n Deepfake Audio Detection")
print(f"Fake probability ❌: {fake_prob*100:.1f}%")
print(f"Real probability✅: {real_prob*100:.1f}%") #.1 accuracy

with open("reality_check.txt", "w") as f:
    verdict = "fake" if fake_prob > real_prob else "real"   #saving to text file for web integration
    f.write(verdict)

print(f"Result saved to reality_check.txt → {verdict.upper()}")
