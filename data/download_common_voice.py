import os
import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

OUT_DIR = "data/raw/common_voice_india"
SAMPLE_RATE = 16000
MAX_SAMPLES = 2000

os.makedirs(OUT_DIR, exist_ok=True)

print("Streaming Common Voice (this bypasses Windows HF bugs)...")

dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "en",
    split="validated",
    streaming=True
)

def is_indian_accent(example):
    accent = example.get("accent")
    if not accent:
        return False
    accent = accent.lower()
    return "india" in accent or "south asia" in accent

count = 0

for sample in tqdm(dataset):
    if count >= MAX_SAMPLES:
        break

    if not is_indian_accent(sample):
        continue

    audio = sample.get("audio")
    text = sample.get("sentence")

    if audio is None or text is None:
        continue

    audio_array = audio["array"]
    sr = audio["sampling_rate"]

    if sr != SAMPLE_RATE:
        audio_array = librosa.resample(audio_array, sr, SAMPLE_RATE)

    wav_path = Path(OUT_DIR) / f"cv_{count}.wav"
    txt_path = wav_path.with_suffix(".txt")

    sf.write(wav_path, audio_array, SAMPLE_RATE)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text.strip())

    count += 1

print(f"Saved {count} Indian-accent samples to {OUT_DIR}")
