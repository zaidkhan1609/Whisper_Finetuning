import librosa
import soundfile as sf
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

SRC_DIR = "data/raw"
DST_DIR = "data/augmented"
NOISE_LEVELS = [0.002, 0.005]

os.makedirs(DST_DIR, exist_ok=True)

def add_noise(audio, level):
    noise = np.random.randn(len(audio))
    return audio + level * noise

for root, _, files in os.walk(SRC_DIR):
    for f in tqdm(files):
        if f.endswith(".wav"):
            src = os.path.join(root, f)
            audio, sr = librosa.load(src, sr=16000)

            for lvl in NOISE_LEVELS:
                noisy = add_noise(audio, lvl)
                out = Path(DST_DIR) / f"{Path(f).stem}_noise{lvl}.wav"
                sf.write(out, noisy, sr)
