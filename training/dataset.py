import pandas as pd
import librosa
from datasets import Dataset

def load_csv_dataset(csv_path, max_samples=None):
    df = pd.read_csv(csv_path)

    if max_samples is not None:
        df = df.head(max_samples)

    return Dataset.from_pandas(df)


import torch
import librosa

def preprocess(batch, processor, sample_rate):
    # Load audio
    audio, sr = librosa.load(batch["audio_path"], sr=sample_rate)

    # Convert audio → log-mel features
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    # Tokenize text → labels ONLY
    labels = processor.tokenizer(
        batch["sentence"],
        return_tensors="pt"
    )

    return {
        "input_features": inputs.input_features[0],  # Tensor [80, T]
        "labels": labels[0],                          # Tensor [L]
    }
