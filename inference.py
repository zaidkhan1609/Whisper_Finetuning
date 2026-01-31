import torch
import librosa
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# ---------- CONFIG ----------
BASE_MODEL = "openai/whisper-medium"
LORA_DIR = "models/whisper-medium-lora"
AUDIO_DIR = Path("data/test_audio")
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

print("Using device:", DEVICE)

# Load processor (always from base model)
processor = WhisperProcessor.from_pretrained(
    BASE_MODEL,
    language="en",
    task="transcribe"
)

# Load base model
base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

# Load LoRA
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.to(DEVICE)
model.eval()

# Collect audio files
audio_files = sorted(AUDIO_DIR.glob("*.mp3"))

assert len(audio_files) > 0, "No mp3 files found in data/test_audio"

print(f"\nFound {len(audio_files)} audio files\n")

for audio_path in audio_files:
    print(f"🎧 Processing: {audio_path.name}")

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features=input_features,
            max_new_tokens=128
        )

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    print("📝 Transcription:")
    print(text)
    print("-" * 60)

print("\n✅ Batch inference complete")
