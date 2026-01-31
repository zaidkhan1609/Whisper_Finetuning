from transformers import WhisperProcessor
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
audio = torch.randn(16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
print(inputs.input_features.shape)
