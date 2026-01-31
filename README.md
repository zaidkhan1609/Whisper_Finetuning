# Whisper Fine-Tuning with LoRA (Production-Ready, Local Pipeline)

A **production-oriented Speech-to-Text (STT) system** built by fine-tuning **OpenAI Whisper-Medium** using **LoRA (Low-Rank Adaptation)** — implemented end-to-end with **native PyTorch**, **local data pipelines**, and **custom training/inference logic**.

This project deliberately avoids high-level “auto-magic” abstractions in favor of **explicit control over data flow, tensor interfaces, and performance**.

---

## 🚀 Project Highlights

- 🔧 **From-scratch pipeline**: Audio → CSV manifests → PyTorch training → inference
- 🎙️ **Whisper-Medium + LoRA** fine-tuning (parameter-efficient, GPU-friendly)
- ⚙️ **Pure PyTorch training loop** (no HF Trainer dependency in final version)
- 🧠 Handles real-world issues:
  - Encoder/decoder interface mismatches
  - AMP / CUDA edge cases
  - Custom collators & tensor shape alignment
- 💾 **Local-first**: No cloud dependency, no HF datasets requirement
- 🎧 **MP3-ready inference** (auto-resampling handled)

---

## 📊 Training Results

| Metric | Value |
|------|------|
| Base Model | `openai/whisper-medium` |
| Fine-tuning Method | LoRA (PEFT) |
| Trainable Params | ~4.7M (~0.61%) |
| Initial Loss | ~0.33 |
| Final Loss | ~0.01 |
| Hardware | Single GPU (CUDA + AMP) |

Training was validated against **unseen audio samples**, achieving clean, accurate transcriptions on noisy, real-world speech.

---

## 🗂️ Repository Structure

```text
.
├── training/
│   ├── train_lora.py        # Full training loop (PyTorch + AMP)
│   ├── model.py             # Whisper + LoRA adapter loading
│   ├── dataset.py           # Audio + text preprocessing
│   ├── collator.py          # Custom batch collation
│   └── config.yaml
│
├── inference.py              # Batch inference (MP3/WAV supported)
├── data/
│   ├── processed/            # CSV manifests
│   └── test_audio/           # Sample inference audio
│
├── models/
│   └── whisper-medium-lora/  # Trained LoRA adapters
│
└── README.md
