import yaml
import torch
import librosa
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


# =========================
# Dataset
# =========================

class WhisperDataset(Dataset):
    def __init__(self, csv_path, processor, sample_rate, max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.head(max_samples)
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio, _ = librosa.load(row["audio_path"], sr=self.sample_rate)

        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(
            row["sentence"],
            return_tensors="pt"
        ).input_ids[0]

        return input_features, labels


# =========================
# Collator
# =========================

def collate_fn(batch):
    inputs, labels = zip(*batch)

    input_features = torch.stack(inputs)

    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )

    return {
        "input_features": input_features,
        "labels": labels,
    }


# =========================
# Training
# =========================

def main():
    with open("training/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    processor = WhisperProcessor.from_pretrained(
        cfg["model"]["name"],
        language=cfg["model"]["language"],
        task=cfg["model"]["task"]
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        cfg["model"]["name"]
    )

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=cfg["model"]["language"],
        task=cfg["model"]["task"]
    )
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # ===== LoRA =====
    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["dropout"],
        task_type="SEQ_2_SEQ_LM",
        bias="none",
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to(device)

    # ===== Data =====
    train_ds = WhisperDataset(
        cfg["data"]["train_csv"],
        processor,
        cfg["data"]["sample_rate"],
        cfg["data"]["max_train_samples"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1,               # Whisper-medium safe on 8GB
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"])
    )

    scaler = torch.cuda.amp.GradScaler()

    # ===== Training Loop =====
    model.train()
    print("Starting training…")

    for epoch in range(cfg["training"]["num_train_epochs"]):
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()

            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model.base_model(
                    input_features=input_features,
                    labels=labels
                )

                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

    # ===== Save =====
    model.save_pretrained(cfg["training"]["output_dir"])
    processor.save_pretrained(cfg["training"]["output_dir"])
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
