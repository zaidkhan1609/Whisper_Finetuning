import torch

class WhisperDataCollator:
    def __call__(self, features):
        # audio features (already extracted by WhisperProcessor)
        input_features = [
            torch.tensor(f["input_features"], dtype=torch.float32)
            for f in features
        ]

        # text labels
        labels = [
            torch.tensor(f["labels"], dtype=torch.long)
            for f in features
        ]

        # pad audio (Whisper expects shape: [B, 80, T])
        input_features = torch.nn.utils.rnn.pad_sequence(
            input_features, batch_first=True
        )

        # pad labels
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_features": input_features,
            "labels": labels
        }
