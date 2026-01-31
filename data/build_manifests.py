import os
import pandas as pd

ROOT = "data/raw"

SPLITS = {
    "train": ("cv-valid-train.csv", "cv-valid-train", "train.csv"),
    "val": ("cv-valid-dev.csv", "cv-valid-dev", "val.csv"),
}

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


def build_split(csv_name, audio_dir, out_name):
    csv_path = os.path.join(ROOT, csv_name)
    audio_root = os.path.join(ROOT, audio_dir)

    df = pd.read_csv(csv_path)

    rows = []
    for _, row in df.iterrows():
        # ✅ YOUR CSV USES `filename`
        audio_path = os.path.join(audio_root, row["filename"])

        if not os.path.exists(audio_path):
            continue

        rows.append({
            "audio_path": os.path.abspath(audio_path),
            "sentence": row["text"],   # ✅ YOUR CSV USES `text`
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, out_name)
    out_df.to_csv(out_path, index=False)

    print(f"Saved {len(out_df)} samples → {out_path}")


if __name__ == "__main__":
    for split in SPLITS.values():
        build_split(*split)
