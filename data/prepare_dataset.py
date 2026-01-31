import os
import csv
from tqdm import tqdm

RAW_DIR = "data/raw"
OUT_DIR = "data/metadata"

os.makedirs(OUT_DIR, exist_ok=True)

SPLITS = {
    "train": "cv-valid-train.csv",
    "val": "cv-valid-dev.csv",
    "test": "cv-valid-test.csv",
}

def process_split(split_name, csv_file):
    rows = []
    csv_path = os.path.join(RAW_DIR, csv_file)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f"Processing {split_name}"):

            audio_path = os.path.join(RAW_DIR, row["filename"].split("/", 1)[0], row["filename"])

            if not os.path.exists(audio_path):
                continue

            # Use duration from CSV if available
            try:
                duration = float(row["duration"]) if row.get("duration") else -1.0
            except ValueError:
                duration = -1.0

            rows.append({
                "audio_path": audio_path.replace("\\", "/"),
                "transcript": row["text"].strip(),
                "duration": duration,
                "accent": row.get("accent", "unknown") or "unknown",
                "noise_level": "clean"
            })

    return rows

def write_csv(name, rows):
    out_path = os.path.join(OUT_DIR, f"{name}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "transcript", "duration", "accent", "noise_level"]
        )
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    counts = {}

    for split, csv_file in SPLITS.items():
        rows = process_split(split, csv_file)
        write_csv(split, rows)
        counts[split] = len(rows)

    print("\nDataset preparation complete:")
    for k, v in counts.items():
        print(f"{k.capitalize()}: {v}")
