import os
import csv
import librosa

DATA_DIR = "data"
META_DIR = "data/metadata"

def test_audio_files_load():
    for root, _, files in os.walk(os.path.join(DATA_DIR, "raw")):
        for f in files:
            if f.endswith(".wav"):
                path = os.path.join(root, f)
                audio, sr = librosa.load(path, sr=16000)
                assert len(audio) > 0
                assert sr == 16000

def test_transcripts_exist():
    for root, _, files in os.walk(os.path.join(DATA_DIR, "raw")):
        for f in files:
            if f.endswith(".wav"):
                txt = os.path.join(root, f.replace(".wav", ".txt"))
                assert os.path.exists(txt)

def test_metadata_files_exist():
    for split in ["train", "val", "test"]:
        path = os.path.join(META_DIR, f"{split}.csv")
        assert os.path.exists(path)

def test_metadata_schema():
    path = os.path.join(META_DIR, "train.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        required = {"audio_path", "transcript", "duration", "accent", "noise_level"}
        assert required.issubset(reader.fieldnames)

def test_audio_paths_in_metadata():
    path = os.path.join(META_DIR, "train.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert os.path.exists(row["audio_path"])
