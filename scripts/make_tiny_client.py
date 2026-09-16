#!/usr/bin/env python3
import argparse
import csv
import os

parser = argparse.ArgumentParser(description="Create a synthetic Flower ASR smoke-test client")
parser.add_argument(
    "--baseline-root",
    default=os.environ.get("FEDWAV2VEC_ROOT"),
    help="Path to the fedwav2vec2 baseline root (or set FEDWAV2VEC_ROOT)",
)
args = parser.parse_args()
if not args.baseline_root:
    parser.error("provide --baseline-root or set FEDWAV2VEC_ROOT")

try:
    import numpy as np
    import soundfile as sf
except ImportError as exc:
    raise RuntimeError("Install requirements.txt before creating test audio.") from exc

root = os.path.abspath(args.baseline_root)
datadir = os.path.join(root, "data", "client_0")
os.makedirs(datadir, exist_ok=True)

# Make a 2-second 16kHz sine wave
sr = 16000
t = np.linspace(0, 2.0, int(sr*2.0), endpoint=False)
y = 0.1*np.sin(2*np.pi*440.0*t).astype("float32")
wav_path = os.path.join(datadir, "tiny.wav")
sf.write(wav_path, y, sr)

# Minimal CSVs expected by dataset.py
rows = [
    {"ID": "utt0", "wav": wav_path, "start_seg":"0.0", "end_seg":"2.0", "char":"HELLO WORLD", "duration":"2.0"}
]
for split in ["ted_train.csv","ted_dev.csv","ted_test.csv"]:
    with open(os.path.join(datadir, split), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID","wav","start_seg","end_seg","char","duration"])
        w.writeheader(); w.writerows(rows)

print("Tiny client created under", datadir)
