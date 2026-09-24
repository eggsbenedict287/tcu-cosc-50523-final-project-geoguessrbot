"""Download the Streetview by Country dataset from Kaggle into data/raw/.

Requires Kaggle credentials: either ~/.kaggle/kaggle.json, or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables (Colab Secrets work too).

Usage:
    pip install kagglehub
    python scripts/download_data.py
"""
from pathlib import Path

import kagglehub

DATASET = "sylshaw/streetview-by-country"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "streetview-by-country"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(DATASET, output_dir=str(OUT_DIR))
    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
