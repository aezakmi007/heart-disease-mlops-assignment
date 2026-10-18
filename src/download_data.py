"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/."""
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_FILE = RAW_DIR / "heart_raw.csv"


def download(url: str = URL, out_file: Path = RAW_FILE) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    tmp = RAW_DIR / "cleveland.data"
    tmp.write_bytes(resp.content)

    df = pd.read_csv(tmp, header=None, names=COLUMNS, na_values="?")
    df.to_csv(out_file, index=False)
    tmp.unlink()
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} cols -> {out_file}")
    return out_file


if __name__ == "__main__":
    download()
