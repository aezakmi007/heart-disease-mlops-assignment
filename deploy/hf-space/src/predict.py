"""Inference helper: raw patient record(s) -> prediction + confidence."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict(records: list, model=None) -> list:
    """records: list of dicts with the 13 feature names as keys."""
    model = model or load_model()
    X = pd.DataFrame(records)[FEATURES].astype("float64")
    p_disease = model.predict_proba(X)[:, 1]
    results = []
    for p in p_disease:
        label = int(p >= 0.5)
        results.append({
            "prediction": label,
            "label": "disease" if label else "no disease",
            "confidence": round(float(p if label else 1 - p), 4),
            "probability_disease": round(float(p), 4),
        })
    return results


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "sample_input.json"
    record = json.loads(path.read_text())
    print(json.dumps(predict([record])[0], indent=2))
