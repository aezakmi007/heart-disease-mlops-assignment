"""Package the trained model: metadata + MLflow-format copy + round-trip check."""
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn

from features import CATEGORICAL_COLS, FEATURES, NUMERIC_COLS, TARGET

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
MLFLOW_DIR = ROOT / "models" / "mlflow_model"
DATA = ROOT / "data" / "processed" / "heart_clean.csv"
COMPARISON = ROOT / "reports" / "model_comparison.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    model = joblib.load(MODEL_PATH)
    best = pd.read_csv(COMPARISON).iloc[0]          # sorted by cv_roc_auc
    X = pd.read_csv(DATA)[FEATURES].astype("float64")

    metadata = {
        "model_name": best["model"],
        "best_params": best["best_params"],
        "metrics": {k: float(best[k]) for k in best.index
                    if k.startswith("test_") or k == "cv_roc_auc"},
        "features": {"numeric": NUMERIC_COLS, "categorical": CATEGORICAL_COLS},
        "target": TARGET,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_data_sha256": sha256(DATA),
        "model_sha256": sha256(MODEL_PATH),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "random_state": 42,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {META_PATH.relative_to(ROOT)}")

    if MLFLOW_DIR.exists():
        shutil.rmtree(MLFLOW_DIR)
    mlflow.sklearn.save_model(
        model, str(MLFLOW_DIR),
        serialization_format="cloudpickle",
        input_example=X.head(2),
    )
    print(f"Wrote {MLFLOW_DIR.relative_to(ROOT)}/")

    # round-trip check: all three loading paths must agree
    from_joblib = joblib.load(MODEL_PATH).predict_proba(X)
    from_mlflow = mlflow.sklearn.load_model(str(MLFLOW_DIR)).predict_proba(X)
    assert np.allclose(from_joblib, from_mlflow), "Predictions differ after reload!"
    print("Round-trip check passed: joblib and MLflow copies give identical predictions.")


if __name__ == "__main__":
    main()