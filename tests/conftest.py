import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import FEATURES, TARGET, build_preprocessor


@pytest.fixture
def sample_df():
    """80 fake patients with the same columns as the real cleaned data."""
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame({
        "age": rng.integers(30, 75, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 350, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.random(n) * 4,
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    }).astype("float64")
    df[TARGET] = ((df["oldpeak"] + rng.normal(0, 0.5, n)) > 1.5).astype(int)
    return df


@pytest.fixture
def trained_model(sample_df):
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    return pipe.fit(sample_df[FEATURES], sample_df[TARGET])