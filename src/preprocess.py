"""Clean the raw heart data and save it to data/processed/."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "heart_raw.csv"
CLEAN_FILE = ROOT / "data" / "processed" / "heart_clean.csv"

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def make_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET] = (df["num"] > 0).astype(int)
    return df.drop(columns=["num"])


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = make_binary_target(df)
    df = fill_missing(df)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(int)
    return df


if __name__ == "__main__":
    raw = load_raw()
    print("Missing values before:\n", raw.isna().sum()[raw.isna().sum() > 0])
    clean = clean_data(raw)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_FILE, index=False)
    print(f"Missing after: {clean.isna().sum().sum()}")
    print(f"Saved {clean.shape} -> {CLEAN_FILE}")
    print("Class balance:\n", clean[TARGET].value_counts())