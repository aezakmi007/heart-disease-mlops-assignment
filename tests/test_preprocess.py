import numpy as np
import pandas as pd

from preprocess import CATEGORICAL_COLS, TARGET, clean_data, fill_missing, make_binary_target


def make_raw():
    return pd.DataFrame({
        "age": [63, 67, 41, 56, 57, 44], "sex": [1, 1, 0, 1, 0, 1],
        "cp": [1, 4, 2, 2, 4, 3], "trestbps": [145, 160, 130, 120, 140, 150],
        "chol": [233, 286, 204, 236, 354, 240], "fbs": [1, 0, 0, 0, 0, 1],
        "restecg": [2, 2, 2, 0, 0, 0], "thalach": [150, 108, 172, 178, 160, 165],
        "exang": [0, 1, 0, 0, 1, 0], "oldpeak": [2.3, 1.5, 1.4, 0.8, 0.6, 0.0],
        "slope": [3, 2, 1, 1, 1, 2], "ca": [0, 3, 0, 0, 0, 0],
        "thal": [6, 3, 3, 3, 3, 7], "num": [0, 2, 0, 0, 1, 4],
    }).astype("float64")


def test_binary_target_maps_any_disease_to_one():
    df = make_binary_target(make_raw())
    assert list(df[TARGET]) == [0, 1, 0, 0, 1, 1]
    assert "num" not in df.columns


def test_binary_target_does_not_modify_input():
    raw = make_raw()
    make_binary_target(raw)
    assert "num" in raw.columns


def test_fill_missing_uses_mode_for_categorical():
    df = make_raw()
    df["ca"] = [1, 1, 1, 2, np.nan, 0]
    assert fill_missing(df)["ca"].iloc[4] == 1


def test_fill_missing_uses_median_for_numeric():
    df = make_raw()
    df["chol"] = [100, 200, 300, np.nan, 400, 500]
    assert fill_missing(df)["chol"].iloc[3] == 300


def test_clean_data_has_no_missing_and_int_categoricals():
    df = make_raw()
    df.loc[0, "thal"] = np.nan
    df.loc[1, "ca"] = np.nan
    out = clean_data(df)
    assert out.isna().sum().sum() == 0
    assert all(out[c].dtype.kind == "i" for c in CATEGORICAL_COLS)


def test_clean_data_drops_duplicates():
    df = pd.concat([make_raw(), make_raw().iloc[[0]]], ignore_index=True)
    assert len(clean_data(df)) == 6