import numpy as np

from features import FEATURES, NUMERIC_COLS, build_preprocessor


def test_output_rows_match_and_no_nans(sample_df):
    X = sample_df[FEATURES]
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert not np.isnan(Xt).any()


def test_numeric_columns_are_standardised(sample_df):
    Xt = build_preprocessor().fit_transform(sample_df[FEATURES])
    num = Xt[:, : len(NUMERIC_COLS)]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(num.std(axis=0), 1, atol=1e-6)


def test_categoricals_are_one_hot_encoded(sample_df):
    Xt = build_preprocessor().fit_transform(sample_df[FEATURES])
    assert Xt.shape[1] > len(FEATURES)          # more columns after one-hot
    cat = Xt[:, len(NUMERIC_COLS):]
    assert set(np.unique(cat)) <= {0.0, 1.0}


def test_unseen_category_does_not_crash(sample_df):
    prep = build_preprocessor().fit(sample_df[FEATURES])
    row = sample_df[FEATURES].head(1).copy()
    row["cp"] = 99.0
    assert prep.transform(row).shape[0] == 1


def test_missing_values_are_imputed(sample_df):
    prep = build_preprocessor().fit(sample_df[FEATURES])
    row = sample_df[FEATURES].head(1).copy()
    row["chol"] = np.nan
    row["thal"] = np.nan
    assert not np.isnan(prep.transform(row)).any()
