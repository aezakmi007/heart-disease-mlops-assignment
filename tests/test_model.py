import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from features import FEATURES, TARGET, build_preprocessor
from train import evaluate


def test_evaluate_returns_all_metrics_in_valid_range(trained_model, sample_df):
    m = evaluate(trained_model, sample_df[FEATURES], sample_df[TARGET])
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_predict_proba_is_valid(trained_model, sample_df):
    proba = trained_model.predict_proba(sample_df[FEATURES])
    assert proba.shape == (len(sample_df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_training_is_reproducible(sample_df):
    def fit():
        pipe = Pipeline([
            ("prep", build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=20, random_state=0)),
        ])
        return pipe.fit(sample_df[FEATURES], sample_df[TARGET])

    X = sample_df[FEATURES]
    assert np.allclose(fit().predict_proba(X), fit().predict_proba(X))