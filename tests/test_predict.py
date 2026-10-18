import pytest

from features import FEATURES
from predict import MODEL_PATH, predict


def test_predict_output_format(sample_df, trained_model):
    records = sample_df[FEATURES].head(3).to_dict("records")
    out = predict(records, model=trained_model)
    assert len(out) == 3
    for r in out:
        assert set(r) == {"prediction", "label", "confidence", "probability_disease"}
        assert r["prediction"] in (0, 1)
        assert 0.5 <= r["confidence"] <= 1.0
        assert r["label"] == ("disease" if r["prediction"] else "no disease")


def test_predict_rejects_missing_feature(sample_df, trained_model):
    record = sample_df[FEATURES].head(1).to_dict("records")[0]
    del record["age"]
    with pytest.raises(KeyError):
        predict([record], model=trained_model)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not present")
def test_saved_model_predicts_on_sample_patient():
    sample = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
              "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
              "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
    assert predict([sample])[0]["prediction"] in (0, 1)