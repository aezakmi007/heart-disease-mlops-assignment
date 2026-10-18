"""Train, tune and evaluate Logistic Regression and Random Forest."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from features import FEATURES, TARGET, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42
EXPERIMENT_NAME = "heart-disease-classification"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 4, 8],
            "clf__min_samples_leaf": [1, 3],
        },
    ),
}


def load_data():
    df = pd.read_csv(DATA)
    return df[FEATURES], df[TARGET]


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def save_plots(name, model, X_test, y_test):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title(f"Confusion matrix – {name}")
    plt.savefig(FIG_DIR / f"cm_{name}.png", dpi=150, bbox_inches="tight"); plt.close()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC curve – {name}")
    plt.savefig(FIG_DIR / f"roc_{name}.png", dpi=150, bbox_inches="tight"); plt.close()


def log_cv_candidates(search):
    """Log every grid-search combination as a nested child run."""
    cv = search.cv_results_
    for i, params in enumerate(cv["params"]):
        with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
            mlflow.log_params({k.replace("clf__", ""): v for k, v in params.items()})
            mlflow.log_metric("cv_roc_auc_mean", cv["mean_test_score"][i])
            mlflow.log_metric("cv_roc_auc_std", cv["std_test_score"][i])


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, best_models, run_ids = [], {}, {}
    for name, (clf, grid) in MODELS.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"model_type": name, "stage": "tuning"})
            mlflow.log_params({
                "model": name,
                "test_size": 0.2,
                "cv_folds": 5,
                "random_state": RANDOM_STATE,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "param_grid": json.dumps(grid),
            })

            pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)

            # every combination tried, as child runs
            log_cv_candidates(search)

            # best params + metrics
            mlflow.log_params({f"best_{k.replace('clf__', '')}": v
                               for k, v in search.best_params_.items()})
            test_metrics = evaluate(search.best_estimator_, X_test, y_test)
            mlflow.log_metric("cv_roc_auc", search.best_score_)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            # plots as artifacts
            save_plots(name, search.best_estimator_, X_test, y_test)
            mlflow.log_artifact(str(FIG_DIR / f"cm_{name}.png"), "plots")
            mlflow.log_artifact(str(FIG_DIR / f"roc_{name}.png"), "plots")

            # model (pipeline = preprocessing + classifier)
            mlflow.sklearn.log_model(
                search.best_estimator_,
                name="model",
                input_example=X_test.head(2),
                serialization_format="cloudpickle",
            )

            results.append({
                "model": name,
                "best_params": search.best_params_,
                "cv_roc_auc": round(search.best_score_, 4),
                **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()},
            })
            best_models[name] = search.best_estimator_
            run_ids[name] = run.info.run_id
            print(f"\n{name}  CV ROC-AUC={search.best_score_:.4f}  test={test_metrics}")

    results_df = pd.DataFrame(results).sort_values("cv_roc_auc", ascending=False)
    (ROOT / "reports").mkdir(exist_ok=True)
    results_df.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    print("\n", results_df.to_string(index=False))

    winner = results_df.iloc[0]["model"]
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_models[winner], MODEL_DIR / "model.joblib")

    # mark the winning run + attach the comparison table to it
    with mlflow.start_run(run_id=run_ids[winner]):
        mlflow.set_tag("selected_as_best", "true")
        mlflow.log_artifact(str(ROOT / "reports" / "model_comparison.csv"))
    print(f"\nBest model: {winner} -> models/model.joblib (run {run_ids[winner]})")


if __name__ == "__main__":
    main()
