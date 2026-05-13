from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - handled at runtime for optional installs
    LGBMClassifier = None


RANDOM_STATE = 42


@dataclass(frozen=True)
class PreparedData:
    x_train: pd.DataFrame
    x_valid: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class ModelResult:
    name: str
    pipeline: Pipeline
    threshold: float
    metrics: dict[str, float | int | list[list[int]] | str]


def split_data(features: pd.DataFrame, target: pd.Series) -> PreparedData:
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    return PreparedData(x_train, x_valid, x_test, y_train, y_valid, y_test)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in features.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_factories() -> dict[str, Callable[[], object]]:
    factories: dict[str, Callable[[], object]] = {
        "logistic_regression": lambda: LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
            solver="saga",
        ),
        "random_forest": lambda: RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=350,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    if LGBMClassifier is not None:
        factories["lightgbm"] = lambda: LGBMClassifier(
            class_weight="balanced",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return factories


def fit_and_evaluate(
    data: PreparedData,
    output_dir: str | Path,
    min_precision: float,
) -> list[ModelResult]:
    output_path = Path(output_dir)
    model_path = output_path / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    results: list[ModelResult] = []
    for name, factory in model_factories().items():
        print(f"Training {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(data.x_train)),
                ("model", factory()),
            ]
        )
        pipeline.fit(data.x_train, data.y_train)

        valid_scores = pipeline.predict_proba(data.x_valid)[:, 1]
        threshold = select_recall_threshold(data.y_valid, valid_scores, min_precision)
        test_scores = pipeline.predict_proba(data.x_test)[:, 1]
        metrics = evaluate_predictions(data.y_test, test_scores, threshold)

        with (model_path / f"{name}.pkl").open("wb") as model_file:
            pickle.dump(pipeline, model_file)
        results.append(ModelResult(name, pipeline, threshold, metrics))

    return results


def select_recall_threshold(y_true: pd.Series, scores: np.ndarray, min_precision: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    threshold_frame = pd.DataFrame(
        {
            "threshold": np.r_[thresholds, 1.0],
            "precision": precision,
            "recall": recall,
        }
    )
    feasible = threshold_frame[threshold_frame["precision"] >= min_precision]
    if feasible.empty:
        # Fall back to the maximum-recall point when the requested precision is unreachable.
        return float(threshold_frame.sort_values(["recall", "precision"], ascending=False).iloc[0]["threshold"])
    best = feasible.sort_values(["recall", "precision"], ascending=False).iloc[0]
    return float(best["threshold"])


def evaluate_predictions(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    return {
        "threshold": threshold,
        "recall_default": float(report["1"]["recall"]),
        "precision_default": float(report["1"]["precision"]),
        "f1_default": float(report["1"]["f1-score"]),
        "approval_risk_false_negatives": int(confusion_matrix(y_true, predictions)[1, 0]),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def save_metrics(results: list[ModelResult], output_dir: str | Path) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"model": result.name, **result.metrics}
        rows.append(row)
    metrics_df = pd.DataFrame(rows).sort_values(
        ["recall_default", "precision_default", "average_precision"],
        ascending=False,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(Path(output_dir) / "metrics.csv", index=False)
    return metrics_df


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=20)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
