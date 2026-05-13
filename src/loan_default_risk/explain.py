from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline


def create_shap_report(
    model_name: str,
    pipeline: Pipeline,
    x_reference: pd.DataFrame,
    output_dir: str | Path,
    max_rows: int = 1000,
) -> None:
    import shap

    output_path = Path(output_dir) / "shap" / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    sample = x_reference.sample(min(max_rows, len(x_reference)), random_state=42)
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()

    dense_transformed = _to_dense(transformed)
    explainer = _build_explainer(estimator, dense_transformed)
    explanation = explainer(dense_transformed)
    values = _positive_class_values(explanation.values)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_path / "feature_importance.csv", index=False)

    shap.summary_plot(values, dense_transformed, feature_names=feature_names, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(output_path / "summary_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        values,
        dense_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=25,
    )
    plt.tight_layout()
    plt.savefig(output_path / "summary_bar.png", dpi=180, bbox_inches="tight")
    plt.close()


def _build_explainer(estimator, dense_transformed):
    import shap

    model_name = estimator.__class__.__name__.lower()
    if "randomforest" in model_name or "lgbm" in model_name:
        return shap.TreeExplainer(estimator)
    if "logisticregression" in model_name:
        masker = shap.maskers.Independent(dense_transformed, max_samples=min(200, dense_transformed.shape[0]))
        return shap.LinearExplainer(estimator, masker)
    return shap.Explainer(estimator.predict_proba, dense_transformed)


def _positive_class_values(values: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def _to_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return matrix
