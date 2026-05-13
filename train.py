from __future__ import annotations

import argparse
from pathlib import Path

from src.loan_default_risk.data import (
    LENDINGCLUB_KAGGLE_DATASET,
    LENDINGCLUB_MODEL_FEATURES,
    download_lendingclub_kaggle,
    load_dataset,
)
from src.loan_default_risk.explain import create_shap_report
from src.loan_default_risk.modeling import fit_and_evaluate, save_metrics, split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train loan default risk classifiers.")
    parser.add_argument("--data", default=None, help="Path to Home Credit or LendingClub CSV file.")
    parser.add_argument(
        "--kaggle-lendingclub",
        action="store_true",
        help="Download the Lending Club dataset from KaggleHub and train on it.",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default=LENDINGCLUB_KAGGLE_DATASET,
        help="Kaggle dataset handle used with --kaggle-lendingclub.",
    )
    parser.add_argument(
        "--kaggle-file",
        default=None,
        help="Optional CSV file path inside the Kaggle dataset. If omitted, the largest CSV is used.",
    )
    parser.add_argument(
        "--kaggle-output-dir",
        default=None,
        help="Optional local directory for KaggleHub downloads. By default KaggleHub uses its cache.",
    )
    parser.add_argument("--force-download", action="store_true", help="Force KaggleHub to download again.")
    parser.add_argument("--target-col", default=None, help="Optional target column. Inferred when omitted.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Read only the first N rows. Useful for quick checks on large datasets.",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Use every CSV column. By default Lending Club runs use a curated non-leakage feature set.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for metrics, models, and SHAP reports.")
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.20,
        help="Minimum default-class precision while maximizing recall. Raise this to reduce false alarms.",
    )
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP report generation.")
    args = parser.parse_args()
    if args.kaggle_lendingclub and args.data:
        parser.error("Use either --data or --kaggle-lendingclub, not both.")
    if not args.kaggle_lendingclub and not args.data:
        parser.error("Provide --data path.csv or use --kaggle-lendingclub.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data
    target_col = args.target_col
    if args.kaggle_lendingclub:
        data_path = download_lendingclub_kaggle(
            dataset=args.kaggle_dataset,
            file_path=args.kaggle_file,
            output_dir=args.kaggle_output_dir,
            force_download=args.force_download,
        )
        target_col = target_col or "loan_status"
        print(f"Using Kaggle dataset CSV: {data_path}")

    feature_cols = None
    if args.kaggle_lendingclub and target_col == "loan_status" and not args.all_features:
        feature_cols = LENDINGCLUB_MODEL_FEATURES

    dataset = load_dataset(data_path, target_col, feature_cols=feature_cols, nrows=args.sample_rows)
    print(f"Loaded {len(dataset.target):,} rows with target '{dataset.target_name}'.")
    print(f"Default rate: {dataset.target.mean():.2%}")

    data = split_data(dataset.features, dataset.target)
    results = fit_and_evaluate(data, output_dir, args.min_precision)
    metrics = save_metrics(results, output_dir)
    print("\nModel leaderboard:")
    print(metrics[["model", "recall_default", "precision_default", "average_precision", "threshold"]].to_string(index=False))

    if not args.skip_shap:
        best = results[0]
        best_name = metrics.iloc[0]["model"]
        best = next(result for result in results if result.name == best_name)
        print(f"\nCreating SHAP report for best recall-oriented model: {best.name}")
        create_shap_report(best.name, best.pipeline, data.x_test, output_dir)


if __name__ == "__main__":
    main()
