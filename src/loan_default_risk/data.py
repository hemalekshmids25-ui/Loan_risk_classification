from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LENDINGCLUB_KAGGLE_DATASET = "adarshsng/lending-club-loan-data-csv"
DEFAULT_TARGET_CANDIDATES = ("TARGET", "loan_status", "default", "Default", "bad_loan")

LENDINGCLUB_MODEL_FEATURES = (
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "collections_12_mths_ex_med",
    "application_type",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "mort_acc",
    "pub_rec_bankruptcies",
    "tax_liens",
)

LENDINGCLUB_DEFAULT_STATUSES = {
    "charged off",
    "default",
    "does not meet the credit policy. status:charged off",
    "late (31-120 days)",
}

LENDINGCLUB_NON_DEFAULT_STATUSES = {
    "fully paid",
    "current",
    "does not meet the credit policy. status:fully paid",
    "in grace period",
    "late (16-30 days)",
}


@dataclass(frozen=True)
class LoadedDataset:
    features: pd.DataFrame
    target: pd.Series
    target_name: str


def download_lendingclub_kaggle(
    dataset: str = LENDINGCLUB_KAGGLE_DATASET,
    file_path: str | None = None,
    output_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download the Lending Club Kaggle dataset and return the CSV path."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for --kaggle-lendingclub. "
            "Install it with: pip install kagglehub[pandas-datasets]"
        ) from exc

    download_path = kagglehub.dataset_download(
        dataset,
        path=file_path,
        output_dir=str(output_dir) if output_dir is not None else None,
        force_download=force_download,
    )
    path = Path(download_path)
    if path.is_file():
        return path

    csv_files = sorted(path.rglob("*.csv"), key=lambda item: item.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in downloaded Kaggle dataset: {path}")
    return csv_files[0]


def load_dataset(
    csv_path: str | Path,
    target_col: str | None = None,
    feature_cols: Iterable[str] | None = None,
    nrows: int | None = None,
) -> LoadedDataset:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    read_kwargs = {"low_memory": True}
    if nrows is not None:
        read_kwargs["nrows"] = nrows
    if feature_cols is not None:
        requested_cols = set(feature_cols)
        if target_col is not None:
            requested_cols.add(target_col)
        read_kwargs["usecols"] = lambda col: col in requested_cols

    df = pd.read_csv(path, **read_kwargs)
    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    target_name = target_col or _infer_target_column(df.columns)
    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' was not found in {path}")

    target = normalize_target(df[target_name], target_name)
    features = df.drop(columns=[target_name])

    valid_mask = target.notna()
    dropped = len(target) - int(valid_mask.sum())
    if dropped:
        print(f"Dropped {dropped:,} rows with unknown target labels.")

    return LoadedDataset(
        features=features.loc[valid_mask].reset_index(drop=True),
        target=target.loc[valid_mask].astype(int).reset_index(drop=True),
        target_name=target_name,
    )


def normalize_target(target: pd.Series, target_name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(target):
        values = pd.to_numeric(target, errors="coerce")
        unique = set(values.dropna().unique().tolist())
        if unique.issubset({0, 1}):
            return values
        return (values > 0).astype("float").where(values.notna())

    cleaned = target.astype("string").str.strip().str.lower()
    if target_name.lower() == "loan_status":
        mapped = pd.Series(np.nan, index=target.index, dtype="float")
        mapped.loc[cleaned.isin(LENDINGCLUB_DEFAULT_STATUSES)] = 1.0
        mapped.loc[cleaned.isin(LENDINGCLUB_NON_DEFAULT_STATUSES)] = 0.0
        return mapped

    positive = {"1", "yes", "y", "true", "default", "bad", "charged off"}
    negative = {"0", "no", "n", "false", "non-default", "good", "fully paid"}
    mapped = pd.Series(np.nan, index=target.index, dtype="float")
    mapped.loc[cleaned.isin(positive)] = 1.0
    mapped.loc[cleaned.isin(negative)] = 0.0
    return mapped


def _infer_target_column(columns: Iterable[str]) -> str:
    columns = list(columns)
    for candidate in DEFAULT_TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not infer target column. Pass --target-col. "
        f"Tried: {', '.join(DEFAULT_TARGET_CANDIDATES)}"
    )
