# Loan Default Risk Classification for Banking

This project trains recall-oriented loan default classifiers for Home Credit or LendingClub-style CSV datasets. It compares:

- Class-weighted Logistic Regression
- Class-weighted Random Forest
- Class-weighted LightGBM

The positive class is `default = 1`. The training script selects the decision threshold that maximizes default-class recall while meeting a configurable minimum precision, which helps minimize risky approvals.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Place datasets under `data/` or pass any CSV path.

To download the Lending Club data directly from Kaggle, use KaggleHub:

```powershell
python train.py --kaggle-lendingclub --output-dir outputs\lendingclub --skip-shap
```

The default Kaggle dataset is `adarshsng/lending-club-loan-data-csv`. If KaggleHub asks for credentials, create an API token from your Kaggle account settings and place `kaggle.json` in `%USERPROFILE%\.kaggle\`.

For a faster first run on a laptop, train on a subset:

```powershell
python train.py --kaggle-lendingclub --sample-rows 100000 --output-dir outputs\lendingclub --skip-shap
```

If you already know the file name inside the Kaggle dataset, you can pass it explicitly:

```powershell
python train.py --kaggle-lendingclub --kaggle-file loan.csv --output-dir outputs\lendingclub --skip-shap
```

Supported target inference:

- Home Credit: `TARGET`
- LendingClub: `loan_status`
- Generic: `default`, `Default`, or `bad_loan`

For LendingClub, default-like statuses such as `Charged Off`, `Default`, and `Late (31-120 days)` are mapped to `1`; known non-default statuses such as `Fully Paid` and `Current` are mapped to `0`.

## Train

```powershell
python train.py --data data\application_train.csv
```

For LendingClub:

```powershell
python train.py --data data\accepted_2007_to_2018Q4.csv --target-col loan_status
```

Tune the recall/false-alarm tradeoff:

```powershell
python train.py --data data\application_train.csv --min-precision 0.30
```

Quick smoke test without SHAP:

```powershell
python train.py --data tests\smoke_data.csv --target-col loan_status --output-dir outputs\smoke --skip-shap
```

If `lightgbm` is not installed, the script still trains Logistic Regression and Random Forest. Install all requirements to include LightGBM and SHAP reporting.

## Outputs

The script writes:

- `outputs/metrics.csv`: model leaderboard with recall, precision, AUC, average precision, threshold, and confusion matrix
- `outputs/models/*.pkl`: trained preprocessing + model pipelines
- `outputs/shap/<model>/feature_importance.csv`: top SHAP drivers
- `outputs/shap/<model>/summary_beeswarm.png`: SHAP directionality plot
- `outputs/shap/<model>/summary_bar.png`: global SHAP importance plot

## Streamlit App

Train first, then run:

```powershell
streamlit run app.py
```

The app loads `outputs/smoke` by default. Change the output directory in the sidebar after training on a full dataset.

## Why Recall First?

For credit risk screening, a false negative means an applicant who later defaults was approved as safe. The default threshold is therefore optimized to catch as many defaults as possible while preserving a minimum precision floor. Increase `--min-precision` when operational review capacity is limited.

## Regulatory Explainability

SHAP reports are generated for the best recall-oriented model so model drivers can be reviewed by risk, compliance, and audit teams. Keep the generated feature importance CSV and plots with each model run to support model governance documentation.
