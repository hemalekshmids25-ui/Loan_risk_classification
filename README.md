# Loan Default Risk Classification for Banking

Machine learning project for identifying loan applicants who are likely to default. The project uses LendingClub-style loan data, recall-oriented model selection, SHAP explainability, and a Streamlit app for interactive risk review.

## Project Overview

Loan default prediction helps financial institutions reduce risky approvals while keeping lending decisions explainable. This repository combines:

- Exploratory data analysis
- Feature engineering and preprocessing
- Class-weighted classification models
- Recall-focused threshold tuning
- SHAP explainability reports
- Streamlit deployment

The positive class is `default = 1`. The training workflow selects the decision threshold that maximizes default-class recall while meeting a configurable minimum precision floor.

## Models

The training script compares:

- Class-weighted Logistic Regression
- Class-weighted Random Forest
- Class-weighted LightGBM, when installed

If `lightgbm` is unavailable, the project still trains Logistic Regression and Random Forest.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Place datasets under `data/` or pass any CSV path directly.

To download the LendingClub data through KaggleHub:

```powershell
python train.py --kaggle-lendingclub --output-dir outputs\lendingclub --skip-shap
```

The default Kaggle dataset is `adarshsng/lending-club-loan-data-csv`. If KaggleHub asks for credentials, create a Kaggle API token and place `kaggle.json` in `%USERPROFILE%\.kaggle\`.

For a faster first run:

```powershell
python train.py --kaggle-lendingclub --sample-rows 100000 --output-dir outputs\lendingclub --skip-shap
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

Tune the recall and false-alarm tradeoff:

```powershell
python train.py --data data\application_train.csv --min-precision 0.30
```

Quick smoke test without SHAP:

```powershell
python train.py --data tests\smoke_data.csv --target-col loan_status --output-dir outputs\smoke --skip-shap
```

## Streamlit App

Train first, then run:

```powershell
streamlit run app.py
```

The app loads `outputs/smoke` by default. Change the output directory in the sidebar after training on a full dataset.

## Outputs

The script writes:

- `outputs/metrics.csv`: model leaderboard with recall, precision, AUC, average precision, threshold, and confusion matrix
- `outputs/models/*.pkl`: trained preprocessing and model pipelines
- `outputs/shap/<model>/feature_importance.csv`: top SHAP drivers
- `outputs/shap/<model>/summary_beeswarm.png`: SHAP directionality plot
- `outputs/shap/<model>/summary_bar.png`: global SHAP importance plot

Generated data, trained models, and output artifacts are ignored by Git.

## Exploratory Analysis

The repository also contains the earlier EDA notebook and charts from the existing GitHub history:

- `Loan_risk_detectin/Notebook/lendingclub_eda.ipynb`
- `Loan_risk_detectin/images/loan_status_distribution.png`
- `Loan_risk_detectin/images/loan_amount_distribution_sample.png`
- `Loan_risk_detectin/images/interest_rate_by_target_sample.png`
- `Loan_risk_detectin/images/default_rate_by_grade.png`

## Repository Structure

```text
.
|-- app.py
|-- train.py
|-- requirements.txt
|-- src/
|   `-- loan_default_risk/
|-- Loan_risk_detectin/
|   |-- Notebook/
|   `-- images/
|-- data/       # ignored
`-- outputs/    # ignored
```

## Why Recall First?

For credit risk screening, a false negative means an applicant who later defaults was approved as safe. Recall is prioritized to catch as many defaults as possible while preserving a minimum precision floor. Increase `--min-precision` when operational review capacity is limited.

## Explainability

SHAP reports are generated for the best recall-oriented model so model drivers can be reviewed by risk, compliance, and audit teams. Keep the generated feature importance CSV and plots with each model run to support model governance documentation.

## License

This project is developed for academic purposes.
