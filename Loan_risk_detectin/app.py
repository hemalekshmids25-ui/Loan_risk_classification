from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_OUTPUT_DIR = Path("outputs/lendingclub")

GRADE_TO_SUB_GRADE = {
    "A": "A1",
    "B": "B1",
    "C": "C1",
    "D": "D1",
    "E": "E1",
    "F": "F1",
    "G": "G1",
}

DEFAULT_APPLICANT = {
    "loan_amnt": 15000.0,
    "funded_amnt": 15000.0,
    "term": "36 months",
    "int_rate": 13.0,
    "installment": 505.0,
    "grade": "C",
    "sub_grade": "C1",
    "emp_length": "10+ years",
    "home_ownership": "MORTGAGE",
    "annual_inc": 65000.0,
    "verification_status": "Source Verified",
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "dti": 18.0,
    "delinq_2yrs": 0.0,
    "earliest_cr_line": "Jan-2005",
    "inq_last_6mths": 0.0,
    "mths_since_last_delinq": 34.0,
    "mths_since_last_record": 73.0,
    "open_acc": 11.0,
    "pub_rec": 0.0,
    "revol_bal": 12000.0,
    "revol_util": 50.0,
    "total_acc": 24.0,
    "initial_list_status": "w",
    "collections_12_mths_ex_med": 0.0,
    "application_type": "Individual",
    "acc_now_delinq": 0.0,
    "tot_coll_amt": 0.0,
    "tot_cur_bal": 140000.0,
    "total_rev_hi_lim": 30000.0,
    "acc_open_past_24mths": 4.0,
    "avg_cur_bal": 12000.0,
    "bc_open_to_buy": 7000.0,
    "bc_util": 55.0,
    "chargeoff_within_12_mths": 0.0,
    "delinq_amnt": 0.0,
    "mort_acc": 1.0,
    "pub_rec_bankruptcies": 0.0,
    "tax_liens": 0.0,
}


st.set_page_config(page_title="Loan Risk Checker", layout="wide")


@st.cache_data
def load_metrics(output_dir: str) -> pd.DataFrame:
    metrics_path = Path(output_dir) / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)


@st.cache_resource
def load_model(model_path: str):
    with Path(model_path).open("rb") as model_file:
        return pickle.load(model_file)


def model_path_for(output_dir: Path, model_name: str) -> Path:
    return output_dir / "models" / f"{model_name}.pkl"


def get_model_columns(pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    columns: list[str] = []
    for _, _, transformer_columns in preprocessor.transformers_:
        columns.extend(list(transformer_columns))
    return columns


def estimate_installment(loan_amount: float, annual_rate: float, term: str) -> float:
    months = 60 if "60" in term else 36
    monthly_rate = annual_rate / 100 / 12
    if monthly_rate <= 0:
        return loan_amount / months
    return loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -months)


def build_applicant_input(model_columns: list[str]) -> pd.DataFrame:
    st.subheader("Loan Check")

    loan_col, borrower_col, credit_col = st.columns(3)
    with loan_col:
        loan_amnt = st.number_input("Loan amount", min_value=500.0, max_value=40000.0, value=15000.0, step=500.0)
        term = st.selectbox("Term", ["36 months", "60 months"])
        int_rate = st.slider("Interest rate (%)", min_value=5.0, max_value=31.0, value=13.0, step=0.1)
        purpose = st.selectbox(
            "Purpose",
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "major_purchase",
                "small_business",
                "medical",
                "car",
                "other",
            ],
        )

    with borrower_col:
        annual_inc = st.number_input("Annual income", min_value=0.0, max_value=1_000_000.0, value=65000.0, step=1000.0)
        dti = st.slider("Debt-to-income ratio", min_value=0.0, max_value=60.0, value=18.0, step=0.5)
        emp_length = st.selectbox(
            "Employment length",
            ["10+ years", "5 years", "3 years", "1 year", "< 1 year", "n/a"],
        )
        home_ownership = st.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN", "ANY", "OTHER"])

    with credit_col:
        grade = st.selectbox("Credit grade", ["A", "B", "C", "D", "E", "F", "G"], index=2)
        verification_status = st.selectbox("Income verification", ["Source Verified", "Verified", "Not Verified"])
        revol_util = st.slider("Revolving utilization (%)", min_value=0.0, max_value=160.0, value=50.0, step=1.0)
        open_acc = st.number_input("Open accounts", min_value=0, max_value=100, value=11, step=1)

    with st.expander("Advanced credit details"):
        adv_left, adv_right, adv_extra = st.columns(3)
        with adv_left:
            delinq_2yrs = st.number_input("Delinquencies in 2 years", min_value=0, max_value=50, value=0, step=1)
            inq_last_6mths = st.number_input("Inquiries last 6 months", min_value=0, max_value=20, value=0, step=1)
            pub_rec = st.number_input("Public records", min_value=0, max_value=30, value=0, step=1)
        with adv_right:
            total_acc = st.number_input("Total accounts", min_value=0, max_value=200, value=24, step=1)
            revol_bal = st.number_input("Revolving balance", min_value=0.0, max_value=2_000_000.0, value=12000.0, step=500.0)
            total_rev_hi_lim = st.number_input("Revolving credit limit", min_value=0.0, max_value=2_000_000.0, value=30000.0, step=500.0)
        with adv_extra:
            mort_acc = st.number_input("Mortgage accounts", min_value=0, max_value=50, value=1, step=1)
            pub_rec_bankruptcies = st.number_input("Bankruptcies", min_value=0, max_value=20, value=0, step=1)
            addr_state = st.text_input("State", value="CA", max_chars=2).upper()

    installment = estimate_installment(loan_amnt, int_rate, term)
    applicant = DEFAULT_APPLICANT.copy()
    applicant.update(
        {
            "loan_amnt": loan_amnt,
            "funded_amnt": loan_amnt,
            "term": term,
            "int_rate": int_rate,
            "installment": round(installment, 2),
            "purpose": purpose,
            "annual_inc": annual_inc,
            "dti": dti,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "grade": grade,
            "sub_grade": GRADE_TO_SUB_GRADE[grade],
            "verification_status": verification_status,
            "revol_util": revol_util,
            "open_acc": float(open_acc),
            "delinq_2yrs": float(delinq_2yrs),
            "inq_last_6mths": float(inq_last_6mths),
            "pub_rec": float(pub_rec),
            "total_acc": float(total_acc),
            "revol_bal": revol_bal,
            "total_rev_hi_lim": total_rev_hi_lim,
            "mort_acc": float(mort_acc),
            "pub_rec_bankruptcies": float(pub_rec_bankruptcies),
            "addr_state": addr_state or "CA",
        }
    )

    return pd.DataFrame([{column: applicant.get(column, 0) for column in model_columns}])


def show_decision(default_probability: float, threshold: float) -> None:
    predicted_default = default_probability >= threshold

    result_col, detail_col = st.columns([1, 2])
    with result_col:
        st.metric("Default Probability", f"{default_probability:.2%}")
        st.metric("Decision Threshold", f"{threshold:.3f}")
    with detail_col:
        if predicted_default:
            st.error("High default risk. Send this application for manual credit review.")
        else:
            st.success("Below the current default-risk threshold.")

        if default_probability >= 0.30:
            band = "Very high"
        elif default_probability >= 0.15:
            band = "High"
        elif default_probability >= 0.05:
            band = "Moderate"
        else:
            band = "Low"
        st.write(f"Risk band: **{band}**")


def main() -> None:
    st.title("Loan Default Risk Checker")
    st.caption("Score a Lending Club-style loan application using the trained default-risk model.")

    with st.sidebar:
        st.header("Model Settings")
        output_dir_text = st.text_input("Output directory", value=str(DEFAULT_OUTPUT_DIR))
        output_dir = Path(output_dir_text)

    metrics = load_metrics(str(output_dir))
    if metrics.empty:
        st.error("No metrics.csv found. Train a model first, then refresh this app.")
        st.code(
            "python train.py --data data\\loan.csv --target-col loan_status "
            "--sample-rows 100000 --output-dir outputs\\lendingclub --skip-shap",
            language="powershell",
        )
        return

    best_model = str(metrics.iloc[0]["model"])
    model_names = metrics["model"].tolist()
    selected_model = st.sidebar.selectbox("Classifier", model_names, index=model_names.index(best_model))
    selected_row = metrics[metrics["model"] == selected_model].iloc[0]
    threshold = float(selected_row["threshold"])
    selected_path = model_path_for(output_dir, selected_model)

    if not selected_path.exists():
        st.error(f"Model file not found: {selected_path}")
        return

    pipeline = load_model(str(selected_path))
    model_columns = get_model_columns(pipeline)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Model", selected_model)
    metric_cols[1].metric("Recall", f"{selected_row['recall_default']:.3f}")
    metric_cols[2].metric("Precision", f"{selected_row['precision_default']:.3f}")
    metric_cols[3].metric("ROC AUC", f"{selected_row['roc_auc']:.3f}")
    metric_cols[4].metric("Threshold", f"{threshold:.3f}")

    with st.expander("Model leaderboard"):
        st.dataframe(metrics, use_container_width=True)

    applicant = build_applicant_input(model_columns)

    if st.button("Score Loan Application", type="primary"):
        default_probability = float(pipeline.predict_proba(applicant)[:, 1][0])
        show_decision(default_probability, threshold)

        with st.expander("Model input row"):
            st.dataframe(applicant, use_container_width=True)


if __name__ == "__main__":
    main()
