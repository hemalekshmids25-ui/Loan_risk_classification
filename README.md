#  Loan Default Risk Classification for Banking

A machine learning and predictive analytics project focused on identifying whether a loan applicant is likely to default on a loan using financial and credit-related information from the LendingClub dataset.

This project combines:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Models
- Explainable AI (SHAP)
- Streamlit Deployment

to build an intelligent and interpretable loan risk prediction system for banking applications.

---

#  Project Overview

Loan default prediction is one of the most important applications of machine learning in the banking and finance sector. Financial institutions need reliable systems to reduce risky loan approvals while maintaining business growth.

This project analyzes customer financial behavior and predicts loan default risk using classification models trained on historical lending data.

The project focuses on:
- Credit risk assessment
- Imbalanced classification
- Recall optimization
- Explainable AI
- End-to-end deployment

---

# Problem Statement

Banks and financial institutions lose significant amounts of money due to loan defaults. Manual risk assessment methods are often inconsistent and inefficient.

The objective of this project is to build a machine learning system that:
- Predicts whether a borrower is likely to default
- Helps banks make safer loan approval decisions
- Minimizes financial risk
- Provides interpretable predictions using SHAP

---

#  Objectives

- Perform complete Exploratory Data Analysis (EDA)
- Understand financial risk patterns
- Handle missing values and class imbalance
- Train multiple machine learning classification models
- Optimize Recall Score to reduce risky approvals
- Compare model performances
- Deploy the model using Streamlit
- Provide explainability using SHAP

---

# Team Members

| Team Member | Contribution |

| Member 1 | Data Collection & EDA |
| Member 2 | Model Building & Evaluation |
| Member 3 | Deployment & Documentation |

---

#  Course Information
- **Project Type:** Predictive Analytics
- **Academic Year:** 2026

---

#  Dataset Information

## Dataset Used
- LendingClub Loan Dataset

## Dataset Source
- Kaggle Public Dataset

## Dataset Description

The dataset contains loan application details, borrower financial information, repayment history, and credit-related features used to predict loan default risk.

---

#  Important Features

The dataset includes:

- Loan Amount
- Interest Rate
- Annual Income
- Debt-to-Income Ratio
- Loan Grade
- Home Ownership
- Employment Length
- Verification Status
- Revolving Credit Utilization
- Loan Purpose
- Credit History

---

#  Target Variable

| Value | Meaning |
|------|------|
| 0 | Non-Default |
| 1 | Default |

---

# Class Imbalance Problem

The dataset is highly imbalanced because most customers successfully repay loans.

To address this issue, the project uses:
- Class-weighted learning
- Recall-focused evaluation
- Ensemble methods
- Stratified splitting techniques

---

#  Data Science Life Cycle

---

## 1️ Problem Understanding

Understanding the business impact of loan defaults and the importance of accurate credit risk prediction.

---

## 2️ Data Collection

Collected LendingClub loan dataset containing borrower financial information and loan repayment details.

---

## 3️ Data Cleaning

Performed:
- Missing value analysis
- Data validation
- Type conversion
- Percentage-based missingness checking
- Invalid value handling

---

## 4️ Exploratory Data Analysis (EDA)

Detailed EDA was performed to understand:
- Loan default distribution
- Missing data patterns
- Numerical feature distributions
- Categorical feature analysis
- Default rates across borrower groups

---

#  EDA Highlights

### ✅ Loan Status Distribution
Analyzed the distribution of default and non-default loans.

### ✅ Missing Value Analysis
Identified columns with high missing percentages and evaluated data quality.

### ✅ Numerical Feature Analysis
Studied distributions of:
- Interest Rate
- Loan Amount
- Revolving Utilization
- Annual Income

### ✅ Categorical Analysis
Explored top categories for:
- Loan Grade
- Purpose
- Home Ownership
- Verification Status

### ✅ Default Rate by Groups
Analyzed default trends across:
- Loan Grades
- Loan Purposes
- Home Ownership Types
- Verification Status
- Application Types

---

# EDA Visualizations

## Loan Status Distribution
![Loan Status Distribution](images/loan_status_distribution.png)

## Loan Amount Distribution
![Loan Amount Distribution](images/loan_amount_distribution.png)

## Interest Rate Analysis
![Interest Rate Analysis](images/interest_rate_analysis.png)

## Default Rate by Grade
![Default Rate by Grade](images/default_rate_by_grade.png)

---

# ⚙️ Feature Engineering

Performed:
- Label Encoding
- One-Hot Encoding
- Numerical Scaling
- Feature Selection
- Derived financial feature creation

---

#  Machine Learning Models

The following models are implemented:

---

## ✅ Logistic Regression

A baseline classification model using class weighting for imbalance handling.

---

## ✅ Random Forest Classifier

An ensemble learning algorithm capable of handling nonlinear relationships and improving robustness.

---

## ✅ LightGBM Classifier

A high-performance gradient boosting framework optimized for structured financial datasets.

---

# Model Evaluation Metrics

Models are evaluated using:

- Accuracy
- Recall Score
- Precision
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

# Model Comparison

| Model | Accuracy | Recall | Precision | F1-Score |
|------|------|------|------|------|
| Logistic Regression | XX | XX | XX | XX |
| Random Forest | XX | XX | XX | XX |
| LightGBM | XX | XX | XX | XX |

> Replace XX with actual evaluation results after training.

---

#  Why Recall is Important

In banking systems, failing to identify risky applicants can lead to major financial losses.

Therefore, Recall Score is prioritized because it:
- Detects more risky applicants
- Reduces bad loan approvals
- Improves risk management

---

# Explainable AI using SHAP

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

SHAP helps:
- Interpret predictions transparently
- Identify important financial features
- Improve regulatory compliance
- Build trust in AI-based decisions

---

# 📷SHAP Explainability

![SHAP Visualization](images/shap_visualization.png)

---

# 🖥️ Streamlit Deployment

The final system is deployed using Streamlit to provide an interactive web interface.

Users can:
- Enter applicant details
- Upload loan information
- Receive instant predictions
- View prediction confidence
- Explore feature importance

---

#  Live Deployment

 **Streamlit App:**  
[Add Deployment Link Here]

---

# 📷 Deployment Screenshots

## Streamlit Home Page
![Streamlit Home](images/streamlit_home.png)

## Prediction Interface
![Prediction Interface](images/prediction_interface.png)

## Prediction Output
![Prediction Output](images/prediction_result.png)

---

# Project Architecture

![Architecture Diagram](images/project_architecture.png)

---

# Workflow Pipeline

```text
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Streamlit Deployment
```

---

# Technologies Used

| Category | Tools & Libraries |
|------|------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, LightGBM |
| Explainability | SHAP |
| Deployment | Streamlit |
| Version Control | Git & GitHub |

---

#  Repository Structure

```bash
Loan_risk_classification/
│
├── Loan_risk_detectin/
│   ├── Notebook/
│   │   └── lendingclub_eda.ipynb
│   │
│   ├── images/
│   │   ├── loan_status_distribution.png
│   │   ├── loan_amount_distribution.png
│   │   ├── default_rate_by_grade.png
│   │   ├── interest_rate_analysis.png
│   │   └── shap_visualization.png
│   │
│   └── app.py
│
├── individual_profiles/
│
├── README.md
├── requirements.txt
└── presentation.pptx
```

---

#  Installation & Setup

## Clone the Repository

```bash
git clone https://github.com/yourusername/Loan_risk_classification.git
```

---

## Navigate to Project Folder

```bash
cd Loan_risk_classification
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit App

```bash
streamlit run app.py
```

---

#  Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
streamlit
shap
joblib
```

---

# 🤝 GitHub Collaboration

This project follows professional GitHub collaboration practices including:
- Feature branching
- Pull Requests
- Commit tracking
- Shared development workflow
- Collaborative documentation

### Example Branches

```text
feature/eda
feature/modeling
feature/deployment
```

---

# 📄Submission Components Included

✅ Source Code  
✅ Jupyter Notebooks  
✅ Streamlit Deployment  
✅ README Documentation  
✅ PPT Presentation   
✅ requirements.txt  

---

#  Limitations

- Model performance depends on dataset quality
- Financial conditions may affect prediction reliability
- Limited external banking indicators

---

#  Future Improvements

- Deep Learning implementation
- Real-time prediction API
- Cloud deployment using AWS/Azure
- Advanced dashboard analytics
- Automated retraining pipeline

---

#  Acknowledgements

- Kaggle Datasets
- Scikit-learn Documentation
- Streamlit Documentation
- SHAP Documentation

---

#  License

This project is developed for academic purposes only.
