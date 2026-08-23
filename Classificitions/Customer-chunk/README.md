# 🏦 Bank Customer Churn Prediction

A Machine Learning classification project that predicts customer attrition (`Exited`: 0 = Retained, 1 = Churned) using demographic info, credit scores, account balances, and product engagement features.

---

## 📌 Problem Statement & Overview

### The Problem
Customer Acquisition Cost (CAC) in retail banking is 5x to 25x higher than retaining an existing customer. Banks experience significant revenue loss when customers close their accounts or transfer funds to competitors. Identifying at-risk customers early enables marketing and retention teams to offer targeted incentives before churn occurs.

### The Machine Learning Solution
This project builds a binary classification pipeline using **Logistic Regression** and **Random Forest** models to predict the probability of customer churn. The workflow drops uninformative identifiers (`CustomerId`, `Surname`), applies one-hot encoding to geographical and demographic features, scales numerical attributes, and uses stratified sampling to maintain class distribution.

---

## ⚙️ Pipeline Architecture

```
            ┌──────────────────────────────────────────────┐
            │ dataset/train.csv    dataset/test.csv        │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │             preprocessing.py                 │
            │ • Remove Duplicates & Drop ID Columns        │
            │ • One-Hot Encode Geography & Gender          │
            │ • Reindex Test Features to Match Train       │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │ clean_train.csv     clean_test.csv           │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │                   train.py                   │
            │ • Stratified 80/20 Train-Test Split          │
            │ • StandardScaler for Numerical Features      │
            │ • Train Logistic Regression & Random Forest  │
            └──────────────────────┬───────────────────────┘
                                   │
     ┌─────────────────────────────┴─────────────────────────────┐
     ▼                                                           ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│         evaluate.py           │               │          predict.py           │
│ • Confusion Matrix Plots      │               │ • Batch Inference on Test Set │
│ • Precision, Recall, F1, ROC  │               │ • Export test_predictions.csv │
└───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

### Input Features

| Feature | Type | Range / Description |
| :--- | :--- | :--- |
| `CreditScore` | Numerical | Creditworthiness score (300 - 850) |
| `Age` | Numerical | Customer age in years |
| `Tenure` | Numerical | Number of years as a bank client |
| `Balance` | Numerical | Current account balance |
| `NumOfProducts` | Numerical | Number of bank products used (1 to 4) |
| `HasCrCard` | Binary | `1` if customer owns a credit card, `0` otherwise |
| `IsActiveMember` | Binary | `1` if active member, `0` otherwise |
| `EstimatedSalary` | Numerical | Estimated annual salary |
| `Geography_*` | Binary (One-Hot) | France, Spain, Germany |
| `Gender_*` | Binary (One-Hot) | Female, Male |

### Target Variable
- **`Exited`**: `0` = Retained Customer, `1` = Churned Customer.

---

## 🤖 Models Trained

1. **Logistic Regression**
   - Scaled using `StandardScaler` (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`).
   - Provides linear decision boundaries and interpretable log-odds coefficients.

2. **Random Forest Classifier**
   - Ensemble of 200 decision trees (`n_estimators=200`, `random_state=42`).
   - Captures non-linear thresholds (e.g., age vs balance vs product count interactions).

---

## 📈 Evaluation & Outputs

Evaluated using classification metrics:
- **Accuracy, Precision, Recall, F1-Score**
- **Confusion Matrix Visualizations** saved in `outputs/`:
  - `confusion_matrix_lr.png`
  - `confusion_matrix_rf.png`
- **Batch Predictions**: Exports `outputs/test_predictions.csv` with predictions and churn probabilities.

---

## 💻 How to Run

```bash
# 1. Preprocess raw train/test data
python preprocessing.py

# 2. Train models and save artifacts
python train.py

# 3. Evaluate performance metrics & generate plots
python evaluate.py

# 4. Run prediction pipeline on unseen test data
python predict.py
```

---

## 📁 Directory Structure

```
Customer-chunk/
├── dataset/
│   ├── train.csv                # Raw training data
│   ├── test.csv                 # Raw test data
│   ├── clean_train.csv          # Preprocessed training dataset
│   └── clean_test.csv           # Preprocessed test dataset
├── models/                      # Saved binaries (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── outputs/                     # Visual diagnostic plots & batch outputs
│   ├── confusion_matrix_lr.png
│   ├── confusion_matrix_rf.png
│   └── test_predictions.csv
├── preprocessing.py             # Cleaning & encoding pipeline
├── train.py                     # Training script
├── evaluate.py                  # Evaluation & confusion matrix script
├── predict.py                   # Inference script
├── requirements.txt             # Requirements
└── README.md                    # Project documentation
```
