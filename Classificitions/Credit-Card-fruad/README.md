# 💳 Credit Card Fraud Detection

A Machine Learning binary classification project that detects fraudulent credit card transactions (`Class`: 0 = Legitimate, 1 = Fraud) in highly imbalanced financial transaction data.

---

## 📌 Problem Statement & Overview

### The Problem
Credit card fraud costs financial institutions and consumers billions of dollars annually. Fraudulent transactions account for a minute fraction of total transaction volume ($< 0.2\%$). Detecting fraud in highly imbalanced datasets requires specialized feature scaling, stratified data splitting, and evaluation metrics focused on precision and recall rather than raw accuracy.

### The Machine Learning Solution
This project implements a binary classification workflow evaluating **Logistic Regression** and **Random Forest** models. Transaction `Amount` and `Time` are normalized using `StandardScaler`, while preserving 28 principal component features (`V1` to `V28`) obtained via PCA for privacy and dimensionality reduction.

---

## ⚙️ Pipeline Architecture

```
                 ┌──────────────────────────┐
                 │ dataset/creditcard.csv   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    preprocessing.py      │
                 │ • Remove Duplicates      │
                 │ • Verify Class Imbalance │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   clean_creditcard.csv   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │        train.py          │
                 │ • Stratified Train/Test  │
                 │ • StandardScaler (Time & │
                 │   Amount)                │
                 │ • Train LR & RF Models   │
                 └────────────┬─────────────┘
                              │
      ┌───────────────────────┴───────────────────────┐
      ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│         evaluate.py           │               │          predict.py           │
│ • Precision, Recall, F1       │               │ • Single Transaction          │
│ • Confusion Matrix Plots      │               │   Inference                   │
└───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

The dataset contains credit card transactions made by European cardholders.

### Input Features
- **`Time`**: Seconds elapsed between each transaction and the first transaction in the dataset.
- **`Amount`**: Transaction monetary value.
- **`V1` to `V28`**: Anonymized numerical principal components resulting from a PCA transformation.

### Target Variable
- **`Class`**: `0` = Legitimate Transaction, `1` = Fraudulent Transaction.

---

## 🤖 Models Trained

1. **Logistic Regression**
   - Scaled `Time` and `Amount` using `StandardScaler`.
   - Fast, linear baseline classifier.

2. **Random Forest Classifier**
   - Ensemble classifier trained on full feature vector.
   - Robust against outliers and complex non-linear fraud signatures.

---

## 📈 Evaluation & Diagnostics

In highly imbalanced datasets, evaluation focuses on:
- **Precision**: Proportion of flagged transactions that were actually fraudulent.
- **Recall**: Proportion of total actual frauds correctly detected.
- **F1-Score & Confusion Matrix Plots** saved in `outputs/`:
  - `confusion_matrix_lr.png`
  - `confusion_matrix_rf.png`

---

## 💻 How to Run

```bash
# 1. Clean dataset & handle duplicates
python preprocessing.py

# 2. Train Logistic Regression & Random Forest classifiers
python train.py

# 3. Evaluate performance & generate confusion matrices
python evaluate.py

# 4. Run sample transaction fraud prediction
python predict.py
```

---

## 📁 Directory Structure

```
Credit-Card-fruad/
├── dataset/
│   ├── creditcard.csv           # Raw dataset
│   └── clean_creditcard.csv     # Cleaned dataset
├── models/                      # Saved binaries (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── outputs/                     # Visual diagnostic plots
│   ├── confusion_matrix_lr.png
│   └── confusion_matrix_rf.png
├── preprocessing.py             # Preprocessing script
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Single transaction prediction script
├── requirements.txt             # Requirements
└── README.md                    # Project documentation
```
