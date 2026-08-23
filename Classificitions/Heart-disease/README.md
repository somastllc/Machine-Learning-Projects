# 🩺 Heart Disease Clinical Risk Classification

A Machine Learning clinical classification project that predicts heart disease presence (`target`: 0 = Normal / Low Risk, 1 = Heart Disease Present) using patient vitals, ECG readings, and diagnostic attributes.

---

## 📌 Problem Statement & Overview

### The Problem
Cardiovascular diseases (CVDs) are the leading cause of death globally. Early clinical risk assessment allows physicians to intervene before critical events (e.g., myocardial infarction) occur. Medical diagnostic tools need to combine multiple physical indicators—such as resting blood pressure, cholesterol levels, chest pain type, and peak exercise ST depression—into a reliable risk probability score.

### The Machine Learning Solution
This project implements a binary classification model comparing **Logistic Regression** and **Random Forest** classifiers. Clinical features are preprocessed by removing duplicate records, one-hot encoding multi-categorical diagnostic indicators (`cp`, `restecg`, `slope`, `thal`), and standardizing numerical vitals.

---

## ⚙️ Pipeline Architecture

```
                 ┌──────────────────────────┐
                 │    dataset/heart.csv     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    preprocessing.py      │
                 │ • Drop Duplicate Records │
                 │ • One-Hot Encode cp,     │
                 │   restecg, slope, thal   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     clean_heart.csv      │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │        train.py          │
                 │ • Stratified Train/Test  │
                 │ • StandardScaler (vitals)│
                 │ • Train LR & RF Models   │
                 └────────────┬─────────────┘
                              │
      ┌───────────────────────┴───────────────────────┐
      ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│         evaluate.py           │               │          predict.py           │
│ • Precision, Recall, F1       │               │ • Patient Clinical Sample     │
│ • Confusion Matrix Plots      │               │   Inference                   │
└───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

### Input Features

| Feature | Description | Range / Categories |
| :--- | :--- | :--- |
| `age` | Patient age | Years |
| `sex` | Patient sex | `0` (Female), `1` (Male) |
| `cp` | Chest pain type | Typical angina, atypical angina, non-anginal, asymptomatic (One-Hot) |
| `trestbps` | Resting blood pressure | mm Hg on admission |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar $> 120$ mg/dl | `1` (True), `0` (False) |
| `restecg` | Resting electrocardiographic results | Normal, ST-T wave abnormality, LV hypertrophy (One-Hot) |
| `thalach` | Maximum heart rate achieved | Beats per minute |
| `exang` | Exercise induced angina | `1` (Yes), `0` (No) |
| `oldpeak` | ST depression induced by exercise | Depression relative to rest |
| `slope` | Slope of peak exercise ST segment | Upsloping, flat, downsloping (One-Hot) |
| `ca` | Number of major vessels colored by fluoroscopy | 0 to 4 |
| `thal` | Thalassemia status | Normal, fixed defect, reversible defect (One-Hot) |

### Target Variable
- **`target`**: `0` = No heart disease, `1` = Heart disease present.

---

## 🤖 Models Trained

1. **Logistic Regression**
   - Standardized numerical vitals (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) with `StandardScaler`.
   - Highly interpretable baseline model for medical diagnosis.

2. **Random Forest Classifier**
   - Ensemble of 100 decision trees.
   - Captures complex diagnostic feature interactions.

---

## 📈 Evaluation & Diagnostics

Evaluated using clinical classification metrics:
- **Accuracy, Precision, Recall (Sensitivity), F1-Score**
- **Confusion Matrix Plots** saved in `outputs/`:
  - `confusion_matrix_lr.png`
  - `confusion_matrix_rf.png`

---

## 💻 How to Run

```bash
# 1. Preprocess dataset & encode categorical attributes
python preprocessing.py

# 2. Train classifiers & export model binaries
python train.py

# 3. Evaluate performance metrics & generate confusion matrix plots
python evaluate.py

# 4. Run sample patient diagnosis prediction
python predict.py
```

---

## 📁 Directory Structure

```
Heart-disease/
├── dataset/
│   ├── heart.csv                # Raw dataset
│   └── clean_heart.csv          # Cleaned & encoded dataset
├── models/                      # Saved binaries (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── outputs/                     # Visual diagnostic plots
│   ├── confusion_matrix_lr.png
│   └── confusion_matrix_rf.png
├── preprocessing.py             # Preprocessing & encoding script
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Patient risk prediction script
├── requirements.txt             # Requirements
└── README.md                    # Project documentation
```
