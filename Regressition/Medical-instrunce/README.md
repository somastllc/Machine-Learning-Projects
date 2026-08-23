# 🏥 Medical Insurance Cost Prediction

A Machine Learning regression project that predicts individual medical insurance costs (`charges`) based on demographic features, body metrics, lifestyle factors, and interaction terms.

---

## 📌 Problem Statement & Overview

### The Problem
Health insurance providers need precise financial modeling to price insurance premiums accurately based on risk. Traditional pricing models struggle to capture non-linear cost multipliers—such as how smoking interacts exponentially with Body Mass Index (BMI). Without accurate cost prediction models, insurance plans risk underpricing high-risk individuals or overpricing healthy policyholders.

### The Machine Learning Solution
This project develops a predictive regression workflow to estimate individual medical charges. It incorporates binary encoding, one-hot encoding for geographic regions, and engineered interaction terms (`bmi_smoker`) to reflect lifestyle risks effectively.

---

## ⚙️ Pipeline Architecture

```
                 ┌──────────────────────────┐
                 │  dataset/insurance.csv   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    preprocessing.py      │
                 │ • Drop Duplicates        │
                 │ • Binary Map sex/smoker  │
                 │ • Create bmi_smoker      │
                 │ • One-Hot Encode region  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   clean_insurance.csv    │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │        train.py          │
                 │ • 80/20 Train-Test Split │
                 │ • Linear Regression      │
                 │ • Save Model Binary      │
                 └────────────┬─────────────┘
                              │
      ┌───────────────────────┴───────────────────────┐
      ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│         evaluate.py           │               │          predict.py           │
│ • Metrics (MAE, RMSE, R²)     │               │ • Single Policyholder         │
│ • Scatter & Residual Plots    │               │   Inference                   │
└───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

The dataset includes demographic and healthcare information for insurance policyholders.

### Input Features

| Feature | Type | Range / Description |
| :--- | :--- | :--- |
| `age` | Numerical | Age of the primary beneficiary (years) |
| `sex` | Binary | `0` (Female), `1` (Male) |
| `bmi` | Numerical | Body Mass Index ($\text{kg/m}^2$) |
| `children` | Numerical | Number of dependent children / dependents covered |
| `smoker` | Binary | `0` (Non-smoker), `1` (Smoker) |
| `bmi_smoker` | Numerical (Engineered) | Interaction feature ($\text{bmi} \times \text{smoker}$) |
| `region_northwest` | Binary (One-Hot) | Policyholder resides in Northwest US |
| `region_southeast` | Binary (One-Hot) | Policyholder resides in Southeast US |
| `region_southwest` | Binary (One-Hot) | Policyholder resides in Southwest US |

### Target Variable
- **`charges`**: Individual medical costs billed by health insurance (USD $).

---

## 🤖 Model & Feature Engineering

- **Linear Regression**: Fits a linear decision boundary across demographic, lifestyle, and geographical features.
- **`bmi_smoker` Feature Engineering**: Captures the major cost jump for high-BMI smokers compared to non-smokers with high BMI.

---

## 📈 Evaluation & Diagnostics

Evaluated using standard regression metrics:
- **MAE (Mean Absolute Error)**: Average dollar error in predictions.
- **RMSE (Root Mean Squared Error)**: Measures variance of prediction errors.
- **$R^2$ Score**: Goodness-of-fit measurement.

Visual outputs in `outputs/`:
- `actual_vs_predicted.png`
- `residuals.png`

---

## 💻 How to Run

### 1. Preprocess & Clean Data
```bash
python preprocessing.py
```

### 2. Train Model
```bash
python train.py
```

### 3. Evaluate Performance
```bash
python evaluate.py
```

### 4. Run Sample Prediction
```bash
python predict.py
```

---

## 📁 Directory Structure

```
Medical-instrunce/
├── dataset/
│   ├── insurance.csv            # Raw dataset
│   └── clean_insurance.csv      # Processed dataset
├── models/
│   └── linear_regression_model.pkl  # Trained model artifact (Git ignored)
├── outputs/
│   ├── actual_vs_predicted.png
│   └── residuals.png
├── preprocessing.py             # Preprocessing & encoding script
├── train.py                     # Training script
├── evaluate.py                  # Evaluation & plotting script
├── predict.py                   # Prediction script
├── requirements.txt             # Requirements
└── README.md                    # Documentation
```
