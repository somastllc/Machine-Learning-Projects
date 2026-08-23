# 🚗 Used Car Price Prediction Pipeline

An advanced Machine Learning regression project designed to estimate the market resale price (`Price`) of used vehicles using raw vehicle specifications, historical usage, engine power extraction, and market trends.

---

## 📌 Problem Statement & Overview

### The Problem
Resale vehicle valuation is notoriously volatile due to non-standardized raw seller listings. Features such as vehicle engine specs, mileage, age, accident history, title status, transmission type, and brand tier are often recorded as messy unstructured strings (e.g., `"300.0HP 3.0L V6 Cylinder Engine"`). Standard valuation tools struggle to accurately parse engine text or handle rare car model categories, leading to inconsistent vehicle appraisals.

### The Machine Learning Solution
This project creates an end-to-end data cleaning, feature extraction, and predictive modeling pipeline. It uses regular expression (Regex) parsing to extract engine horsepower and liter capacity, performs quantile-based outlier capping, computes annual mileage degradation metrics, and trains both **Linear Regression** and **Random Forest** regressors for pricing estimation.

---

## ⚙️ Pipeline Architecture

```
                 ┌──────────────────────────┐
                 │  dataset/used_cars.csv   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    preprocessing.py      │
                 │ • Regex clean Price &    │
                 │   Mileage                │
                 │ • Regex extract HP & L   │
                 │   from Engine String     │
                 │ • 99th Percentile Price  │
                 │   Outlier Capping        │
                 │ • Compute CarAge &       │
                 │   MileagePerYear         │
                 │ • Group Rare Models      │
                 │ • One-Hot Encode Brands &│
                 │   Transmissions          │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │  clean_used_cars.csv     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │        train.py          │
                 │ • Train-Test Split (80/20│
                 │ • StandardScaler (LR)    │
                 │ • Linear Regression      │
                 │ • Random Forest          │
                 │ • Export Feature Columns │
                 └────────────┬─────────────┘
                              │
      ┌───────────────────────┴───────────────────────┐
      ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│         evaluate.py           │               │          predict.py           │
│ • Metrics (MAE, RMSE, R²)     │               │ • Full Single Vehicle         │
│ • Actual vs Predicted Plot    │               │   Appraisal Inference         │
│ • Residual Plot               │               └───────────────────────────────┘
└───────────────────────────────┘
```

---

## 📊 Dataset & Feature Engineering

### Raw Data Extraction & Cleaning
- **Price & Mileage Parsing**: Strips dollar signs, commas, and textual suffixes using regex expressions.
- **Engine Text Mining**:
  - `Horsepower`: Extracted via `r"(\d+(?:\.\d+)?)HP"` (imputed with median if missing).
  - `Engine_Liters`: Extracted via `r"(\d+(?:\.\d+)?)L"` (imputed with median if missing).
- **Outlier Capping**: Caps extreme luxury supercar price spikes at the **99th percentile**.

### Key Derived & Engineered Features

| Feature | Source / Formula | Description |
| :--- | :--- | :--- |
| `CarAge` | $2026 - \text{model\_year}$ | Vehicle age in years |
| `MileagePerYear` | $\frac{\text{Mileage}}{\text{CarAge}}$ | Annual driving intensity index |
| `Horsepower` | Parsed from `engine` | Engine power output (HP) |
| `Engine_Liters` | Parsed from `engine` | Engine displacement size (Liters) |
| `Transmission_Type` | Categorized | Grouped into `Automatic`, `Manual`, `CVT`, or `Other` |
| `Model_Group` | Frequency thresholding | Models with $<5$ occurrences grouped as `"Other"` |
| `accident` | Categorical | Accident record (`None reported`, `At least 1 accident`, `Unknown`) |
| `clean_title` | Categorical | Title status (`Yes`, `Unknown`) |

---

## 🤖 Models Trained

1. **Linear Regression**
   - Uses `StandardScaler` to scale numerical features (`Mileage`, `Horsepower`, `Engine_Liters`, `CarAge`, `MileagePerYear`).
   - Serves as an interpretable baseline model.

2. **Random Forest Regressor**
   - Ensemble of 100 Decision Trees.
   - Accurately captures non-linear depreciation curves and brand premium interactions without requiring feature scaling.

---

## 📈 Evaluation & Diagnostics

Evaluated using standard regression metrics:
- **MAE (Mean Absolute Error)**: Average dollar prediction error.
- **RMSE (Root Mean Squared Error)**: Measures error variance and large deviation penalties.
- **$R^2$ Score**: Variance ratio explained by the model.

Visual outputs saved in `outputs/`:
- `actual_vs_predicted.png`
- `residuals.png`

---

## 💻 How to Run

### 1. Preprocess Raw Data
```bash
python preprocessing.py
```

### 2. Train Models
```bash
python train.py
```

### 3. Evaluate Performance & Generate Plots
```bash
python evaluate.py
```

### 4. Run Single Vehicle Appraisal
```bash
python predict.py
```

---

## 📁 Directory Structure

```
Car-price-prediction/
├── dataset/
│   ├── used_cars.csv            # Raw dataset
│   └── clean_used_cars.csv      # Cleaned & encoded dataset
├── models/                      # Trained model binaries & feature metadata
│   ├── common_models.pkl
│   ├── feature_columns.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── outputs/                     # Diagnostic visual outputs
│   ├── actual_vs_predicted.png
│   └── residuals.png
├── preprocessing.py             # Regex cleaning & feature engineering
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Prediction script
├── requirements.txt             # Requirements
└── README.md                    # Project documentation
```
