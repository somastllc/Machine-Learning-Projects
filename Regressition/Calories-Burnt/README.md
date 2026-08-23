# 🏋️ Calories Burnt Prediction Pipeline

An end-to-end Machine Learning regression project that predicts the number of calories burned during physical exercise based on physiological characteristics and real-time exercise metrics.

---

## 📌 Problem Statement & Overview

### The Problem
Traditional fitness devices and static calculators rely on simplified mathematical formulas that estimate calorie expenditure based solely on weight and exercise duration. These static formulas fail to account for critical dynamic factors such as **heart rate**, **body temperature**, **age**, **gender**, and **Body Mass Index (BMI)**. This often leads to inaccurate calorie tracking, which negatively impacts fitness tracking, athletic training, and nutritional planning.

### The Machine Learning Solution
This project builds a predictive Machine Learning workflow that combines user body metrics with real-time exercise sensor data. By evaluating multiple regression models (**Linear Regression** and **Random Forest Regressor**), the pipeline accurately models complex linear and non-linear physiological interactions to deliver personalized calorie burn estimates.

---

## ⚙️ Pipeline Architecture

```
                    ┌────────────────────────┐
                    │ exercise.csv           │
                    │ calories.csv           │
                    └───────────┬────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │   preprocessing.py       │
                   │ • Merge on User_ID       │
                   │ • Remove Duplicates      │
                   │ • Encode Gender (0/1)    │
                   │ • Compute BMI Feature    │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │    clean_calories.csv    │
                   └────────────┬─────────────┘
                                │
                                ▼
                   ┌──────────────────────────┐
                   │        train.py          │
                   │ • 80/20 Train-Test Split │
                   │ • StandardScaler (LR)    │
                   │ • Linear Regression      │
                   │ • Random Forest          │
                   └────────────┬─────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│         evaluate.py           │               │          predict.py           │
│ • Metrics (MAE, RMSE, R²)     │               │ • Inference on new workout    │
│ • Actual vs Predicted Plot    │               │   data                        │
│ • Residual Plot               │               └───────────────────────────────┘
└───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

The dataset is created by merging two datasets on `User_ID`:
- **`exercise.csv`**: Contains participant background info and real-time exercise body metrics.
- **`calories.csv`**: Contains ground truth burned calorie values.

### Input Features

| Feature | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `Gender` | Binary (Categorical) | `0` (Female), `1` (Male) | Gender classification |
| `Age` | Numerical | Years | Age of the individual |
| `Height` | Numerical | Centimeters (cm) | Height of the individual |
| `Weight` | Numerical | Kilograms (kg) | Body weight of the individual |
| `Duration` | Numerical | Minutes | Duration of the exercise session |
| `Heart_Rate` | Numerical | Beats Per Minute (BPM) | Average heart rate during exercise |
| `Body_Temp` | Numerical | Degrees Celsius (°C) | Core body temperature during exercise |
| `BMI` | Numerical (Engineered) | $\text{kg/m}^2$ | $\frac{\text{Weight (kg)}}{(\text{Height (m)})^2}$ |

### Target Variable
- **`Calories`**: Total energy burned during the workout session (in kilocalories).

---

## 🤖 Models Trained

1. **Linear Regression**
   - Baseline linear model.
   - Requires feature scaling using `StandardScaler` fitted on training numerical attributes (`Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`, `BMI`).

2. **Random Forest Regressor**
   - Ensembled tree-based algorithm (`n_estimators=100`, `random_state=42`).
   - Handles non-linear feature interaction (e.g., non-linear temperature spikes & heart rate thresholds vs duration).
   - Does not require feature scaling.

---

## 📈 Model Evaluation & Visualization

The pipeline evaluates performance on the 20% holdout test set using key regression metrics:
- **MAE (Mean Absolute Error)**: Average magnitude of prediction errors in calories.
- **RMSE (Root Mean Squared Error)**: Penalizes larger prediction deviations.
- **$R^2$ Score**: Proportion of variance in calorie burn explained by the model.

### Output Visualizations
Evaluation generates diagnostic plots in `outputs/`:
- **`actual_vs_predicted.png`**: Scatter plot assessing model alignment against ground truth.
- **`residuals.png`**: Residual analysis plot verifying unbiased error distribution.

---

## 💻 How to Run the Pipeline

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess & Clean Data
Merges raw datasets, handles missing values, encodes categorical variables, engineers `BMI`, and saves `clean_calories.csv`:
```bash
python preprocessing.py
```

### 3. Train Models
Splits dataset, scales numerical features, fits Linear Regression and Random Forest models, and exports model artifacts (`.pkl`) to `models/`:
```bash
python train.py
```

### 4. Evaluate Models & Generate Diagnostics
Loads models and outputs performance metrics alongside diagnostic scatter & residual plots:
```bash
python evaluate.py
```

### 5. Run Inference / Single Prediction
Pass custom exercise metrics (gender, age, height, weight, duration, heart rate, body temp) to predict calories burned:
```bash
python predict.py
```

---

## 📁 Directory Structure

```
Calories-Burnt/
├── dataset/
│   ├── calories.csv             # Raw calorie labels
│   ├── exercise.csv             # Raw exercise sensor & demographic data
│   └── clean_calories.csv       # Processed & feature-engineered dataset
├── models/                      # Trained model binaries (ignored by Git)
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── outputs/                     # Diagnostic plots
│   ├── actual_vs_predicted.png
│   └── residuals.png
├── preprocessing.py             # Data cleaning & feature engineering script
├── train.py                     # Model training pipeline
├── evaluate.py                  # Model evaluation & plotting script
├── predict.py                   # Single sample inference script
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```
