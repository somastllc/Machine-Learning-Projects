# 🎓 Student Dropout Early Warning & Risk Classification

An end-to-end Machine Learning binary classification project that predicts student academic attrition and dropout risk (`Dropout`: 0 = Retained / Continuing, 1 = Dropped Out) using demographic, academic performance, behavioral, and socio-economic attributes.

---

## 📌 Problem Statement & Overview

### The Problem
Student attrition in higher education leads to substantial academic, financial, and personal setbacks for students, as well as institutional loss for educational organizations. Identifying students vulnerable to academic failure or dropout at an early stage allows academic counselors, advisors, and faculty to deliver timely, targeted interventions (such as supplemental tutoring, mental health counseling, or financial aid assistance) before withdrawal occurs.

### The Machine Learning Solution
This project establishes a standardized, end-to-end Machine Learning pipeline comparing **Logistic Regression** and **Random Forest** classifiers. Features are cleaned by removing unique identifiers, imputing missing values (median for numerical indicators and mode for categorical features), mapping binary variables (`Gender`, `Internet_Access`, `Part_Time_Job`, `Scholarship`), one-hot encoding multi-categorical factors (`Semester`, `Department`, `Parental_Education`), and standardizing numerical metrics.

---

## ⚙️ Pipeline Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │ dataset/student_dropout_dataset_v3.csv       │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │               preprocessing.py               │
                 │ • Drop Duplicates & Student_ID Identifier   │
                 │ • Impute Missing (Median for Numeric,        │
                 │   Mode for Categorical)                      │
                 │ • Binary Map (Gender, Internet, Job, Schol.) │
                 │ • One-Hot Encode (Semester, Dept, Parental)  │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │ dataset/clean_student_dropout.csv            │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │                   train.py                   │
                 │ • Stratified Train/Test Split (80/20)        │
                 │ • StandardScaler on Continuous Features      │
                 │ • Train Balanced LR & Random Forest          │
                 │ • Export *.pkl Models, Scaler & Columns      │
                 └──────────────────────┬───────────────────────┘
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        ▼                                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│          evaluate.py          │               │          predict.py           │
│ • Precision, Recall, F1, ROC  │               │ • Student Risk Probability    │
│ • Confusion Matrix Plots      │               │   & Decision Inference        │
└───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

The dataset contains 10,000 student records characterized by demographic, academic, and socio-economic dimensions.

### Input Features

| Feature | Type | Description | Values / Encoding |
| :--- | :--- | :--- | :--- |
| `Age` | Numerical | Student age | Continuous (Years) |
| `Gender` | Binary | Student gender | `0` (Female), `1` (Male) |
| `Family_Income` | Numerical | Annual household income | Imputed with median |
| `Internet_Access` | Binary | Home internet availability | `0` (No), `1` (Yes) |
| `Study_Hours_per_Day` | Numerical | Daily study duration | Hours (Imputed with median) |
| `Attendance_Rate` | Numerical | Course attendance rate | Percentage ($0 - 100\%$) |
| `Assignment_Delay_Days`| Numerical | Average delay in homework submissions | Days |
| `Travel_Time_Minutes` | Numerical | Daily commute time to campus | Minutes |
| `Part_Time_Job` | Binary | Employment status during studies | `0` (No), `1` (Yes) |
| `Scholarship` | Binary | Scholarship beneficiary status | `0` (No), `1` (Yes) |
| `Stress_Index` | Numerical | Self-reported stress level index | Scale (Imputed with median) |
| `GPA` | Numerical | Overall grade point average | Scale ($0.0 - 4.0$) |
| `Semester_GPA` | Numerical | Current semester grade point average | Scale ($0.0 - 4.0$) |
| `CGPA` | Numerical | Cumulative grade point average | Scale ($0.0 - 4.0$) |
| `Semester` | Categorical | Academic progression stage | `Year 1`, `Year 2`, `Year 3`, `Year 4` (One-Hot) |
| `Department` | Categorical | Field of study / major | `Arts`, `Business`, `CS`, `Engineering` (One-Hot) |
| `Parental_Education` | Categorical | Highest educational level of parents | `High School`, `Bachelor`, `Master`, `PhD` (One-Hot) |

### Target Variable
- **`Dropout`**: 
  - `0` = Enrolled / Graduated (Retained, $76.46\%$)
  - `1` = Dropped Out (At Risk, $23.54\%$)

---

## 🤖 Models Trained

1. **Logistic Regression**
   - Fitted with `class_weight="balanced"` to handle class distribution skew.
   - Continuous numerical features scaled using `StandardScaler`.
   - Offers interpretable odds ratios and fast inference for early warning screening.

2. **Random Forest Classifier**
   - Ensemble of 100 decision trees (`n_estimators=100`, `random_state=42`, `class_weight="balanced"`).
   - Models complex non-linear interactions across GPA trends, stress indices, and attendance metrics without requiring feature scaling.

---

## 📈 Evaluation & Diagnostics

Both models were evaluated on an unseen 20% holdout test set with stratified class distribution (`test_size=0.2`, `random_state=42`):

| Evaluation Metric | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| **Accuracy** | 74.40% | **79.55%** |
| **Precision** | 47.30% | **56.30%** |
| **Recall (Sensitivity)** | **76.22%** | 58.81% |
| **F1-Score** | **0.5837** | 0.5753 |
| **ROC-AUC** | **0.8204** | 0.8045 |

> [!TIP]
> **Clinical/Academic Trade-off**: In student early warning systems, **Recall** is critical: catching an at-risk student early (76.22% Recall via Logistic Regression) is far more important than a false alarm. Random Forest provides higher precision and overall accuracy (79.55%) when intervention resources are strictly constrained.

### Confusion Matrix Plots (Saved in `outputs/`)
- `outputs/confusion_matrix_lr.png`: Diagnostic confusion matrix for Logistic Regression.
- `outputs/confusion_matrix_rf.png`: Diagnostic confusion matrix for Random Forest.

---

## 💻 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clean dataset, impute missing values, and encode categorical columns
python preprocessing.py

# 3. Train models, scale numerical features, and export *.pkl binaries
python train.py

# 4. Evaluate metrics & generate confusion matrix plots
python evaluate.py

# 5. Run inference on a sample student profile
python predict.py
```

---

## 📁 Directory Structure

```
student_dropout_classification/
├── dataset/
│   ├── student_dropout_dataset_v3.csv   # Raw dataset (10,000 records)
│   └── clean_student_dropout.csv        # Preprocessed, imputed & encoded dataset
├── models/                              # Saved binaries (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scale_columns.pkl
│   └── scaler.pkl
├── outputs/                             # Visual diagnostic plots
│   ├── confusion_matrix_lr.png
│   └── confusion_matrix_rf.png
├── preprocessing.py                     # Cleaning, missing imputation & encoding script
├── train.py                             # Stratified splitting & model training script
├── evaluate.py                          # Performance evaluation & confusion matrix generator
├── predict.py                           # Single-student prediction & probability script
├── requirements.txt                     # Python package dependencies
└── README.md                            # Comprehensive project documentation
```
