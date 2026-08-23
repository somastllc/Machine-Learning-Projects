# 🚀 Data Science & Machine Learning — Hands-On Projects

[![YouTube Watch Course](https://img.shields.io/badge/YouTube-Watch%20Full%20Course-red?style=for-the-badge&logo=youtube)](https://youtu.be/Ln9f16vI67U?si=JUgKclZXp7HDYFs4)

Welcome to the official hands-on repository accompanying the **Full Data Science & Machine Learning Course** on YouTube! 

> [!IMPORTANT]
> **Recommended Learning Path**: Simply copying or building projects without understanding the core theoretical and practical fundamentals of Data Science and Machine Learning will not give you a solid foundation. If you are new to ML/Data Science, **watch the full YouTube video course first** to master the concepts before diving into these 9 hands-on projects!
> 
> 📺 **Watch the Full Video Course on YouTube**: [Data Science & Machine Learning Full Course](https://youtu.be/Ln9f16vI67U?si=JUgKclZXp7HDYFs4)

This repository is designed to help students and developers master Data Science and Machine Learning step by step across three core pillars: **Regression**, **Classification**, and **Clustering** by building **9 real-world projects**.

---

## 🎓 About The Course

Instead of focusing solely on theoretical math or toy datasets, this course walks you through building production-ready Machine Learning workflows:
- Data cleaning, merging & exploratory data analysis (EDA)
- Feature engineering & regular expression (Regex) text parsing
- Training, evaluating, and tuning Machine Learning algorithms
- Saving model pipelines (`joblib`/`pickle`) and deploying real-world inference scripts

---

## 🗺️ 9-Project Learning Roadmap

The course repository is structured into **3 main sections**, featuring **3 practical projects per section** (9 projects in total):

```
Machine-Learning-Project/
├── 📈 Regressition/            # ✅ SECTION 1: REGRESSION (3/3 COMPLETED)
│   ├── Calories-Burnt/         # Project 1: Calorie expenditure estimation from body & exercise sensors
│   ├── Car-price-prediction/   # Project 2: Used car market valuation with engine text mining
│   └── Medical-instrunce/      # Project 3: Health insurance charge prediction with risk interaction terms
│
├── 🎯 Classificitions/         # ✅ SECTION 2: CLASSIFICATION (3/3 COMPLETED)
│   ├── Customer-chunk/         # Project 4: Bank customer churn & attrition prediction
│   ├── Credit-Card-fruad/      # Project 5: Highly imbalanced credit card fraud detection
│   └── Heart-disease/          # Project 6: Cardiovascular clinical risk classification
│
└── 🧩 Clustering/              # 🔄 SECTION 3: CLUSTERING (3 PROJECTS - UPCOMING)
    ├── Project-7/              # Project 7: E-Commerce Customer Segmentation (K-Means)
    ├── Project-8/              # Project 8: Spatial & Behavioral Anomaly Clustering
    └── Project-9/              # Project 9: Hierarchical / Pattern Analysis
```

---

## 📈 Section 1: Regression (Completed)

| # | Project | Key ML Concepts & Feature Engineering | Documentation |
| :-: | :--- | :--- | :-: |
| **01** | 🏋️ **Calories Burnt** | Physiological metric modeling, `BMI` feature derivation, Linear Regression vs. Random Forest Regressor | [`README`](./Regressition/Calories-Burnt/README.md) |
| **02** | 🚗 **Car Price Prediction** | Regex string mining (`HP`, `Engine_Liters`), 99th percentile price capping, `CarAge`, `MileagePerYear` | [`README`](./Regressition/Car-price-prediction/README.md) |
| **03** | 🏥 **Medical Insurance** | Non-linear interaction terms (`bmi_smoker`), one-hot regional encoding, Linear Regression modeling | [`README`](./Regressition/Medical-instrunce/README.md) |

---

## 🎯 Section 2: Classification (Completed)

| # | Project | Key ML Concepts & Feature Engineering | Documentation |
| :-: | :--- | :--- | :-: |
| **04** | 🏦 **Bank Customer Churn** | Feature alignment across train/test splits, one-hot encoding (`Geography`, `Gender`), Logistic Regression & Random Forest Classifier | [`README`](./Classificitions/Customer-chunk/README.md) |
| **05** | 💳 **Credit Card Fraud** | Imbalanced dataset handling, PCA feature preservation (`V1`-`V28`), `StandardScaler` (`Time`, `Amount`), Precision/Recall analysis | [`README`](./Classificitions/Credit-Card-fruad/README.md) |
| **06** | 🩺 **Heart Disease Risk** | Clinical risk classification, multi-categorical dummy encoding (`cp`, `restecg`, `slope`, `thal`), Confusion Matrix diagnostics | [`README`](./Classificitions/Heart-disease/README.md) |

---

## 🛠️ Project Design & Code Organization

Each project directory follows a clean, standardized layout:

```
<Project-Folder>/
├── dataset/             # Raw and preprocessed CSV datasets
├── models/              # Saved model binaries (*.pkl - ignored by Git)
├── outputs/             # Saved visual plots (confusion matrices, residual plots)
├── preprocessing.py     # Data cleaning, encoding & feature engineering
├── train.py             # Model training & serialization
├── evaluate.py          # Metrics evaluation (Accuracy, Precision, Recall, F1) & plotting
├── predict.py           # Single-sample / batch inference pipeline
├── requirements.txt     # Python package dependencies
└── README.md            # Individual project documentation
```

---

## ⚙️ Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/goobolabs/Machine-Learning-Projects.git
cd Machine-Learning-Projects
```

### 2. Run a Specific Project
Navigate into any completed project folder (e.g., Bank Customer Churn), install dependencies, and execute the scripts in sequence:

```bash
cd Classificitions/Customer-chunk
pip install -r requirements.txt

# Run the 4-step pipeline
python preprocessing.py   # Clean raw data & engineer features
python train.py           # Train models & save binaries
python evaluate.py        # Evaluate metrics & generate plots
python predict.py         # Run inference on sample / test data
```

---

## 📺 Course Links & Instructor
- 🎥 **YouTube Course**: [Watch Full Course on YouTube](https://youtu.be/Ln9f16vI67U?si=JUgKclZXp7HDYFs4)
- 👨‍🏫 **Instructors**: Omar Tood & Mr Sharafdin
- 💻 **GitHub Repository**: [github.com/goobolabs/Machine-Learning-Projects](https://github.com/goobolabs/Machine-Learning-Projects)
