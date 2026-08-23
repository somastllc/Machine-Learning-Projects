# 🚀 Data Science & Machine Learning — Hands-On Projects

[![YouTube Watch Course](https://img.shields.io/badge/YouTube-Watch%20Full%20Course-red?style=for-the-badge&logo=youtube)](https://youtu.be/Ln9f16vI67U?si=JUgKclZXp7HDYFs4)

Welcome to the official hands-on repository accompanying the **Full Data Science & Machine Learning Course** on YouTube! 

> 📺 **Watch the Full Video Course on YouTube**: [Data Science & Machine Learning Full Course](https://youtu.be/Ln9f16vI67U?si=JUgKclZXp7HDYFs4)

This repository is designed to help students and developers master Data Science and Machine Learning by building **9 real-world projects** step by step across three core pillars: **Regression**, **Classification**, and **Clustering**.

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
├── 🎯 Classification/          # 🔄 SECTION 2: CLASSIFICATION (3 PROJECTS - UPCOMING)
│   ├── Project-4/              # Project 4: Credit Risk / Customer Churn Prediction
│   ├── Project-5/              # Project 5: Medical Diagnostics Classification
│   └── Project-6/              # Project 6: Fraud / Spam Detection
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

## 🛠️ Project Design & Code Organization

Each project directory follows a clean, standardized layout:

```
<Project-Folder>/
├── dataset/             # Raw and preprocessed CSV datasets
├── models/              # Saved model binaries (*.pkl - ignored by Git)
├── outputs/             # Saved visual plots (residuals, scatter plots)
├── preprocessing.py     # Data cleaning, encoding & feature engineering
├── train.py             # Model training & serialization
├── evaluate.py          # Metrics evaluation (MAE, RMSE, R²) & plotting
├── predict.py           # Single-sample inference pipeline
├── requirements.txt     # Python package dependencies
└── README.md            # Individual project documentation
```

---

## ⚙️ Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/goobolabs/Machine-Learning-Project.git
cd Machine-Learning-Project
```

### 2. Run a Specific Project
Navigate into any completed project folder, install dependencies, and execute the scripts in sequence:

```bash
cd Regressition/Calories-Burnt
pip install -r requirements.txt

# Run the 4-step pipeline
python preprocessing.py   # Clean raw data & engineer features
python train.py           # Train models & save binaries
python evaluate.py        # Evaluate metrics & generate plots
python predict.py         # Run inference on sample data
```

---

## 📺 Course Links & Instructor
- 🎥 **YouTube Course**: [Watch Full Course on YouTube](https://youtu.be/Ln9f16vI67U?si=JUgKclZXp7HDYFs4)
- 👨‍🏫 **Instructors**: Omar Tood & Mr Sharafdin
- 💻 **GitHub Repository**: [github.com/goobolabs/Machine-Learning-Project](https://github.com/goobolabs/Machine-Learning-Project)
