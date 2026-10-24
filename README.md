# 🚀 Data Science & Machine Learning. Hands-On Projects

[![YouTube Watch Course](https://img.shields.io/badge/YouTube-Watch%20Full%20Course-red?style=for-the-badge&logo=youtube)](https://youtu.be/Ln9f16vI67U?si=JUgKclZXp7HDYFs4)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

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
│   ├── Energy-consumption/     # Project 2: Building energy consumption forecasting with operational & temporal features
│   └── Medical-instrunce/      # Project 3: Health insurance charge prediction with risk interaction terms
│
├── 🎯 Classificitions/         # ✅ SECTION 2: CLASSIFICATION (3/3 COMPLETED)
│   ├── Network-Intrusion/      # Project 4: Cyber security network intrusion detection & anomaly classification
│   ├── student_dropout_classification/ # Project 5: Student academic dropout prediction & early risk classification
│   └── Heart-disease/          # Project 6: Cardiovascular clinical risk classification
│
└── 🧩 Clustering/              # ✅ SECTION 3: CLUSTERING (3/3 COMPLETED)
    ├── mall_customer_segmentation/         # Project 7: Income & spending-based customer segmentation
    ├── football_player_clustering/         # Project 8: Statistical playing-style clustering
    └── mobile_usage_clustering/            # Project 9: Smartphone usage behavior segmentation
```

---

## 📈 Section 1: Regression (Completed)

| # | Project | Key ML Concepts & Feature Engineering | Documentation |
| :-: | :--- | :--- | :-: |
| **01** | 🏋️ **Calories Burnt** | Physiological metric modeling, `BMI` feature derivation, Linear Regression vs. Random Forest Regressor | [`README`](./Regressition/Calories-Burnt/README.md) |
| **02** | ⚡ **Energy Consumption** | Environmental & operational scaling (`StandardScaler`), temporal hour extraction, HVAC/Lighting binary encoding | [`README`](./Regressition/Energy-consumption/README.md) |
| **03** | 🏥 **Medical Insurance** | Non-linear interaction terms (`bmi_smoker`), one-hot regional encoding, Linear Regression modeling | [`README`](./Regressition/Medical-instrunce/README.md) |

---

## 🎯 Section 2: Classification (Completed)

| # | Project | Key ML Concepts & Feature Engineering | Documentation |
| :-: | :--- | :--- | :-: |
| **04** | 🛡️ **Network Intrusion** | Cyber-security traffic classification, protocol/service one-hot encoding, train/test feature alignment, LR vs RF | [`README`](./Classificitions/Network-Intrusion/README.md) |
| **05** | 🎓 **Student Dropout Risk** | Academic attrition prediction, demographic & behavioral encoding, missing value imputation, `StandardScaler`, LR vs RF | [`README`](./Classificitions/student_dropout_classification/README.md) |
| **06** | 🩺 **Heart Disease Risk** | Clinical risk classification, multi-categorical dummy encoding (`cp`, `restecg`, `slope`, `thal`), Confusion Matrix diagnostics | [`README`](./Classificitions/Heart-disease/README.md) |

---

## 🧩 Section 3: Clustering (Completed)

| # | Project | Key ML Concepts & Feature Engineering | Documentation |
| :-: | :--- | :--- | :-: |
| **07** | **Mall Customer Segmentation** | Income and spending features, `StandardScaler`, K-Means (`k=5`), cluster profiles & Elbow Method | [`README`](./Clustering/mall_customer_segmentation/README.md) |
| **08** | **Football Player Clustering** | 15 playing-style statistics, minimum playing-time filter, `StandardScaler`, K-Means (`k=3`), position interpretation & PCA | [`README`](./Clustering/football_player_clustering/README.md) |
| **09** | **Mobile Usage Clustering** | Four mobile usage features, `StandardScaler`, K-Means (`k=5`), sampled silhouette evaluation & PCA | [`README`](./Clustering/mobile_usage_clustering/README.md) |

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
├── evaluate.py          # Task-specific metrics evaluation & plotting
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
Navigate into any completed project folder (e.g., Network Intrusion Detection), install dependencies, and execute the scripts in sequence:

```bash
cd Classificitions/Network-Intrusion
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

---

## 📜 License & Copyright

Copyright (c) 2026 **Goobo Labs**.

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

- **Attribution**: You must give appropriate credit to **Goobo Labs**.
- **NonCommercial**: You may not use the material for commercial purposes.
- **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license.
