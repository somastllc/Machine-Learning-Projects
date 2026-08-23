# 🤖 Machine Learning Projects Repository

Welcome to the **Machine Learning Projects** repository developed as part of the **Goobo Labs Bootcamp**. This repository serves as a comprehensive portfolio of end-to-end Machine Learning implementations—ranging from data preprocessing and feature engineering to model training, evaluation, and inference.

---

## 🗺️ Repository Structure & Learning Roadmap

The repository is organized by Machine Learning paradigms:

```
Machine-Learning-Project/
├── 📈 Regressition/            # ✅ COMPLETED
│   ├── Calories-Burnt/         # Calorie expenditure estimation from body & workout sensors
│   ├── Car-price-prediction/   # Used vehicle valuation with regex engine text mining
│   └── Medical-instrunce/      # Health insurance charge prediction with risk interaction features
├── 🎯 Classification/          # 🔄 UPCOMING
└── 🧩 Clustering/              # 🔄 UPCOMING
```

---

## 📈 1. Regression Projects (`/Regressition`)

| Project Name | Problem Description | Key Algorithms | Features & Engineering | Subfolder |
| :--- | :--- | :--- | :--- | :--- |
| 🏋️ **Calories Burnt** | Predict calories burned during workouts | Linear Regression, Random Forest | Heart Rate, Body Temp, Workout Duration, `BMI` | [`/Calories-Burnt`](./Regressition/Calories-Burnt) |
| 🚗 **Car Price Prediction** | Estimate resale value of used cars | Linear Regression, Random Forest | Regex Engine parsing (HP/Liters), `CarAge`, `MileagePerYear` | [`/Car-price-prediction`](./Regressition/Car-price-prediction) |
| 🏥 **Medical Insurance** | Predict individual health insurance charges | Linear Regression | Demographic data, One-Hot region encoding, `bmi_smoker` interaction | [`/Medical-instrunce`](./Regressition/Medical-instrunce) |

> 📌 *Each project subfolder contains its own detailed [`README.md`](./Regressition/Calories-Burnt/README.md) with architectural flowcharts, feature breakdowns, and instructions.*

---

## 🎯 2. Classification Projects (`/Classification`) — *Upcoming*

*Planned projects focusing on binary and multi-class classification tasks (e.g., Logistic Regression, Decision Trees, Support Vector Machines, XGBoost).*

---

## 🧩 3. Clustering Projects (`/Clustering`) — *Upcoming*

*Planned projects focusing on unsupervised learning, customer segmentation, and pattern recognition (e.g., K-Means, Hierarchical Clustering, DBSCAN).*

---

## 🛠️ End-to-End Project Workflow Standard

Every project in this repository adheres to a modular, production-ready project layout:

1. **`preprocessing.py`**: Merges raw datasets, handles missing values, removes duplicates, encodes categorical features, and performs custom feature engineering. Exports clean dataset to `dataset/clean_*.csv`.
2. **`train.py`**: Splits data (80/20 train/test split), scales numerical attributes (where applicable), fits models, and exports trained model binaries and metadata (`.pkl`) to `models/`.
3. **`evaluate.py`**: Evaluates models on holdout test data using metrics ($R^2$, MAE, RMSE) and generates diagnostic visual outputs (`actual_vs_predicted.png`, `residuals.png`) saved to `outputs/`.
4. **`predict.py`**: Runs inference on new, unseen data samples using saved model pipelines.

---

## 🔒 Git & Binary File Management

To maintain repository efficiency and comply with GitHub file size recommendations:
- Heavy serialized model binaries (`*.pkl`) are excluded via [.gitignore](./.gitignore).
- Data cleaning scripts (`preprocessing.py`) and training scripts (`train.py`) allow full reproducibility of all model artifacts locally.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- `pip` package manager

### Installation
Clone the repository:
```bash
git clone https://github.com/goobolabs/Machine-Learning-Project.git
cd Machine-Learning-Project
```

To run a specific project (e.g. Calories Burnt):
```bash
cd Regressition/Calories-Burnt
pip install -r requirements.txt
python preprocessing.py
python train.py
python evaluate.py
```

---

## 👤 Author
**Omar Tood** — *Goobo Labs Bootcamp*
