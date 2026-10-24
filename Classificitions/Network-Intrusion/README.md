# 🛡️ Network Intrusion Detection Pipeline

An end-to-end Machine Learning binary classification project that identifies malicious network intrusions (`class`: 0 = Normal, 1 = Anomaly) across cyber-security network connection sessions.

---

## 📌 Problem Statement & Overview

### The Problem
Modern network infrastructure faces constant threat from cyber attacks, unauthorized probes, denial-of-service (DoS), and intrusion attempts. Automated intrusion detection systems (IDS) must evaluate incoming network packets and connection signatures in real time to distinguish benign user activity from security threats.

### The Machine Learning Solution
This project builds a multi-stage classification pipeline that ingests raw network traffic metrics, preprocesses categorical connection attributes (`protocol_type`, `service`, `flag`), standardizes numerical metrics, and trains **Logistic Regression** and **Random Forest** classifiers to flag anomalous connections.

---

## ⚙️ Pipeline Architecture

```
                 ┌────────────────────────────────┐
                 │ dataset/Train_data.csv         │
                 │ dataset/Test_data.csv          │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │        preprocessing.py        │
                 │ • Deduplicate Training Set     │
                 │ • Binary Map Class (0 / 1)     │
                 │ • One-Hot Encode (Protocol,    │
                 │   Service, Flag)               │
                 │ • Reindex Test Feature Columns │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │ dataset/clean_train.csv        │
                 │ dataset/clean_test.csv         │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │           train.py             │
                 │ • Stratified Train-Test Split  │
                 │ • StandardScaler (LR Features) │
                 │ • Fit Logistic Regression & RF │
                 │ • Save *.pkl Artifacts & Scaler│
                 └───────────────┬────────────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
┌───────────────────────────────┐                 ┌───────────────────────────────┐
│          evaluate.py          │                 │          predict.py           │
│ • Metrics: Accuracy, Precision│                 │ • Batch Inference on          │
│   Recall, F1, ROC-AUC         │                 │   `clean_test.csv`            │
│ • Confusion Matrix Plots      │                 │ • Save `test_predictions.csv` │
└───────────────────────────────┘                 └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

The dataset contains network connection records characterized by network communication parameters.

### Input Features

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `protocol_type` | Categorical (Encoded) | Protocol used (e.g., `tcp`, `udp`, `icmp`) |
| `service` | Categorical (Encoded) | Destination network service (e.g., `http`, `smtp`, `ftp`, `private`) |
| `flag` | Categorical (Encoded) | Connection status flag (e.g., `SF`, `S0`, `REJ`) |
| Numerical Metrics | Numerical | Duration, source bytes, destination bytes, error rates, connection counts scaled via `StandardScaler` |

### Target Variable
- **`class`**: `0` = Normal Network Traffic, `1` = Malicious Anomaly / Intrusion.

---

## 🤖 Models Trained

1. **Logistic Regression**
   - Fits a linear decision boundary using standardized numerical attributes (`StandardScaler`).
   - Serves as a fast, low-latency intrusion detection baseline.

2. **Random Forest Classifier**
   - Ensemble of 100 decision trees (`n_estimators=100`, `random_state=42`).
   - Learns non-linear feature combinations across protocol flags and packet rate statistics.

---

## 📈 Evaluation & Diagnostics

Models are evaluated on holdout validation data using key classification metrics:
- **Accuracy**: Overall proportion of correctly categorized connections.
- **Precision**: Ratio of true anomalies out of all flagged intrusion alerts.
- **Recall (Sensitivity)**: Ratio of detected intrusions out of all actual network attacks.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Model discriminative capability across all decision thresholds.

### Confusion Matrix Plots (Saved in `outputs/`)
- `confusion_matrix_lr.png`: Confusion matrix for Logistic Regression.
- `confusion_matrix_rf.png`: Confusion matrix for Random Forest Classifier.

---

## 💻 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clean datasets & align feature columns
python preprocessing.py

# 3. Train Logistic Regression & Random Forest classifiers
python train.py

# 4. Evaluate metrics & render confusion matrices
python evaluate.py

# 5. Execute test set batch predictions
python predict.py
```

### View all features in one chart

```bash
python visualize_features.py
```

Saves `outputs/all_features.png` with all 41 original features and the class
target, using colors to compare normal and anomaly records. Counts use log
axes; long-tailed numeric features use `log1p`. Large categorical features show
the top 10 values plus an `Other` group. No training or preprocessing is needed.

To plot another dataset:

```bash
python visualize_features.py --data dataset/Test_data.csv --output outputs/test_features.png
```
