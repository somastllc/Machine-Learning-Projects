import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

DATA_PATH = "./dataset/clean_student_dropout.csv"
df = pd.read_csv(DATA_PATH)

# 1) Define features and target
X = df.drop(columns=["Dropout"])
y = df["Dropout"]

# 2) Same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3) Load models and preprocessing objects
lr = joblib.load(
    "./models/logistic_regression.pkl"
)

rf = joblib.load(
    "./models/random_forest.pkl"
)

scaler = joblib.load(
    "./models/scaler.pkl"
)

scale_cols = joblib.load(
    "./models/scale_columns.pkl"
)

# 4) Prepare Logistic Regression test data
X_test_lr = X_test.copy()

X_test_lr[scale_cols] = scaler.transform(
    X_test[scale_cols]
)

# 5) Predictions
p_lr = lr.predict(X_test_lr)
p_rf = rf.predict(X_test)

prob_lr = lr.predict_proba(X_test_lr)[:, 1]
prob_rf = rf.predict_proba(X_test)[:, 1]

# 6) Evaluation function
def evaluate_model(name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    p_lr,
    prob_lr
)

evaluate_model(
    "RANDOM FOREST",
    y_test,
    p_rf,
    prob_rf
)

# 7) Single-row sanity check
# Pick one unseen row from X_test and predict with both models.
# Use iloc[[i]] to keep it as a DataFrame with column names.

i = 3

x_one_df = X_test.iloc[[i]]
y_true = y_test.iloc[i]

x_one_lr = x_one_df.copy()

x_one_lr[scale_cols] = scaler.transform(
    x_one_df[scale_cols]
)

p_lr_one = int(
    lr.predict(x_one_lr)[0]
)

p_rf_one = int(
    rf.predict(x_one_df)[0]
)

print("\n=== SINGLE-ROW SANITY CHECK ===")
print(f"Actual Dropout: {y_true}")
print(f"LR Prediction : {p_lr_one}")
print(f"RF Prediction : {p_rf_one}")

# 8) Confusion Matrix - Logistic Regression
ConfusionMatrixDisplay.from_predictions(
    y_test,
    p_lr
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.tight_layout()

plt.savefig(
    "./outputs/confusion_matrix_lr.png"
)

plt.close()

# 9) Confusion Matrix - Random Forest
ConfusionMatrixDisplay.from_predictions(
    y_test,
    p_rf
)

plt.title(
    "Confusion Matrix - Random Forest"
)

plt.tight_layout()

plt.savefig(
    "./outputs/confusion_matrix_rf.png"
)

plt.close()
