import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

DATA_PATH = "./dataset/clean_calories.csv"

df = pd.read_csv(DATA_PATH)

# 1) Define features and target
X = df.drop(columns=["Calories"])
y = df["Calories"]

# 2) Same train/test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3) Load models and scaler
lr = joblib.load("./models/linear_regression.pkl")
rf = joblib.load("./models/random_forest.pkl")
scaler = joblib.load("./models/scaler.pkl")

scale_cols = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "BMI"
]

X_test_lr = X_test.copy()
X_test_lr[scale_cols] = scaler.transform(X_test[scale_cols])

# 4) Predictions
p_lr = lr.predict(X_test_lr)
p_rf = rf.predict(X_test)

# 5) Evaluation function
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {name} ===")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")

evaluate_model("LINEAR REGRESSION", y_test, p_lr)
evaluate_model("RANDOM FOREST", y_test, p_rf)

# 6) Single-row sanity check
# Pick one unseen row from X_test and predict with both models.
# Use iloc[[i]] to keep it as a DataFrame with column names.

i = 3

x_one_df = X_test.iloc[[i]]
y_true = y_test.iloc[i]

x_one_lr = x_one_df.copy()
x_one_lr[scale_cols] = scaler.transform(x_one_df[scale_cols])

p_lr_one = float(lr.predict(x_one_lr)[0])
p_rf_one = float(rf.predict(x_one_df)[0])

print("\n=== SINGLE-ROW SANITY CHECK ===")
print(f"Actual Calories: {y_true:,.0f}")
print(f"LR Prediction  : {p_lr_one:,.2f}")
print(f"RF Prediction  : {p_rf_one:,.2f}")

# 7) Actual vs Predicted
plt.figure()
plt.scatter(y_test, p_rf)
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted - Random Forest")
plt.tight_layout()
plt.savefig("./outputs/actual_vs_predicted.png")
plt.close()

# 8) Residual plot
residuals = y_test - p_rf

plt.figure()
plt.scatter(p_rf, residuals)
plt.axhline(0)
plt.xlabel("Predicted Calories")
plt.ylabel("Residuals")
plt.title("Residual Plot - Random Forest")
plt.tight_layout()
plt.savefig("./outputs/residuals.png")
plt.close()
