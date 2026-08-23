import joblib
import pandas as pd

# 1) Load saved objects
lr = joblib.load(
    "./models/logistic_regression.pkl"
)

rf = joblib.load(
    "./models/random_forest.pkl"
)

scaler = joblib.load(
    "./models/scaler.pkl"
)

feature_columns = joblib.load(
    "./models/feature_columns.pkl"
)

# 2) New patient data
new_data = pd.DataFrame({
    "age": [54],
    "sex": [1],
    "cp": [1],
    "trestbps": [130],
    "chol": [240],
    "fbs": [0],
    "restecg": [1],
    "thalach": [150],
    "exang": [0],
    "oldpeak": [1.2],
    "slope": [2],
    "ca": [0],
    "thal": [2]
})

# 3) Encode categorical columns
categorical_cols = [
    "cp",
    "restecg",
    "slope",
    "thal"
]

new_data = pd.get_dummies(
    new_data,
    columns=categorical_cols,
    dtype=int
)

new_data = new_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# 4) Prepare Logistic Regression input
scale_cols = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

new_data_lr = new_data.copy()

new_data_lr[scale_cols] = scaler.transform(
    new_data[scale_cols]
)

# 5) Predict
lr_prediction = int(
    lr.predict(new_data_lr)[0]
)

rf_prediction = int(
    rf.predict(new_data)[0]
)

lr_probability = float(
    lr.predict_proba(new_data_lr)[0][1]
)

rf_probability = float(
    rf.predict_proba(new_data)[0][1]
)

print("\n=== HEART DISEASE PREDICTION ===")
print(
    f"Logistic Regression: {lr_prediction} "
    f"({lr_probability:.2%})"
)
print(
    f"Random Forest      : {rf_prediction} "
    f"({rf_probability:.2%})"
)
