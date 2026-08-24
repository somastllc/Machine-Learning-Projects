import joblib
import pandas as pd

# 1) Load clean test data
test = pd.read_csv(
    "./dataset/clean_test.csv"
)

# 2) Load saved objects
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

feature_columns = joblib.load(
    "./models/feature_columns.pkl"
)

test = test.reindex(
    columns=feature_columns,
    fill_value=0
)

# 3) Prepare Logistic Regression input
test_lr = test.copy()

test_lr[scale_cols] = scaler.transform(
    test[scale_cols]
)

# 4) Predict
lr_predictions = lr.predict(test_lr)
rf_predictions = rf.predict(test)

rf_probabilities = rf.predict_proba(
    test
)[:, 1]

# 5) Save predictions
results = pd.DataFrame({
    "LR_Prediction": lr_predictions,
    "RF_Prediction": rf_predictions,
    "RF_Anomaly_Probability": rf_probabilities
})

results["RF_Label"] = results[
    "RF_Prediction"
].map({
    0: "normal",
    1: "anomaly"
})

results.to_csv(
    "./outputs/test_predictions.csv",
    index=False
)

print("\n=== TEST PREDICTIONS ===")
print(results.head())

print(
    "\nSaved predictions to: "
    "./outputs/test_predictions.csv"
)
