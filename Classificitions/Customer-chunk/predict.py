import joblib
import pandas as pd

# 1) Load clean test data and original IDs
test = pd.read_csv(
    "./dataset/clean_test.csv"
)

raw_test = pd.read_csv(
    "./dataset/test.csv"
)

# 2) Load model and feature columns
rf = joblib.load(
    "./models/random_forest.pkl"
)

feature_columns = joblib.load(
    "./models/feature_columns.pkl"
)

test = test.reindex(
    columns=feature_columns,
    fill_value=0
)

# 3) Predict churn
predictions = rf.predict(test)

probabilities = rf.predict_proba(
    test
)[:, 1]

# 4) Save predictions
results = pd.DataFrame({
    "id": raw_test["id"],
    "Exited": predictions,
    "Churn_Probability": probabilities
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
