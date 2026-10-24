import joblib
import pandas as pd

# 1) Load saved objects
kmeans = joblib.load(
    "./models/kmeans.pkl"
)

scaler = joblib.load(
    "./models/scaler.pkl"
)

feature_columns = joblib.load(
    "./models/feature_columns.pkl"
)

# 2) New shopping session
new_data = pd.DataFrame([{
    "Administrative": 2,
    "Administrative_Duration": 60.0,
    "Informational": 1,
    "Informational_Duration": 30.0,
    "ProductRelated": 25,
    "ProductRelated_Duration": 900.0,
    "BounceRates": 0.01,
    "ExitRates": 0.03,
    "PageValues": 7.5,
    "SpecialDay": 0.0
}])

# 3) Keep same feature order
new_data = new_data.reindex(
    columns=feature_columns
)

# 4) Scale
new_data_scaled = scaler.transform(
    new_data
)

# 5) Predict cluster
cluster = int(
    kmeans.predict(
        new_data_scaled
    )[0]
)

print("\n=== ONLINE SHOPPING BEHAVIOR CLUSTER ===")
print(f"Predicted Cluster: {cluster}")
