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

# 2) New activity window
# These values summarize a window of sensor readings.
new_data = pd.DataFrame([{
    "x_mean": 0.8,
    "x_std": 3.5,
    "y_mean": 8.5,
    "y_std": 4.0,
    "z_mean": 0.2,
    "z_std": 3.2,
    "magnitude_mean": 11.0,
    "magnitude_std": 4.2
}])

# 3) Keep the same feature order
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

print("\n=== HUMAN ACTIVITY CLUSTER ===")
print(f"Predicted Cluster: {cluster}")
