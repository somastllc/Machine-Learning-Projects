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

# 2) New customer
new_data = pd.DataFrame([{
    "Annual Income (k$)": 75,
    "Spending Score (1-100)": 80
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

print("\n=== MALL CUSTOMER SEGMENT ===")
print(f"Predicted Cluster: {cluster}")
