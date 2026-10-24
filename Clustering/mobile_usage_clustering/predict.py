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

# 2) New mobile user
new_data = pd.DataFrame([{
    "Daily_Phone_Hours": 8.5,
    "Social_Media_Hours": 5.0,
    "App_Usage_Count": 25,
    "Weekend_Screen_Time_Hours": 10.0
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

print("\n=== MOBILE USAGE CLUSTER ===")
print(f"Predicted Cluster: {cluster}")
