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

# 2) New football player stats
new_data = pd.DataFrame([{
    "Shots": 2.5,
    "SoT": 1.0,
    "Assists": 0.20,
    "PasAss": 1.8,
    "PasProg": 4.5,
    "SCA": 3.8,
    "Tkl": 1.1,
    "Int": 0.4,
    "Blocks": 0.7,
    "Clr": 0.6,
    "ToAtt": 3.0,
    "ToSuc": 1.4,
    "CarProg": 3.2,
    "RecProg": 7.0,
    "AerWon": 0.9
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

print("\n=== FOOTBALL PLAYER CLUSTER ===")
print(
    f"Predicted Cluster: {cluster}"
)
