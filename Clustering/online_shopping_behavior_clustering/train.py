import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = "./dataset/clean_online_shopping.csv"
df = pd.read_csv(DATA_PATH)

# 1) Behavioral features used for clustering
feature_columns = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

X = df[feature_columns]

# 2) Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# 3) Train KMeans
# k=3 was selected after comparing multiple k values.
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# 4) Save model objects
joblib.dump(
    kmeans,
    "./models/kmeans.pkl"
)

joblib.dump(
    scaler,
    "./models/scaler.pkl"
)

joblib.dump(
    feature_columns,
    "./models/feature_columns.pkl"
)

print("Training complete.")
print("Saved KMeans, scaler, and feature columns.")
