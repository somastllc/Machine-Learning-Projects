import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = "./dataset/clean_human_activity_windows.csv"
df = pd.read_csv(DATA_PATH)

# 1) Features used for clustering
feature_columns = [
    "x_mean",
    "x_std",
    "y_mean",
    "y_std",
    "z_mean",
    "z_std",
    "magnitude_mean",
    "magnitude_std"
]

X = df[feature_columns]

# 2) Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# 3) Train KMeans
# k=3 was chosen after comparing several values
# using Silhouette Score and the Elbow Method.
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# 4) Save model and preprocessing objects
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
