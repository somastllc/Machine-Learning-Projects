import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = "./dataset/clean_mall_customers.csv"
df = pd.read_csv(DATA_PATH)

# 1) Features used for clustering
feature_columns = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[feature_columns]

# 2) Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# 3) Train KMeans
# k=5 was selected using the Elbow Method
# and Silhouette Score.
kmeans = KMeans(
    n_clusters=5,
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
