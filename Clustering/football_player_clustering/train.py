import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = (
    "./dataset/"
    "clean_football_players.csv"
)

df = pd.read_csv(DATA_PATH)

# 1) Playing-style features
feature_columns = [
    "Shots",
    "SoT",
    "Assists",
    "PasAss",
    "PasProg",
    "SCA",
    "Tkl",
    "Int",
    "Blocks",
    "Clr",
    "ToAtt",
    "ToSuc",
    "CarProg",
    "RecProg",
    "AerWon"
]

X = df[feature_columns]

# 2) Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# 3) Train KMeans
# k=3 gave the best Silhouette Score
# among the tested values.
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
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
