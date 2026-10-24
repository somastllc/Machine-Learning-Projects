import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = (
    "./dataset/"
    "clean_movie_preference_profiles.csv"
)

df = pd.read_csv(DATA_PATH)

# ==========================================
# 1) SELECT PREFERENCE FEATURES
# ==========================================

feature_columns = [
    col for col in df.columns
    if col.startswith("Preference_")
]

X = df[feature_columns]

# ==========================================
# 2) SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# 3) TRAIN KMEANS
# ==========================================

# k=4 gives useful movie preference groups.
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# ==========================================
# 4) SAVE MODEL OBJECTS
# ==========================================

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
