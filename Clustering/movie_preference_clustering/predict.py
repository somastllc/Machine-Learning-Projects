import joblib
import pandas as pd

# ==========================================
# 1) LOAD SAVED OBJECTS
# ==========================================

kmeans = joblib.load(
    "./models/kmeans.pkl"
)

scaler = joblib.load(
    "./models/scaler.pkl"
)

feature_columns = joblib.load(
    "./models/feature_columns.pkl"
)

# ==========================================
# 2) NEW USER MOVIE PREFERENCES
# ==========================================

# These values are preference scores:
# watch share * average rating / 5

new_data = pd.DataFrame([{
    "Preference_Action": 0.38,
    "Preference_Adventure": 0.22,
    "Preference_Animation": 0.03,
    "Preference_Children's": 0.02,
    "Preference_Comedy": 0.15,
    "Preference_Crime": 0.10,
    "Preference_Documentary": 0.01,
    "Preference_Drama": 0.18,
    "Preference_Fantasy": 0.05,
    "Preference_Film-Noir": 0.02,
    "Preference_Horror": 0.08,
    "Preference_Musical": 0.02,
    "Preference_Mystery": 0.05,
    "Preference_Romance": 0.06,
    "Preference_Sci-Fi": 0.30,
    "Preference_Thriller": 0.20,
    "Preference_War": 0.08,
    "Preference_Western": 0.02
}])

# ==========================================
# 3) KEEP SAME FEATURE ORDER
# ==========================================

new_data = new_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# ==========================================
# 4) SCALE
# ==========================================

new_data_scaled = scaler.transform(
    new_data
)

# ==========================================
# 5) PREDICT CLUSTER
# ==========================================

cluster = int(
    kmeans.predict(
        new_data_scaled
    )[0]
)

print("\n=== MOVIE PREFERENCE CLUSTER ===")
print(f"Predicted Cluster: {cluster}")
