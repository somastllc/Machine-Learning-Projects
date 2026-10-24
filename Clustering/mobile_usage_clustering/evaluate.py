import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

DATA_PATH = "./dataset/clean_mobile_usage.csv"
df = pd.read_csv(DATA_PATH)

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

X = df[feature_columns]

# 2) Scale and predict clusters
X_scaled = scaler.transform(X)

clusters = kmeans.predict(
    X_scaled
)

df["Cluster"] = clusters

# 3) Evaluation metrics
sample_size = min(
    10000,
    len(df)
)

sample = df.sample(
    n=sample_size,
    random_state=42
)

sample_scaled = scaler.transform(
    sample[feature_columns]
)

sample_clusters = kmeans.predict(
    sample_scaled
)

silhouette = silhouette_score(
    sample_scaled,
    sample_clusters
)

davies_bouldin = davies_bouldin_score(
    X_scaled,
    clusters
)

print("\n=== CLUSTERING METRICS ===")
print(
    f"Inertia              : "
    f"{kmeans.inertia_:.2f}"
)
print(
    f"Silhouette Score     : "
    f"{silhouette:.4f}"
)
print(
    f"Davies-Bouldin Score : "
    f"{davies_bouldin:.4f}"
)

# 4) Cluster sizes
print("\n=== CLUSTER SIZES ===")

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_sizes)

# 5) Mobile usage profiles
print("\n=== MOBILE USAGE PROFILES ===")

usage_profiles = (
    df.groupby("Cluster")[
        feature_columns
    ]
    .mean()
    .round(2)
)

print(usage_profiles)

usage_profiles.to_csv(
    "./outputs/cluster_profiles.csv"
)

# 6) Lifestyle interpretation
# These columns are NOT used to train KMeans.
print("\n=== LIFESTYLE PROFILE BY CLUSTER ===")

lifestyle_columns = [
    "Age",
    "Work_Productivity_Score",
    "Sleep_Hours",
    "Stress_Level",
    "Caffeine_Intake_Cups"
]

print(
    df.groupby("Cluster")[
        lifestyle_columns
    ]
    .mean()
    .round(2)
)

# 7) Device distribution
print("\n=== DEVICE TYPE BY CLUSTER ===")

print(
    pd.crosstab(
        df["Cluster"],
        df["Device_Type"],
        normalize="index"
    ).round(3)
)

# 8) Elbow Method
k_values = range(2, 9)
inertias = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=5
    )

    model.fit(
        X_scaled
    )

    inertias.append(
        model.inertia_
    )

plt.figure()

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.tight_layout()

plt.savefig(
    "./outputs/elbow_method.png"
)

plt.close()

# 9) PCA visualization
pca = PCA(
    n_components=2
)

sample_pca = pca.fit_transform(
    sample_scaled
)

plt.figure()

plt.scatter(
    sample_pca[:, 0],
    sample_pca[:, 1],
    c=sample_clusters,
    s=10,
    alpha=0.5
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Mobile Usage Clusters")

plt.tight_layout()

plt.savefig(
    "./outputs/pca_clusters.png"
)

plt.close()

# 10) Cluster size plot
plt.figure()

cluster_sizes.plot(
    kind="bar"
)

plt.xlabel("Cluster")
plt.ylabel("Number of Users")
plt.title("Cluster Sizes")

plt.tight_layout()

plt.savefig(
    "./outputs/cluster_sizes.png"
)

plt.close()

# 11) Single-row sanity check
i = 3

x_one_df = X.iloc[[i]]

x_one_scaled = scaler.transform(
    x_one_df
)

cluster = int(
    kmeans.predict(
        x_one_scaled
    )[0]
)

print("\n=== SINGLE-ROW SANITY CHECK ===")
print(f"User ID   : {df.iloc[i]['User_ID']}")
print(f"Cluster   : {cluster}")
