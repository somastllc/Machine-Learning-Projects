import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

DATA_PATH = "./dataset/clean_human_activity_windows.csv"
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

# 2) Scale data and predict clusters
X_scaled = scaler.transform(X)

clusters = kmeans.predict(
    X_scaled
)

df["Cluster"] = clusters

# 3) Evaluation metrics
# Silhouette Score can be expensive,
# so we calculate it on a reproducible sample.
sample_size = min(
    20000,
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
print(f"Inertia              : {kmeans.inertia_:.2f}")
print(f"Silhouette Score     : {silhouette:.4f}")
print(f"Davies-Bouldin Score : {davies_bouldin:.4f}")

# 4) Cluster sizes
print("\n=== CLUSTER SIZES ===")
print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)

# 5) Cluster profiles
print("\n=== CLUSTER PROFILES ===")
print(
    df.groupby("Cluster")[
        feature_columns
    ].mean().round(2)
)

# 6) Reference activity distribution
# The activity label was NOT used during training.
print("\n=== ACTIVITY DISTRIBUTION BY CLUSTER ===")
print(
    pd.crosstab(
        df["Cluster"],
        df["activity"],
        normalize="index"
    ).round(3)
)

# 7) Elbow Method
sample_for_k = df.sample(
    n=min(10000, len(df)),
    random_state=42
)

sample_for_k_scaled = scaler.transform(
    sample_for_k[feature_columns]
)

k_values = range(2, 9)
inertias = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(
        sample_for_k_scaled
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

# 8) PCA cluster visualization
pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    sample_scaled
)

plt.figure()

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=sample_clusters,
    s=10,
    alpha=0.5
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Human Activity Clusters")

plt.tight_layout()

plt.savefig(
    "./outputs/pca_clusters.png"
)

plt.close()

# 9) Cluster size plot
cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

plt.figure()

cluster_sizes.plot(
    kind="bar"
)

plt.xlabel("Cluster")
plt.ylabel("Number of Windows")
plt.title("Cluster Sizes")

plt.tight_layout()

plt.savefig(
    "./outputs/cluster_sizes.png"
)

plt.close()

# 10) Single-row sanity check
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
print(f"Row index : {i}")
print(f"Cluster   : {cluster}")
