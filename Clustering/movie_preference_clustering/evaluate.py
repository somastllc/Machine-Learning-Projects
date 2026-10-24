import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

DATA_PATH = (
    "./dataset/"
    "clean_movie_preference_profiles.csv"
)

df = pd.read_csv(DATA_PATH)

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

X = df[feature_columns]

# ==========================================
# 2) SCALE AND PREDICT CLUSTERS
# ==========================================

X_scaled = scaler.transform(X)

clusters = kmeans.predict(
    X_scaled
)

df["Cluster"] = clusters

# ==========================================
# 3) CLUSTERING METRICS
# ==========================================

silhouette = silhouette_score(
    X_scaled,
    clusters
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

# ==========================================
# 4) CLUSTER SIZES
# ==========================================

print("\n=== CLUSTER SIZES ===")

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_sizes)

# ==========================================
# 5) CLUSTER PROFILES
# ==========================================

profiles = (
    df.groupby("Cluster")[
        feature_columns
    ]
    .mean()
)

print("\n=== TOP GENRES PER CLUSTER ===")

for cluster_id in profiles.index:

    top_genres = (
        profiles.loc[cluster_id]
        .sort_values(
            ascending=False
        )
        .head(6)
    )

    print(
        f"\nCluster {cluster_id}"
    )

    for feature, value in top_genres.items():

        genre = feature.replace(
            "Preference_",
            ""
        )

        print(
            f"  {genre:<15} "
            f"{value:.3f}"
        )

# Save profile table
profiles.to_csv(
    "./outputs/cluster_profiles.csv"
)

# ==========================================
# 6) OPTIONAL USER METADATA SUMMARY
# ==========================================

print("\n=== USER METADATA BY CLUSTER ===")

print(
    df.groupby("Cluster")[
        "Age"
    ].mean().round(1)
)

print("\nGender distribution:")

print(
    pd.crosstab(
        df["Cluster"],
        df["Gender"],
        normalize="index"
    ).round(3)
)

# ==========================================
# 7) ELBOW METHOD
# ==========================================

k_values = range(2, 9)
inertias = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
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

# ==========================================
# 8) PCA VISUALIZATION
# ==========================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

plt.figure()

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    s=10,
    alpha=0.5
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Movie Preference Clusters")

plt.tight_layout()

plt.savefig(
    "./outputs/pca_clusters.png"
)

plt.close()

# ==========================================
# 9) CLUSTER SIZE PLOT
# ==========================================

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

# ==========================================
# 10) SINGLE-ROW SANITY CHECK
# ==========================================

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
print(f"UserID  : {df.iloc[i]['UserID']}")
print(f"Cluster : {cluster}")
