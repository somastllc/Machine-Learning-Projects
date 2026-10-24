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
    "clean_football_players.csv"
)

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

# 4) Cluster sizes
print("\n=== CLUSTER SIZES ===")

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_sizes)

# 5) Cluster profiles
print("\n=== CLUSTER PROFILES ===")

profiles = (
    df.groupby("Cluster")[
        feature_columns
    ]
    .mean()
    .round(2)
)

print(profiles)

profiles.to_csv(
    "./outputs/cluster_profiles.csv"
)

# 6) Position distribution by cluster
# Position is NOT used to train the model.
print("\n=== POSITION DISTRIBUTION BY CLUSTER ===")

position_distribution = pd.crosstab(
    df["Cluster"],
    df["Pos"],
    normalize="index"
).round(3)

print(position_distribution)

position_distribution.to_csv(
    "./outputs/position_distribution.csv"
)

# 7) Example players from each cluster
print("\n=== EXAMPLE PLAYERS BY CLUSTER ===")

for cluster_id in sorted(
    df["Cluster"].unique()
):
    print(
        f"\nCluster {cluster_id}:"
    )

    example_players = (
        df[
            df["Cluster"] == cluster_id
        ][
            [
                "Player",
                "Pos",
                "Squad",
                "Comp"
            ]
        ]
        .head(10)
    )

    print(
        example_players.to_string(
            index=False
        )
    )

# 8) Elbow Method
k_values = range(2, 9)
inertias = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
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

X_pca = pca.fit_transform(
    X_scaled
)

plt.figure()

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    s=15,
    alpha=0.6
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Football Player Clusters")

plt.tight_layout()

plt.savefig(
    "./outputs/player_clusters.png"
)

plt.close()

# 10) Cluster size plot
plt.figure()

cluster_sizes.plot(
    kind="bar"
)

plt.xlabel("Cluster")
plt.ylabel("Number of Players")
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
print(
    f"Player  : "
    f"{df.iloc[i]['Player']}"
)
print(
    f"Position: "
    f"{df.iloc[i]['Pos']}"
)
print(
    f"Cluster : "
    f"{cluster}"
)
