import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

DATA_PATH = "./dataset/clean_mall_customers.csv"
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
        [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ]
    .mean()
    .round(2)
)

print(profiles)

profiles.to_csv(
    "./outputs/cluster_profiles.csv"
)

# 6) Gender distribution by cluster
print("\n=== GENDER DISTRIBUTION BY CLUSTER ===")

print(
    pd.crosstab(
        df["Cluster"],
        df["Gender"],
        normalize="index"
    ).round(3)
)

# 7) Elbow Method
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

# 8) Customer cluster scatter plot
plt.figure()

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=clusters,
    s=40,
    alpha=0.7
)

centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=180
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segments")

plt.tight_layout()

plt.savefig(
    "./outputs/customer_segments.png"
)

plt.close()

# 9) Cluster size plot
plt.figure()

cluster_sizes.plot(
    kind="bar"
)

plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
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
print(f"Income    : {x_one_df.iloc[0]['Annual Income (k$)']}")
print(f"Spending  : {x_one_df.iloc[0]['Spending Score (1-100)']}")
print(f"Cluster   : {cluster}")
