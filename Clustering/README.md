# Clustering Projects

Three unsupervised machine learning projects using StandardScaler and K-Means, with preprocessing, model persistence, cluster evaluation, and sample inference.

| Project | Training features | Clusters | Documentation |
| --- | --- | --- | --- |
| Mall Customer Segmentation | Annual income and spending score | 5 | [README](mall_customer_segmentation/README.md) |
| Football Player Clustering | 15 playing-style statistics | 3 | [README](football_player_clustering/README.md) |
| Mobile Usage Clustering | Phone time, social media, app count, weekend screen time | 5 | [README](mobile_usage_clustering/README.md) |

## Setup & Run

From `Clustering`, create one environment for all three projects:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

cd mall_customer_segmentation
New-Item -ItemType Directory -Force models, outputs | Out-Null
../.venv/Scripts/python.exe preprocessing.py
../.venv/Scripts/python.exe train.py
../.venv/Scripts/python.exe evaluate.py
../.venv/Scripts/python.exe predict.py
```

Use either of the other project directories to run its pipeline. Direct interpreter paths avoid requiring PowerShell activation. For optional activation, run `.\.venv\Scripts\Activate.ps1` from `Clustering`. On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

## Evaluation

Each project reports inertia, silhouette score, Davies-Bouldin score, cluster sizes, and mean profiles. Plots show the elbow curve and cluster distribution. Mall customers use an income/spending scatter plot; football and mobile usage use PCA. Mobile silhouette evaluation samples up to 10,000 records.

Cluster labels describe similarity, not a supervised prediction target. Model files and environments are excluded from Git; datasets and diagnostic outputs are kept. Recreate `models/` and train after cloning.

[Back to the repository overview](../README.md)
