# 📱 Mobile Usage Behavior Clustering

An end-to-end unsupervised Machine Learning project that discovers similar smartphone users using **K-Means Clustering**, standardized numerical features, cluster profiles, and visual diagnostics.

---

## 📌 Problem Statement & Overview

### The Problem
Smartphone users differ in daily phone time, social media use, app usage, and weekend screen time. Discovering groups with similar usage patterns makes these differences easier to inspect alongside lifestyle and device information.

### The Machine Learning Solution
This project removes duplicate records, standardizes four numerical usage features, and fits **K-Means Clustering** with five clusters. Lifestyle and device attributes are retained for interpreting the resulting groups. Preprocessing reports missing values but does not impute them.

---

## ⚙️ Pipeline Architecture

```text
    ┌────────────────────────────────────────────────────────────────┐
    │ dataset/Smartphone_Usage_Productivity_Dataset_50000.csv        │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ preprocessing.py                                               │
    │ * Inspect missing values                                       │
    │ * Remove duplicate records                                     │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ dataset/clean_mobile_usage.csv                                 │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ train.py                                                       │
    │ * StandardScaler on clustering features                        │
    │ * Fit KMeans (k=5, n_init=10, random_state=42)                 │
    │ * Save model, scaler and feature columns                       │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ evaluate.py                         predict.py                 │
    │ * Silhouette and Davies-Bouldin      * Align sample features   │
    │ * Elbow curve and cluster plots      * Scale sample input      │
    │ * Cluster profiles and sizes         * Predict cluster ID      │
    └────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

- **Raw dataset:** `dataset/Smartphone_Usage_Productivity_Dataset_50000.csv` (50,000 records in the supplied file).
- **Clean dataset:** `dataset/clean_mobile_usage.csv` (50,000 records in the supplied file).

### Input Features

| Feature | Description | Values / Units |
| :--- | :--- | :--- |
| `Daily_Phone_Hours` | Daily phone usage duration | Hours |
| `Social_Media_Hours` | Social media usage duration | Hours |
| `App_Usage_Count` | App usage count | Count |
| `Weekend_Screen_Time_Hours` | Weekend screen time | Hours |

### Metadata & Interpretation

- Age, Work_Productivity_Score, Sleep_Hours, Stress_Level and Caffeine_Intake_Cups: mean lifestyle profiles.
- Device_Type: normalized device distribution.
- User_ID: identifier for the single-row sanity check.

### Cluster Assignment

- **`Cluster`**: an unsupervised group identifier from `0` to `4`.
- No target labels are used during training. Cluster IDs are arbitrary; interpret each group using its feature means and metadata distributions.

---

## 🤖 Models Trained

1. **K-Means Clustering**
   - Fits **5 clusters** using `n_clusters=5`, `n_init=10`, and `random_state=42`.
   - Uses **4 numerical features** standardized with `StandardScaler`.
   - Saves `kmeans.pkl`, `scaler.pkl`, and `feature_columns.pkl` in `models/`.
   - Assigns a new sample by restoring the saved feature order and applying the fitted scaler before prediction.

The training script uses a fixed cluster count. The evaluation script compares inertia for candidate values of `k` from 2 through 8; it does not automatically change the trained model.

---

## 📈 Evaluation & Diagnostics

Cluster quality is evaluated using unsupervised metrics:

| Evaluation Metric | Description | Interpretation |
| :--- | :--- | :--- |
| **Inertia** | Sum of squared distances to the assigned cluster centers | Compare across cluster counts on the same scaled data |
| **Silhouette Score** | Cohesion within clusters relative to separation between clusters | Higher generally indicates better separation |
| **Davies-Bouldin Score** | Within-cluster scatter relative to separation between cluster centers | Lower generally indicates better separation |

Silhouette is computed on a reproducible sample of up to **10,000 records** (`random_state=42`); Davies-Bouldin uses all records. PCA visualization uses the same sample.

The script also prints cluster sizes, mean profiles, metadata summaries, and a single-row prediction sanity check. These diagnostics describe the fitted groups; they are not supervised classification accuracy.

### Diagnostic Artifacts (Saved in `outputs/`)

- `outputs/cluster_profiles.csv`: Mean training features by cluster.
- `outputs/cluster_sizes.png`: Number of records assigned to each cluster.
- `outputs/elbow_method.png`: Inertia for candidate cluster counts from 2 through 8.
- `outputs/pca_clusters.png`: Two-dimensional PCA projection of the sampled standardized usage features.

---

## 💻 How to Run

Run the commands from `mobile_usage_clustering/`. Create and activate the shared environment once from `Clustering/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd mobile_usage_clustering
```

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Create directories for generated artifacts after cloning
New-Item -ItemType Directory -Force models, outputs | Out-Null

# 3. Clean the raw dataset
python preprocessing.py

# 4. Scale features, train K-Means, and export model binaries
python train.py

# 5. Evaluate clustering metrics and generate diagnostic plots
python evaluate.py

# 6. Assign a cluster to the example input
python predict.py
```

If the environment is already active, navigate directly into the project folder. On macOS/Linux, activate with `source .venv/bin/activate` and create the artifact folders with `mkdir -p models outputs`. Edit the sample in `predict.py` to use a different input.

---

## 📁 Directory Structure

```text
mobile_usage_clustering/
├── dataset/
│   ├── Smartphone_Usage_Productivity_Dataset_50000.csv  # Raw dataset
│   └── clean_mobile_usage.csv  # Cleaned dataset
├── models/  # Saved binaries (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── kmeans.pkl
│   └── scaler.pkl
├── outputs/  # Diagnostic profiles and plots
│   ├── cluster_profiles.csv
│   ├── cluster_sizes.png
│   ├── elbow_method.png
│   └── pca_clusters.png
├── preprocessing.py  # Data cleaning script
├── train.py          # Scaling and model training
├── evaluate.py       # Metrics and visual diagnostics
├── predict.py        # Sample cluster assignment
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```
