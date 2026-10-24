# 🛍️ Mall Customer Segmentation

An end-to-end unsupervised Machine Learning project that discovers similar customers using **K-Means Clustering**, standardized numerical features, cluster profiles, and visual diagnostics.

---

## 📌 Problem Statement & Overview

### The Problem
Customer groups with similar incomes can have very different spending habits. Grouping customers by both annual income and spending score helps describe these differences and supports more focused marketing analysis.

### The Machine Learning Solution
This project removes duplicate customer records and drops CustomerID, standardizes annual income and spending score, and fits **K-Means Clustering** with five clusters. Age and Gender are retained for interpreting customer groups but are excluded from training.

---

## ⚙️ Pipeline Architecture

```text
    ┌────────────────────────────────────────────────────────────────┐
    │ dataset/Mall_Customers.csv                                     │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ preprocessing.py                                               │
    │ * Drop duplicate customer records                              │
    │ * Remove CustomerID                                            │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ dataset/clean_mall_customers.csv                               │
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

- **Raw dataset:** `dataset/Mall_Customers.csv` (200 records in the supplied file).
- **Clean dataset:** `dataset/clean_mall_customers.csv` (200 records in the supplied file).

### Input Features

| Feature | Description | Values / Units |
| :--- | :--- | :--- |
| `Annual Income (k$)` | Annual customer income | Thousands of dollars |
| `Spending Score (1-100)` | Customer spending score | Score from 1 to 100 |

### Metadata & Interpretation

- Age: mean age for each cluster.
- Gender: normalized distribution per cluster.

### Cluster Assignment

- **`Cluster`**: an unsupervised group identifier from `0` to `4`.
- No target labels are used during training. Cluster IDs are arbitrary; interpret each group using its feature means and metadata distributions.

---

## 🤖 Models Trained

1. **K-Means Clustering**
   - Fits **5 clusters** using `n_clusters=5`, `n_init=10`, and `random_state=42`.
   - Uses **2 numerical features** standardized with `StandardScaler`.
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

Metrics are computed on the standardized cleaned dataset used for training.

The script also prints cluster sizes, mean profiles, metadata summaries, and a single-row prediction sanity check. These diagnostics describe the fitted groups; they are not supervised classification accuracy.

### Diagnostic Artifacts (Saved in `outputs/`)

- `outputs/cluster_profiles.csv`: Mean training features by cluster.
- `outputs/cluster_sizes.png`: Number of records assigned to each cluster.
- `outputs/elbow_method.png`: Inertia for candidate cluster counts from 2 through 8.
- `outputs/customer_segments.png`: Income and spending scatter plot with cluster centers.

---

## 💻 How to Run

Run the commands from `mall_customer_segmentation/`. Create and activate the shared environment once from `Clustering/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd mall_customer_segmentation
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
mall_customer_segmentation/
├── dataset/
│   ├── Mall_Customers.csv  # Raw dataset
│   └── clean_mall_customers.csv  # Cleaned dataset
├── models/  # Saved binaries (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── kmeans.pkl
│   └── scaler.pkl
├── outputs/  # Diagnostic profiles and plots
│   ├── cluster_profiles.csv
│   ├── cluster_sizes.png
│   ├── elbow_method.png
│   └── customer_segments.png
├── preprocessing.py  # Data cleaning script
├── train.py          # Scaling and model training
├── evaluate.py       # Metrics and visual diagnostics
├── predict.py        # Sample cluster assignment
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```
