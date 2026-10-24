# ⚽ Football Player Playing-Style Clustering

An end-to-end unsupervised Machine Learning project that discovers similar football players using **K-Means Clustering**, standardized numerical features, cluster profiles, and visual diagnostics.

---

## 📌 Problem Statement & Overview

### The Problem
Football position labels alone do not capture differences in attacking, passing, defensive, and ball-carrying contributions. Grouping players by statistical similarity provides another way to explore playing styles across squads and competitions.

### The Machine Learning Solution
This project reads the semicolon-separated football dataset, removes duplicates, and keeps players with at least 450 minutes. Missing Nation values are filled with Unknown and missing clustering features with their medians. Fifteen numerical features are standardized before fitting **K-Means Clustering** with three clusters. Position labels are reserved for interpretation.

---

## ⚙️ Pipeline Architecture

```text
    ┌────────────────────────────────────────────────────────────────┐
    │ dataset/2022-2023 Football Player Stats.csv                    │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ preprocessing.py                                               │
    │ * Read semicolon / Latin-1 CSV                                 │
    │ * Drop duplicates; keep Min >= 450                             │
    │ * Fill missing Nation and features                             │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ dataset/clean_football_players.csv                             │
    └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │ train.py                                                       │
    │ * StandardScaler on clustering features                        │
    │ * Fit KMeans (k=3, n_init=20, random_state=42)                 │
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

- **Raw dataset:** `dataset/2022-2023 Football Player Stats.csv` (2,689 records in the supplied file).
- **Clean dataset:** `dataset/clean_football_players.csv` (1,632 records in the supplied file).

### Input Features

| Feature | Description | Values / Units |
| :--- | :--- | :--- |
| `Shots` | Shots | Numerical |
| `SoT` | Shots on target | Numerical |
| `Assists` | Assists | Numerical |
| `PasAss` | Passes leading to a shot | Numerical |
| `PasProg` | Progressive passes | Numerical |
| `SCA` | Shot-creating actions | Numerical |
| `Tkl` | Tackles | Numerical |
| `Int` | Interceptions | Numerical |
| `Blocks` | Blocks | Numerical |
| `Clr` | Clearances | Numerical |
| `ToAtt` | Take-ons attempted | Numerical |
| `ToSuc` | Successful take-ons | Numerical |
| `CarProg` | Progressive carries | Numerical |
| `RecProg` | Progressive passes received | Numerical |
| `AerWon` | Aerial duels won | Numerical |

### Metadata & Interpretation

- Player, Nation, Pos, Squad, Comp, Age and Min: metadata excluded from training.
- Pos: normalized position distribution by cluster.
- Player, Pos, Squad and Comp: example players printed for each cluster.

### Cluster Assignment

- **`Cluster`**: an unsupervised group identifier from `0` to `2`.
- No target labels are used during training. Cluster IDs are arbitrary; interpret each group using its feature means and metadata distributions.

---

## 🤖 Models Trained

1. **K-Means Clustering**
   - Fits **3 clusters** using `n_clusters=3`, `n_init=20`, and `random_state=42`.
   - Uses **15 numerical features** standardized with `StandardScaler`.
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
- `outputs/player_clusters.png`: Two-dimensional PCA projection of the standardized playing-style features.
- `outputs/position_distribution.csv`: Normalized position distribution for each cluster.

---

## 💻 How to Run

Run the commands from `football_player_clustering/`. Create and activate the shared environment once from `Clustering/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd football_player_clustering
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
football_player_clustering/
├── dataset/
│   ├── 2022-2023 Football Player Stats.csv  # Raw dataset
│   └── clean_football_players.csv  # Cleaned dataset
├── models/  # Saved binaries (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── kmeans.pkl
│   └── scaler.pkl
├── outputs/  # Diagnostic profiles and plots
│   ├── cluster_profiles.csv
│   ├── cluster_sizes.png
│   ├── elbow_method.png
│   ├── player_clusters.png
│   └── position_distribution.csv
├── preprocessing.py  # Data cleaning script
├── train.py          # Scaling and model training
├── evaluate.py       # Metrics and visual diagnostics
├── predict.py        # Sample cluster assignment
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```
