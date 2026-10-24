# 🏃 Human Activity Recognition & Sensor Clustering

An unsupervised Machine Learning clustering project that discovers natural groupings of physical human activities (ambulatory movement, vigorous motion, and stationary postures) from raw continuous tri-axial accelerometer time-series sensor data.

---

## 📌 Problem Statement & Overview

### The Problem
Wearable Internet-of-Things (IoT) devices, fitness trackers, and modern smartphones generate continuous streams of 3-axis accelerometer readings. In many real-world environments, ground-truth activity labels are unavailable or expensive to acquire. An unsupervised clustering system must autonomously aggregate sensor signals across temporal windows and segment continuous bodily motion into distinct behavioral states without relying on ground-truth activity labels.

### The Machine Learning Solution
This project implements a complete end-to-end unsupervised pipeline:
1. Calculates instantaneous 3D sensor Euclidean acceleration magnitude ($\sqrt{x^2 + y^2 + z^2}$).
2. Segments continuous time-series into fixed sliding windows of 100 consecutive readings.
3. Derives window-level statistical features (mean and standard deviation for axes and magnitude).
4. Standardizes window features via `StandardScaler`.
5. Fits a **K-Means Clustering** model ($k=3$) to uncover distinct motion intensity clusters.
6. Evaluates cluster quality using **Elbow Method**, **Silhouette Score**, **Davies-Bouldin Index**, and **Principal Component Analysis (PCA)** 2D projections.

---

## ⚙️ Pipeline Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │ dataset/time_series_data_human_activities.csv│
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │               preprocessing.py               │
                 │ • Deduplicate raw sensor signals             │
                 │ • Compute 3D Acceleration Magnitude          │
                 │ • Rolling Temporal Windowing (100 readings)  │
                 │ • Aggregate Statistics (mean, std per axis)  │
                 │ • Filter boundary windows (< 50 readings)    │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │ dataset/clean_human_activity_windows.csv     │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │                   train.py                   │
                 │ • StandardScaler on Window Features          │
                 │ • Fit KMeans (k=3, random_state=42)          │
                 │ • Export *.pkl Model, Scaler & Columns       │
                 └──────────────────────┬───────────────────────┘
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        ▼                                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│          evaluate.py          │               │          predict.py           │
│ • Silhouette & Davies-Bouldin │               │ • Single Window Sensor State  │
│ • Elbow Method & PCA 2D Plot  │               │   Cluster Assignment          │
│ • Cluster Size Distribution   │               │                               │
└───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

The dataset contains continuous high-frequency accelerometer logs characterized by user IDs, timestamps, and 3-axis readings ($x, y, z$). After temporal windowing, each record represents a 100-reading summary window.

### Window Features (Used for Clustering)

| Feature | Type | Description | Formula / Scope |
| :--- | :--- | :--- | :--- |
| `x_mean` | Numerical | Mean horizontal lateral acceleration | $\mu(x)$ |
| `x_std` | Numerical | Standard deviation of lateral acceleration | $\sigma(x)$ |
| `y_mean` | Numerical | Mean vertical gravitational acceleration | $\mu(y)$ |
| `y_std` | Numerical | Standard deviation of vertical acceleration | $\sigma(y)$ |
| `z_mean` | Numerical | Mean forward/backward acceleration | $\mu(z)$ |
| `z_std` | Numerical | Standard deviation of forward acceleration | $\sigma(z)$ |
| `magnitude_mean` | Numerical | Mean resultant acceleration magnitude | $\mu(\sqrt{x^2 + y^2 + z^2})$ |
| `magnitude_std` | Numerical | Standard deviation of resultant magnitude | $\sigma(\sqrt{x^2 + y^2 + z^2})$ |

> [!NOTE]
> The `activity` label (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing) was **never used during training or clustering**. It was preserved purely as a post-hoc diagnostic ground truth to inspect the physical validity of the unsupervised clusters.

---

## 🤖 Model & Clustering Approach

- **Algorithm**: K-Means Clustering (`n_clusters=3`, `n_init=10`, `random_state=42`)
- **Scaling**: All 8 statistical window features standardized with `StandardScaler`.
- **Optimal $k$ Selection**: Determined via the **Elbow Method** (inertia trajectory flattening at $k=3$) and silhouette maximization across candidate $k \in [2, 8]$.

---

## 📈 Evaluation & Diagnostics

### Quantitative Metrics

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Inertia** | `38,955.61` | Sum of squared distances to closest cluster center |
| **Silhouette Score** | `0.3797` | Cluster cohesion vs. separation across sampled windows |
| **Davies-Bouldin Score** | `1.0946` | Ratio of within-cluster scatter to between-cluster separation (lower is better) |

### Cluster Profiles & Semantic Interpretation

| Cluster ID | Size (Windows) | % Share | Dominant Physical Activity | Mean Magnitude | Mean Std ($\sigma_{\text{mag}}$) | Semantic Profile |
| :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **`Cluster 0`** | 6,375 | 62.5% | **Walking** (60.7%), Upstairs (16.4%), Downstairs (14.9%) | 11.18 | 4.52 | **Moderate Ambulatory Motion** |
| **`Cluster 1`** | 2,719 | 26.7% | **Jogging** (91.0%) | 14.26 | 6.58 | **High-Intensity Dynamic Motion** |
| **`Cluster 2`** | 1,104 | 10.8% | **Sitting** (51.2%), **Standing** (42.2%) | 9.84 | 0.24 | **Sedentary / Stationary Postures** |

### Visual Diagnostic Artifacts (Saved in `outputs/`)
- `outputs/elbow_method.png`: Inertia curve across $k \in [2, 8]$ showing the elbow inflection point at $k=3$.
- `outputs/pca_clusters.png`: 2D Principal Component Analysis scatter plot demonstrating distinct cluster boundaries.
- `outputs/cluster_sizes.png`: Distribution of sample windows assigned across each discovered cluster.

---

## 💻 How to Run

```bash
# 1. Activate project virtual environment & install dependencies
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Extract 100-sample temporal windows & compute aggregate metrics
python preprocessing.py

# 3. Train KMeans clustering model & serialize *.pkl artifacts
python train.py

# 4. Compute Silhouette, Davies-Bouldin scores & export PCA/Elbow plots
python evaluate.py

# 5. Run inference on a sample sensor window
python predict.py
```

---

## 📁 Directory Structure

```
human_activity_clustering/
├── dataset/
│   ├── time_series_data_human_activities.csv  # Raw continuous sensor records
│   └── clean_human_activity_windows.csv       # Preprocessed temporal window records
├── models/                                     # Serialized models (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── kmeans.pkl
│   └── scaler.pkl
├── outputs/                                    # Generated diagnostic figures
│   ├── cluster_sizes.png
│   ├── elbow_method.png
│   └── pca_clusters.png
├── preprocessing.py                            # Window slicing & feature derivation script
├── train.py                                    # K-Means model training & serialization script
├── evaluate.py                                 # Metrics evaluation & visualization generator
├── predict.py                                  # Single-window activity cluster prediction script
├── requirements.txt                            # Project package requirements
└── README.md                                   # Comprehensive project documentation
```
