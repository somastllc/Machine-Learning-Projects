# 🛍️ Online Shopper Intent & Behavioral Clustering

An unsupervised Machine Learning clustering project that segments e-commerce web session traffic into distinct shopper intent cohorts (disengaged bouncers, casual window shoppers, and high-intent buyers) using web analytics behavioral telemetry and K-Means clustering.

---

## 📌 Problem Statement & Overview

### The Problem
E-commerce retail platforms experience thousands of digital shopping visits every hour, each with vastly differing levels of purchasing intent. Some visitors bounce immediately after viewing a landing page, others casually explore catalogs, while a critical subset actively evaluates products, shipping policies, and administrative account details with immediate purchase readiness. Real-time marketing personalization, exit-intent incentives, and recommendation engines require automated behavioral clustering to distinguish high-value shoppers from disengaged visitors without invasive surveys.

### The Machine Learning Solution
This project builds an end-to-end unsupervised audience segmentation workflow:
1. Ingests 12,330 real-world e-commerce browsing sessions from Google Analytics.
2. Deduplicates session records and cleans behavioral telemetry.
3. Selects 10 continuous session features capturing page interactions, durations, bounce rates, exit rates, and page value metrics.
4. Normalizes behavioral distributions using `StandardScaler`.
5. Fits **K-Means Clustering** ($k=3$) to identify clear shopper intent tiers.
6. Evaluates cluster cohesion via the **Elbow Method**, **Silhouette Score**, **Davies-Bouldin Index**, and **PCA** 2D projections.
7. Post-hoc validates commercial efficacy against downstream session conversion (`Revenue`: False / True) and visitor loyalty (`VisitorType`).

---

## ⚙️ Pipeline Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │ dataset/online_shoppers_intention.csv        │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │               preprocessing.py               │
                 │ • Inspect session attributes & missingness   │
                 │ • Drop duplicate web browsing sessions       │
                 │ • Export cleaned baseline dataset            │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │ dataset/clean_online_shopping.csv            │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │                   train.py                   │
                 │ • Select 10 core behavioral web metrics      │
                 │ • Fit StandardScaler & K-Means (k=3)         │
                 │ • Export *.pkl Models, Scaler & Columns      │
                 └──────────────────────┬───────────────────────┘
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        ▼                                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│          evaluate.py          │               │          predict.py           │
│ • Silhouette & Davies-Bouldin │               │ • Session Telemetry Inference │
│ • Purchase Rate by Cluster    │               │ • Buyer Intent Tier Match     │
│ • Elbow & PCA 2D Visuals      │               │                               │
└───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

The dataset captures session telemetry across 12,330 web visits.

### Input Behavioral Features (Used for Clustering)

| Feature | Type | Description | Metric Range / Meaning |
| :--- | :--- | :--- | :--- |
| `Administrative` | Numerical | Number of account/administrative pages viewed | Discrete count |
| `Administrative_Duration` | Numerical | Total time spent on administrative pages | Seconds |
| `Informational` | Numerical | Number of informational / FAQ / policy pages visited | Discrete count |
| `Informational_Duration` | Numerical | Total time spent on informational pages | Seconds |
| `ProductRelated` | Numerical | Number of product catalog pages viewed | Discrete count |
| `ProductRelated_Duration` | Numerical | Total duration exploring product items | Seconds |
| `BounceRates` | Numerical | Average bounce rate of pages entered by visitor | Ratio ($0.0 - 0.20$) |
| `ExitRates` | Numerical | Average exit rate of pages viewed during session | Ratio ($0.0 - 0.20$) |
| `PageValues` | Numerical | Google Analytics average page e-commerce value | Monetary metric |
| `SpecialDay` | Numerical | Closeness of site visiting date to a specific holiday | Scale ($0.0 - 1.0$) |

> [!NOTE]
> The final conversion indicator (`Revenue`: False / True) was **excluded during training**. It serves as an objective external validation benchmark to verify whether the unsupervised clusters truly differentiate purchasing likelihood.

---

## 🤖 Models & Clustering Approach

- **Algorithm**: K-Means Clustering (`n_clusters=3`, `n_init=10`, `random_state=42`)
- **Feature Standardization**: Continuous features transformed using `StandardScaler` to balance durations, counts, and page values.
- **Optimal $k$ Selection**: Selected using the **Elbow Method** ($k \in [2, 8]$) and maximizing business interpretability across conversion tiers.

---

## 📈 Evaluation & Diagnostics

### Quantitative Clustering Metrics

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Inertia** | `80,037.67` | Within-cluster sum of squared distances to centroids |
| **Silhouette Score** | `0.4534` | High cluster cohesion and well-delineated decision boundaries |
| **Davies-Bouldin Score** | `1.2609` | Low intra-cluster spread relative to inter-cluster separation |

### Discovered Shopper Segments (Behavioral Cohorts)

| Cluster ID | Size (Sessions) | % Share | Mean Admin Pages | Mean Product Duration | Mean Page Value | Actual Purchase Rate | Shopper Persona |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Cluster 0`** | 932 | 7.6% | 0.06 pages | 1.6 sec | 0.00 | **0.6%** | **Bouncers / Disengaged Visitors** |
| **`Cluster 1`** | 9,647 | 79.0% | 1.71 pages | 51.2 sec | 5.89 | **15.0%** | **Casual Browsers / Window Shoppers** |
| **`Cluster 2`** | 1,626 | 13.3% | 7.38 pages | 308.0 sec | 9.73 | **27.8%** | **High-Intent / Transaction-Ready Buyers** |

### Commercial Insights
- **Cluster 2 (High-Intent Buyers)** demonstrates deep engagement: nearly 5 minutes exploring administrative and account workflows, highest product engagement, and nearly **doubles the baseline conversion rate (27.8%)**.
- **Cluster 0 (Bouncers)** represents sessions with negligible dwell time and virtually zero conversion (0.6%). Platforms can use this insight to trigger exit-intent engagement or investigate broken landing experiences.

### Visual Diagnostic Artifacts (Saved in `outputs/`)
- `outputs/elbow_method.png`: Inertia decay curve demonstrating the elbow inflection at $k=3$.
- `outputs/pca_clusters.png`: 2D PCA projection visually confirming distinct spatial separation between clusters.
- `outputs/cluster_sizes.png`: Session distribution across the 3 behavioral tiers.
- `outputs/purchase_rate_by_cluster.png`: Purchase conversion contrast validating cluster predictive value.
- `outputs/cluster_profiles.csv`: Full mean matrix for all 10 features across clusters.

---

## 💻 How to Run

```bash
# 1. Activate project virtual environment & install dependencies
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Deduplicate session data & create clean dataset
python preprocessing.py

# 3. Train K-Means clustering model & save model binaries
python train.py

# 4. Evaluate metrics, calculate purchase conversion rates & render plots
python evaluate.py

# 5. Predict shopper intent tier for a new web session
python predict.py
```

---

## 📁 Directory Structure

```
online_shopping_behavior_clustering/
├── dataset/
│   ├── online_shoppers_intention.csv         # Raw Google Analytics session data (12,330 visits)
│   └── clean_online_shopping.csv             # Deduplicated session dataset
├── models/                                   # Serialized model binaries (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── kmeans.pkl
│   └── scaler.pkl
├── outputs/                                  # Generated visual and tabular diagnostics
│   ├── cluster_profiles.csv
│   ├── cluster_sizes.png
│   ├── elbow_method.png
│   ├── pca_clusters.png
│   └── purchase_rate_by_cluster.png
├── preprocessing.py                          # Session cleaning & deduplication script
├── train.py                                  # K-Means clustering training script
├── evaluate.py                               # Metrics evaluation & visualization generator
├── predict.py                                # Single-session shopper intent prediction script
├── requirements.txt                          # Project package requirements
└── README.md                                 # Comprehensive project documentation
```
