# 🎬 Movie Preference & User Taste Clustering

An end-to-end unsupervised Machine Learning clustering project that segments streaming platform movie watchers into distinct viewing taste cohorts using the MovieLens dataset, custom genre preference score feature engineering, and K-Means clustering.

---

## 📌 Problem Statement & Overview

### The Problem
Modern digital media streaming platforms (such as Netflix, Hulu, or Amazon Prime Video) host vast catalogs of movies rated by diverse user populations. To personalize home screens, recommend niche titles, and solve cold-start onboarding, platforms cannot treat every user identically. Users must be segmented into cohesive behavioral taste clusters based on their demonstrated historical viewing frequency and genre satisfaction.

### The Machine Learning Solution
This project implements an end-to-end recommendation audience clustering system:
1. Ingests relational MovieLens records across movies, user ratings, and user demographics.
2. Extracts 18 individual genres from multi-label movie metadata using one-hot encoding.
3. Computes each user's viewing genre share and weighted average rating per genre.
4. Engineers a composite continuous **Genre Preference Score**:
   $$\text{Preference}_{\text{genre}} = \text{Genre Watch Share} \times \left(\frac{\text{Average Genre Rating}}{5.0}\right)$$
5. Standardizes all 18 genre preference metrics using `StandardScaler`.
6. Fits **K-Means Clustering** ($k=4$) to discover distinct taste profiles.
7. Validates cluster structures using the **Elbow Method**, **Silhouette Score**, **Davies-Bouldin Index**, and **PCA** 2D projections.

---

## ⚙️ Pipeline Architecture

```
        ┌─────────────────────────────────────────────────────────┐
        │ dataset/movies.dat   ratings.dat   users.dat            │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │                    preprocessing.py                     │
        │ • Multi-label Genre One-Hot Encoding (18 genres)        │
        │ • Relational Table Merging (User x Movie x Rating)      │
        │ • Compute User Genre Watch Shares & Average Ratings     │
        │ • Feature Engineer Preference Score = Share * (Avg / 5) │
        │ • Merge User Demographics (Age, Gender, Occupation)     │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │ dataset/clean_movie_preference_profiles.csv             │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │                        train.py                         │
        │ • Extract 18 Preference_* Columns                       │
        │ • Fit StandardScaler & K-Means (k=4, random_state=42)   │
        │ • Export *.pkl Model, Scaler & Columns                  │
        └────────────────────────────┬────────────────────────────┘
                                     │
        ┌────────────────────────────┴────────────────────────────┐
        ▼                                                         ▼
┌───────────────────────────────┐                 ┌───────────────────────────────┐
│          evaluate.py          │                 │          predict.py           │
│ • Silhouette & Davies-Bouldin │                 │ • New User Profile Inference  │
│ • Top Genres & Demographics   │                 │ • Cluster Taste Cohort Match  │
│ • Elbow & PCA 2D Plots        │                 │                               │
└───────────────────────────────┘                 └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

The dataset integrates the relational MovieLens benchmark tables containing ratings from thousands of verified movie watchers.

### Input Preference Features (Used for Clustering)

The model clusters users across 18 engineered continuous genre preference scores:

| Genre Preference Feature | Feature Range | Meaning |
| :--- | :---: | :--- |
| `Preference_Action` | $[0.0, 1.0]$ | Combined affinity for Action films |
| `Preference_Adventure` | $[0.0, 1.0]$ | Combined affinity for Adventure films |
| `Preference_Animation` | $[0.0, 1.0]$ | Combined affinity for Animated titles |
| `Preference_Children's` | $[0.0, 1.0]$ | Combined affinity for Kids & Family content |
| `Preference_Comedy` | $[0.0, 1.0]$ | Combined affinity for Comedy films |
| `Preference_Crime` | $[0.0, 1.0]$ | Combined affinity for Crime & Heist movies |
| `Preference_Documentary` | $[0.0, 1.0]$ | Combined affinity for Non-fiction / Documentaries |
| `Preference_Drama` | $[0.0, 1.0]$ | Combined affinity for Drama & Character-driven films |
| `Preference_Fantasy` | $[0.0, 1.0]$ | Combined affinity for Fantasy titles |
| `Preference_Film-Noir` | $[0.0, 1.0]$ | Combined affinity for Film-Noir classics |
| `Preference_Horror` | $[0.0, 1.0]$ | Combined affinity for Horror & Suspense |
| `Preference_Musical` | $[0.0, 1.0]$ | Combined affinity for Musical & Stage films |
| `Preference_Mystery` | $[0.0, 1.0]$ | Combined affinity for Whodunit / Mystery movies |
| `Preference_Romance` | $[0.0, 1.0]$ | Combined affinity for Romantic comedies & dramas |
| `Preference_Sci-Fi` | $[0.0, 1.0]$ | Combined affinity for Science Fiction titles |
| `Preference_Thriller` | $[0.0, 1.0]$ | Combined affinity for Psychological & Action Thrillers |
| `Preference_War` | $[0.0, 1.0]$ | Combined affinity for Historical & War movies |
| `Preference_Western` | $[0.0, 1.0]$ | Combined affinity for Western cinema |

> [!NOTE]
> Demographic metadata (`Age`, `Gender`, `Occupation`, `ZipCode`) and total `Rating_Count` were **not included during training**. They are evaluated post-clustering to assess how real-world demographics naturally align with movie taste cohorts.

---

## 🤖 Models & Clustering Approach

- **Algorithm**: K-Means Clustering (`n_clusters=4`, `n_init=10`, `random_state=42`)
- **Feature Scaling**: All 18 genre preference scores normalized via `StandardScaler`.
- **Optimal $k$ Selection**: Determined via the **Elbow Method** ($k \in [2, 8]$) and cluster interpretability, producing 4 distinct, commercially actionable audience segments.

---

## 📈 Evaluation & Diagnostics

### Quantitative Clustering Metrics

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Inertia** | `80,621.10` | Compactness of users around genre centroid centers |
| **Silhouette Score** | `0.2139` | Overall geometric separation across multi-dimensional preference space |
| **Davies-Bouldin Score** | `1.6914` | Ratio of within-cluster dispersion to cluster separation distance |

### Discovered Audience Segments (Taste Cohorts)

| Cluster ID | Size (Users) | % Share | Primary Preferred Genres | Demographic Profile | Audience Archetype |
| :---: | :---: | :---: | :--- | :---: | :--- |
| **`Cluster 0`** | 1,489 | 24.6% | **Action** (0.362), **Sci-Fi** (0.253), **Thriller** (0.191) | Avg Age: 29.5 (87.6% Male) | **Blockbuster & Sci-Fi Enthusiasts** |
| **`Cluster 1`** | 3,826 | 63.3% | **Drama** (0.323), **Comedy** (0.290), **Romance** (0.124) | Avg Age: 31.0 (67.7% Male, 32.3% Female) | **Mainstream Drama & Comedy Viewers** |
| **`Cluster 2`** | 345 | 5.7% | **Comedy** (0.299), **Children's** (0.228), **Animation** (0.169), **Musical** (0.144) | Avg Age: 26.5 (53.0% Female) | **Family & Animation Fans** |
| **`Cluster 3`** | 380 | 6.3% | **Drama** (0.326), **Thriller** (0.280), **Crime** (0.159), **Mystery** (0.120) | Avg Age: 34.8 (Oldest demographic) | **Crime, Mystery & Thriller Connoisseurs** |

### Visual Diagnostic Artifacts (Saved in `outputs/`)
- `outputs/elbow_method.png`: Inertia decay curve displaying the elbow trajectory across $k \in [2, 8]$.
- `outputs/pca_clusters.png`: 2D PCA projection visually separating the 4 user preference cohorts.
- `outputs/cluster_sizes.png`: Distribution bar chart showing user counts in each taste cluster.
- `outputs/cluster_profiles.csv`: Full matrix of average genre preference scores across all 4 clusters.

---

## 💻 How to Run

```bash
# 1. Activate project virtual environment & install dependencies
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Extract multi-table features & generate composite preference scores
python preprocessing.py

# 3. Train K-Means clustering model & save model binaries
python train.py

# 4. Evaluate clustering metrics & generate PCA/Elbow diagnostic plots
python evaluate.py

# 5. Predict cluster cohort for a sample user taste profile
python predict.py
```

---

## 📁 Directory Structure

```
movie_preference_clustering/
├── dataset/
│   ├── movies.dat                             # Raw MovieLens movie titles & genres
│   ├── ratings.dat                            # Raw user ratings & timestamps
│   ├── users.dat                              # Raw user demographics
│   ├── README                                 # MovieLens dataset documentation
│   └── clean_movie_preference_profiles.csv    # Engineered user preference dataset
├── models/                                     # Serialized models (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── kmeans.pkl
│   └── scaler.pkl
├── outputs/                                    # Visual plots & tabular summaries
│   ├── cluster_profiles.csv
│   ├── cluster_sizes.png
│   ├── elbow_method.png
│   └── pca_clusters.png
├── preprocessing.py                            # Relational merging & preference score pipeline
├── train.py                                    # K-Means model training & serialization script
├── evaluate.py                                 # Evaluation metrics, PCA & profile generator
├── predict.py                                  # Inference script for new user taste profiles
├── requirements.txt                            # Project package requirements
└── README.md                                   # Comprehensive project documentation
```
