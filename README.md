# Machine Learning Projects

Nine practical machine learning projects by **Somast**, covering regression, classification, and clustering. Each project takes you from preparing a dataset to training a model, evaluating its results, and making predictions.

Course link: Coming soon.

## Projects

Explore three projects in each area. The project names link to detailed documentation with datasets, features, models, and instructions for running the code.

### Regression

Predict numerical values from exercise, building, and insurance data.

| # | Project | Goal | What you'll practice |
| --- | --- | --- | --- |
| 01 | [Calories Burnt](Regressition/Calories-Burnt/README.md) | Estimate calories burned during exercise. | Derive BMI, scale numerical features, and compare Linear Regression with Random Forest. |
| 02 | [Energy Consumption](Regressition/Energy-consumption/README.md) | Predict building energy consumption. | Extract time features, encode building operations, and compare regression models. |
| 03 | [Medical Insurance](Regressition/Medical-instrunce/README.md) | Estimate medical insurance charges. | Encode categories and model the interaction between BMI and smoking status. |

### Classification

Predict categories from network traffic, student records, and clinical measurements.

| # | Project | Goal | What you'll practice |
| --- | --- | --- | --- |
| 04 | [Network Intrusion](Classificitions/Network-Intrusion/README.md) | Classify network connections as normal or anomalous. | Align encoded training and test features, compare classifiers, and evaluate detection performance. |
| 05 | [Student Dropout Risk](Classificitions/student_dropout_classification/README.md) | Predict student dropout risk. | Prepare academic and demographic features, scale inputs, and compare Logistic Regression with Random Forest. |
| 06 | [Heart Disease Risk](Classificitions/Heart-disease/README.md) | Classify heart disease risk from clinical data. | Encode categorical measurements and assess classifiers with metrics and confusion matrices. |

### Clustering

Discover groups of similar customers, football players, and smartphone users.

| # | Project | Goal | What you'll practice |
| --- | --- | --- | --- |
| 07 | [Mall Customer Segmentation](Clustering/mall_customer_segmentation/README.md) | Group customers by income and spending patterns. | Scale features, fit K-Means, explore the elbow curve, and interpret customer segments. |
| 08 | [Football Player Clustering](Clustering/football_player_clustering/README.md) | Group players by statistical playing style. | Prepare 15 playing-style features, fit K-Means, and visualize groups with PCA. |
| 09 | [Mobile Usage Clustering](Clustering/mobile_usage_clustering/README.md) | Group smartphone users by usage behavior. | Cluster four usage features, evaluate separation, and interpret cluster profiles. |

## What you'll learn

- Clean datasets, handle missing values where needed, and prepare features for modeling.
- Encode categorical data, scale numerical inputs, and create useful feature interactions.
- Train and compare regression and classification models.
- Explore K-Means clustering through profiles, evaluation scores, and visualizations.
- Save models and preprocessing objects with `joblib`, then reuse them for predictions.

## Getting started

Choose a project from the tables above and open a terminal in the repository folder. Each project has its own dependencies and runs from its own directory.

For example, to run Network Intrusion on Windows with PowerShell:

```powershell
cd Classificitions/Network-Intrusion

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force models, outputs | Out-Null

.\.venv\Scripts\python.exe preprocessing.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe predict.py
```

On macOS or Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe` and `mkdir -p models outputs` to create the output folders.

Run the scripts in this order from the selected project folder so that dataset and model paths resolve correctly. The same sequence applies to all nine projects; see each project's README for its inputs and outputs.

## Project structure

The projects share a common layout:

| File or folder | Purpose |
| --- | --- |
| `dataset/` | Source datasets and cleaned data produced by preprocessing. |
| `preprocessing.py` | Data cleaning, encoding, and feature preparation. |
| `train.py` | Model training and saving models and preprocessing objects. |
| `evaluate.py` | Evaluation metrics, plots, and diagnostic summaries. |
| `predict.py` | Predictions or cluster assignments using saved models. |
| `models/` | Generated model files and preprocessing objects, excluded from Git. |
| `outputs/` | Evaluation plots, reports, and other generated results. |
| `requirements.txt` | Dependencies for the selected project. |
| `README.md` | Project-specific explanation and usage instructions. |

## License

Copyright (c) 2026 Somast. Licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](LICENSE).
