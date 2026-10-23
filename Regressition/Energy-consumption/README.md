# ⚡ Building Energy Consumption Prediction Pipeline

An end-to-end Machine Learning regression project that predicts building energy consumption based on environmental factors, building physical attributes, operational usage, renewable energy supply, and temporal schedule patterns.

---

## 📌 Problem Statement & Overview

### The Problem
Energy grid management and building energy efficiency require accurate forecasts of power usage. Static estimation rules fail to capture the complex interaction between fluctuating external temperatures, occupant counts, heating/ventilation/air-conditioning (HVAC) demand, lighting schedules, and renewable energy generation. Inaccurate predictions lead to energy waste, higher utility costs, and grid instability.

### The Machine Learning Solution
This project implements a complete regression modeling pipeline to forecast `EnergyConsumption`. By preprocessing time-series timestamps into hourly and daily features, encoding operational binary switches (HVAC and Lighting), and scaling environmental metrics, we evaluate both **Linear Regression** (with `StandardScaler`) and **Random Forest Regressor** to model non-linear energy consumption patterns.

---

## ⚙️ Pipeline Architecture

```
                 ┌────────────────────────────────┐
                 │ dataset/Energy_consumption.csv │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │        preprocessing.py        │
                 │ • Drop Duplicates              │
                 │ • Parse Timestamp to Hour      │
                 │ • Binary Map (HVAC, Lighting)  │
                 │ • One-Hot Encode DayOfWeek     │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │      dataset/clean_energy.csv  │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │           train.py             │
                 │ • 80/20 Train-Test Split       │
                 │ • StandardScaler (LR Features) │
                 │ • Fit Linear Regression & RF   │
                 │ • Save *.pkl Models & Scaler   │
                 └───────────────┬────────────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
┌───────────────────────────────┐                 ┌───────────────────────────────┐
│          evaluate.py          │                 │          predict.py           │
│ • Evaluate MAE, MSE, RMSE, R² │                 │ • Run Single-Sample           │
│ • Actual vs Predicted Plot    │                 │   Energy Inference            │
│ • Residual Diagnostics Plot   │                 └───────────────────────────────┘
└───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

The dataset contains environmental, structural, operational, and temporal observations collected from building monitoring systems.

### Input Features

| Feature | Type | Scale / Range | Description |
| :--- | :--- | :--- | :--- |
| `Temperature` | Numerical | °C / °F | Ambient temperature reading |
| `Humidity` | Numerical | % | Relative environmental humidity |
| `SquareFootage` | Numerical | Sq Ft | Total operational floor area |
| `Occupancy` | Numerical | Integer | Number of active occupants |
| `HVACUsage` | Binary | `0` (Off), `1` (On) | HVAC system activation state |
| `LightingUsage` | Binary | `0` (Off), `1` (On) | Interior lighting activation state |
| `RenewableEnergy` | Numerical | kWh / % | On-site renewable energy contribution |
| `Holiday` | Binary | `0` (No), `1` (Yes) | Holiday calendar indicator |
| `Hour` | Numerical | `0` to `23` | Hour of day extracted from timestamp |
| `DayOfWeek` | Categorical (Encoded) | `DayOfWeek_*` | Dummy indicators for Monday–Sunday |

### Target Variable
- **`EnergyConsumption`**: Continuous measure of building power consumed.

---

## 🤖 Models Trained

1. **Linear Regression**
   - Scaled numerical features (`Temperature`, `Humidity`, `SquareFootage`, `Occupancy`, `RenewableEnergy`, `Hour`) using `StandardScaler`.
   - Serves as an interpretable linear baseline model.

2. **Random Forest Regressor**
   - Ensemble of 100 decision trees (`n_estimators=100`, `random_state=42`).
   - Captures non-linear interactions between temperature, HVAC demand, and occupant load.

---

## 📈 Evaluation & Diagnostics

Models are evaluated on an 80/20 test split using standard regression metrics:
- **MAE (Mean Absolute Error)**: Average absolute prediction deviation.
- **MSE (Mean Squared Error)**: Mean squared prediction residual.
- **RMSE (Root Mean Squared Error)**: Standard error in target consumption units.
- **R² Score**: Proportion of energy consumption variance explained by model features.

### Visualization Outputs (Saved in `outputs/`)
- `actual_vs_predicted.png`: Scatter plot comparing ground truth energy vs. Random Forest predictions.
- `residuals.png`: Residual distribution plot inspecting error variance homoscedasticity.

---

## 💻 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data & engineer timestamp features
python preprocessing.py

# 3. Train Linear Regression & Random Forest models
python train.py

# 4. Evaluate metrics & generate diagnostic plots
python evaluate.py

# 5. Run inference on custom building scenario
python predict.py
```

Set `price_per_kwh` and `currency` at the top of `predict.py` to your actual
tariff. The default `0.50 USD` is only an example.
The estimate multiplies each model's predicted energy by this price;
it assumes `EnergyConsumption` is measured in kWh and covers the same period as
the prediction. Confirm the dataset's units before using the estimate. Fixed
fees, taxes, and tiered or time-dependent tariffs are not included.
