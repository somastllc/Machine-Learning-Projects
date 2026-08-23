
import pandas as pd
import numpy as np
import joblib

CSV_PATH = "./dataset/used_cars.csv"
df = pd.read_csv(CSV_PATH)

# 1) Initial snapshot
print("\n=== INITIAL HEAD ===")
print(df.head())

print("\n=== INITIAL INFO ===")
df.info()

print("\n=== INITIAL MISSING VALUES ===")
print(df.isnull().sum())

# 2) Clean price and mileage
df["Price"] = (
    df["price"]
    .replace(r"[\$,]", "", regex=True)
    .astype(float)
)

df["Mileage"] = (
    df["milage"]
    .replace(r"[^\d.]", "", regex=True)
    .astype(float)
)

# 3) Fix missing categorical values
df["fuel_type"] = df["fuel_type"].replace(
    {"–": pd.NA, "not supported": pd.NA}
)

df["fuel_type"] = df["fuel_type"].fillna("Unknown")
df["accident"] = df["accident"].fillna("Unknown")
df["clean_title"] = df["clean_title"].fillna("Unknown")

# 4) Remove duplicates
df = df.drop_duplicates()

# 5) Remove extreme price outliers
price_limit = df["Price"].quantile(0.99)
df = df[df["Price"] <= price_limit].copy()

# 6) Feature engineering
CURRENT_YEAR = 2026

df["CarAge"] = CURRENT_YEAR - df["model_year"]

df["MileagePerYear"] = (
    df["Mileage"] / df["CarAge"].clip(lower=1)
)

df["Horsepower"] = (
    df["engine"]
    .str.extract(r"(\d+(?:\.\d+)?)HP", expand=False)
    .astype(float)
)

df["Engine_Liters"] = (
    df["engine"]
    .str.extract(r"(\d+(?:\.\d+)?)L", expand=False)
    .astype(float)
)

df["Horsepower"] = df["Horsepower"].fillna(
    df["Horsepower"].median()
)

df["Engine_Liters"] = df["Engine_Liters"].fillna(
    df["Engine_Liters"].median()
)

# 7) Simplify transmission
def clean_transmission(value):
    value = str(value).lower()

    if "cvt" in value:
        return "CVT"
    elif "m/t" in value or "manual" in value:
        return "Manual"
    elif (
        "a/t" in value
        or "automatic" in value
        or "dual shift" in value
    ):
        return "Automatic"
    else:
        return "Other"

df["Transmission_Type"] = df["transmission"].apply(
    clean_transmission
)

# 8) Group rare car models
model_counts = df["model"].value_counts()
common_models = model_counts[model_counts >= 5].index.tolist()

df["Model_Group"] = df["model"].where(
    df["model"].isin(common_models),
    "Other"
)

joblib.dump(
    common_models,
    "./models/common_models.pkl"
)

# 9) Keep useful columns
df = df[
    [
        "brand",
        "Model_Group",
        "fuel_type",
        "accident",
        "clean_title",
        "Transmission_Type",
        "Mileage",
        "Horsepower",
        "Engine_Liters",
        "CarAge",
        "MileagePerYear",
        "Price",
    ]
]

# 10) One-hot encode categorical columns
categorical_cols = [
    "brand",
    "Model_Group",
    "fuel_type",
    "accident",
    "clean_title",
    "Transmission_Type",
]

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    dtype=int
)

# 11) Final snapshot
print("\n=== FINAL HEAD ===")
print(df.head())

print("\n=== FINAL INFO ===")
df.info()

print("\n=== FINAL MISSING VALUES ===")
print(df.isnull().sum().sum())

# 12) Save clean dataset
OUT_PATH = "./dataset/clean_used_cars.csv"
df.to_csv(OUT_PATH, index=False)

print(f"\nSaved clean dataset to: {OUT_PATH}")
