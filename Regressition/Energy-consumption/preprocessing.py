import pandas as pd

CSV_PATH = "./dataset/Energy_consumption.csv"
df = pd.read_csv(CSV_PATH)

# 1) Initial snapshot
print("\n=== INITIAL HEAD ===")
print(df.head())

print("\n=== INITIAL INFO ===")
df.info()

print("\n=== INITIAL MISSING VALUES ===")
print(df.isnull().sum())

# 2) Remove duplicates
df = df.drop_duplicates()

# 3) Convert Timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# 4) Feature engineering
df["Hour"] = df["Timestamp"].dt.hour

# 5) Encode binary categorical columns
df["HVACUsage"] = df["HVACUsage"].map({
    "Off": 0,
    "On": 1
})

df["LightingUsage"] = df["LightingUsage"].map({
    "Off": 0,
    "On": 1
})

df["Holiday"] = df["Holiday"].map({
    "No": 0,
    "Yes": 1
})

# 6) One-hot encode DayOfWeek
df = pd.get_dummies(
    df,
    columns=["DayOfWeek"],
    dtype=int
)

# 7) Drop original Timestamp
df = df.drop(columns=["Timestamp"])

# 8) Final snapshot
print("\n=== FINAL HEAD ===")
print(df.head())

print("\n=== FINAL INFO ===")
df.info()

print("\n=== FINAL MISSING VALUES ===")
print(df.isnull().sum())

# 9) Save clean dataset
OUT_PATH = "./dataset/clean_energy.csv"
df.to_csv(OUT_PATH, index=False)

print(f"\nSaved clean dataset to: {OUT_PATH}")
