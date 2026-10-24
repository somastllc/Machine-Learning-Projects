import pandas as pd

CSV_PATH = (
    "./dataset/"
    "Smartphone_Usage_Productivity_Dataset_50000.csv"
)

df = pd.read_csv(CSV_PATH)

# 1) Initial snapshot
print("\n=== INITIAL HEAD ===")
print(df.head())

print("\n=== INITIAL INFO ===")
df.info()

print("\n=== INITIAL MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLICATES ===")
print(df.duplicated().sum())

# 2) Remove duplicates
df = df.drop_duplicates()

# 3) Final checks
print("\n=== CLEAN DATASET ===")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

# 4) Save clean dataset
OUT_PATH = "./dataset/clean_mobile_usage.csv"

df.to_csv(
    OUT_PATH,
    index=False
)

print(f"\nSaved clean dataset to: {OUT_PATH}")
