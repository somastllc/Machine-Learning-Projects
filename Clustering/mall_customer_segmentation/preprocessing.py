import pandas as pd

CSV_PATH = "./dataset/Mall_Customers.csv"
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

# 3) Drop identifier
df = df.drop(columns=["CustomerID"])

# 4) Final checks
print("\n=== CLEAN DATASET ===")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

# 5) Save clean dataset
OUT_PATH = "./dataset/clean_mall_customers.csv"

df.to_csv(
    OUT_PATH,
    index=False
)

print(f"\nSaved clean dataset to: {OUT_PATH}")
