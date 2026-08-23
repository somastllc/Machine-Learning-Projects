import pandas as pd

CSV_PATH = "./dataset/heart.csv"
df = pd.read_csv(CSV_PATH)

# 1) Initial snapshot
print("\n=== INITIAL HEAD ===")
print(df.head())

print("\n=== INITIAL INFO ===")
df.info()

print("\n=== INITIAL MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== TARGET DISTRIBUTION ===")
print(df["target"].value_counts())

# 2) Remove duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f"\nDuplicates removed: {before - after}")

# 3) Encode categorical columns
categorical_cols = [
    "cp",
    "restecg",
    "slope",
    "thal"
]

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    dtype=int
)

# 4) Final snapshot
print("\n=== FINAL HEAD ===")
print(df.head())

print("\n=== FINAL MISSING VALUES ===")
print(df.isnull().sum().sum())

print("\n=== FINAL TARGET DISTRIBUTION ===")
print(df["target"].value_counts())

# 5) Save clean dataset
OUT_PATH = "./dataset/clean_heart.csv"
df.to_csv(OUT_PATH, index=False)

print(f"\nSaved clean dataset to: {OUT_PATH}")
