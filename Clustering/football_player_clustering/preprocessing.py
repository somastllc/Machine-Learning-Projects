import pandas as pd

CSV_PATH = (
    "./dataset/"
    "2022-2023 Football Player Stats.csv"
)

# 1) Load dataset
# The file uses ";" instead of "," as the separator.
df = pd.read_csv(
    CSV_PATH,
    sep=";",
    encoding="latin-1"
)

print("\n=== INITIAL HEAD ===")
print(df.head())

print("\n=== INITIAL INFO ===")
df.info()

print("\n=== INITIAL MISSING VALUES ===")
print(
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n=== DUPLICATES ===")
print(df.duplicated().sum())

# 2) Remove duplicates
df = df.drop_duplicates()

# 3) Keep players with enough playing time
# This reduces noisy stats from players
# who played only a few minutes.
df = df[
    df["Min"] >= 450
].copy()

# 4) Features used for clustering
feature_columns = [
    "Shots",
    "SoT",
    "Assists",
    "PasAss",
    "PasProg",
    "SCA",
    "Tkl",
    "Int",
    "Blocks",
    "Clr",
    "ToAtt",
    "ToSuc",
    "CarProg",
    "RecProg",
    "AerWon"
]

# 5) Keep useful metadata + clustering features
keep_columns = [
    "Player",
    "Nation",
    "Pos",
    "Squad",
    "Comp",
    "Age",
    "Min"
] + feature_columns

df = df[keep_columns]

# 6) Fill the one possible missing metadata value
df["Nation"] = df["Nation"].fillna(
    "Unknown"
)

# 7) Fill missing feature values if any
for col in feature_columns:
    df[col] = df[col].fillna(
        df[col].median()
    )

# 8) Final checks
print("\n=== CLEAN DATASET ===")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

# 9) Save clean dataset
OUT_PATH = (
    "./dataset/"
    "clean_football_players.csv"
)

df.to_csv(
    OUT_PATH,
    index=False
)

print(
    f"\nSaved clean dataset to: {OUT_PATH}"
)
