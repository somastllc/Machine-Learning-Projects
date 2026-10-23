import pandas as pd

CSV_PATH = "./dataset/student_dropout_dataset_v3.csv"
df = pd.read_csv(CSV_PATH)

# 1) Initial snapshot
print("\n=== INITIAL HEAD ===")
print(df.head())

print("\n=== INITIAL INFO ===")
df.info()

print("\n=== INITIAL MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== TARGET DISTRIBUTION ===")
print(df["Dropout"].value_counts())

# 2) Remove duplicates
df = df.drop_duplicates()

# 3) Drop identifier
df = df.drop(columns=["Student_ID"])

# 4) Fill missing numerical values
numeric_missing = [
    "Family_Income",
    "Study_Hours_per_Day",
    "Stress_Index"
]

for col in numeric_missing:
    df[col] = df[col].fillna(
        df[col].median()
    )

# 5) Fill missing categorical value
df["Parental_Education"] = df[
    "Parental_Education"
].fillna(
    df["Parental_Education"].mode()[0]
)

# 6) Encode binary categorical columns
df["Gender"] = df["Gender"].map({
    "Female": 0,
    "Male": 1
})

df["Internet_Access"] = df["Internet_Access"].map({
    "No": 0,
    "Yes": 1
})

df["Part_Time_Job"] = df["Part_Time_Job"].map({
    "No": 0,
    "Yes": 1
})

df["Scholarship"] = df["Scholarship"].map({
    "No": 0,
    "Yes": 1
})

# 7) One-hot encode categorical columns
categorical_cols = [
    "Semester",
    "Department",
    "Parental_Education"
]

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    dtype=int
)

# 8) Final snapshot
print("\n=== FINAL HEAD ===")
print(df.head())

print("\n=== FINAL MISSING VALUES ===")
print(df.isnull().sum().sum())

print("\n=== FINAL TARGET DISTRIBUTION ===")
print(df["Dropout"].value_counts())

# 9) Save clean dataset
OUT_PATH = "./dataset/clean_student_dropout.csv"
df.to_csv(OUT_PATH, index=False)

print(f"\nSaved clean dataset to: {OUT_PATH}")
