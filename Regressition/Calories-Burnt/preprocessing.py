import pandas as pd

CALORIES_PATH = "./dataset/calories.csv"
EXERCISE_PATH = "./dataset/exercise.csv"

calories = pd.read_csv(CALORIES_PATH)
exercise = pd.read_csv(EXERCISE_PATH)

# 1) Initial snapshot
print("\n=== CALORIES HEAD ===")
print(calories.head())

print("\n=== EXERCISE HEAD ===")
print(exercise.head())

print("\n=== MISSING VALUES ===")
print("Calories:")
print(calories.isnull().sum())
print("\nExercise:")
print(exercise.isnull().sum())

# 2) Merge datasets
df = exercise.merge(
    calories,
    on="User_ID",
    how="inner"
)

# 3) Remove duplicates
df = df.drop_duplicates()

# 4) Drop identifier
df = df.drop(columns=["User_ID"])

# 5) Encode Gender
df["Gender"] = df["Gender"].map({
    "female": 0,
    "male": 1
})

# 6) Feature engineering
df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

# 7) Final snapshot
print("\n=== FINAL HEAD ===")
print(df.head())

print("\n=== FINAL INFO ===")
df.info()

print("\n=== FINAL MISSING VALUES ===")
print(df.isnull().sum())

# 8) Save clean dataset
OUT_PATH = "./dataset/clean_calories.csv"
df.to_csv(OUT_PATH, index=False)

print(f"\nSaved clean dataset to: {OUT_PATH}")
