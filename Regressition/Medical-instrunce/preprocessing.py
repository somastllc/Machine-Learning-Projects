import pandas as pd

CSV_PATH = './dataset/insurance.csv'
OUT_PATH = './dataset/clean_insurance.csv'

# 1) Load dataset
df = pd.read_csv(CSV_PATH)

# 2) Initial snapshot
print('\n=== INITIAL HEAD ===')
print(df.head())

print('\n=== INITIAL INFO ===')
df.info()

print('\n=== INITIAL MISSING VALUES ===')
print(df.isnull().sum())

# 3) Remove duplicates
df = df.drop_duplicates()

# 4) Encode binary categorical columns
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

# 5) Feature engineering
# Smoking and BMI together have a strong relationship with insurance charges.
df['bmi_smoker'] = df['bmi'] * df['smoker']

# 6) One-hot encode region
df = pd.get_dummies(df, columns=['region'], drop_first=True, dtype=int)

# 7) Final snapshot
print('\n=== FINAL HEAD ===')
print(df.head())

print('\n=== FINAL INFO ===')
df.info()

print('\n=== FINAL MISSING VALUES ===')
print(df.isnull().sum())

# 8) Save clean dataset
df.to_csv(OUT_PATH, index=False)
print(f'\nSaved cleaned dataset to: {OUT_PATH}')
