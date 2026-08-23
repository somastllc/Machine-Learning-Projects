import pandas as pd

CSV_PATH = './dataset/creditcard.csv'
df = pd.read_csv(CSV_PATH)

# 1) Initial snapshot
print('\n=== INITIAL HEAD ===')
print(df.head())

print('\n=== INITIAL INFO ===')
df.info()

print('\n=== INITIAL MISSING VALUES ===')
print(df.isnull().sum())

print('\n=== TARGET DISTRIBUTION ===')
print(df['Class'].value_counts())

# 2) Remove exact duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f'\nDuplicates removed: {before - after}')

# 3) Final snapshot
print('\n=== FINAL SHAPE ===')
print(df.shape)

print('\n=== FINAL MISSING VALUES ===')
print(df.isnull().sum().sum())

print('\n=== FINAL TARGET DISTRIBUTION ===')
print(df['Class'].value_counts())

# 4) Save clean dataset
OUT_PATH = './dataset/clean_creditcard.csv'
df.to_csv(OUT_PATH, index=False)
print(f'\nSaved clean dataset to: {OUT_PATH}')
