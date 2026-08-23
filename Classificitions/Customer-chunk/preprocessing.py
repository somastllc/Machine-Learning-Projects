import pandas as pd

TRAIN_PATH = "./dataset/train.csv"
TEST_PATH = "./dataset/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# 1) Initial snapshot
print("\n=== TRAIN HEAD ===")
print(train.head())

print("\n=== TRAIN INFO ===")
train.info()

print("\n=== TRAIN MISSING VALUES ===")
print(train.isnull().sum())

print("\n=== TARGET DISTRIBUTION ===")
print(train["Exited"].value_counts())

# 2) Remove duplicates
train = train.drop_duplicates()
test = test.drop_duplicates()

# 3) Drop identifiers / high-cardinality names
drop_cols = ["id", "CustomerId", "Surname"]

train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)

# 4) Encode categorical columns
categorical_cols = ["Geography", "Gender"]

train = pd.get_dummies(
    train,
    columns=categorical_cols,
    dtype=int
)

test = pd.get_dummies(
    test,
    columns=categorical_cols,
    dtype=int
)

# 5) Make train and test feature columns match
target = "Exited"

feature_cols = [
    col for col in train.columns
    if col != target
]

test = test.reindex(
    columns=feature_cols,
    fill_value=0
)

# 6) Convert target to integer
train[target] = train[target].astype(int)

# 7) Final snapshot
print("\n=== FINAL TRAIN HEAD ===")
print(train.head())

print("\n=== FINAL MISSING VALUES ===")
print("Train:", train.isnull().sum().sum())
print("Test :", test.isnull().sum().sum())

# 8) Save cleaned datasets
train.to_csv(
    "./dataset/clean_train.csv",
    index=False
)

test.to_csv(
    "./dataset/clean_test.csv",
    index=False
)

print("\nSaved clean_train.csv and clean_test.csv")
