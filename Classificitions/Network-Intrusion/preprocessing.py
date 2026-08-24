import pandas as pd

TRAIN_PATH = "./dataset/Train_data.csv"
TEST_PATH = "./dataset/Test_data.csv"

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
print(train["class"].value_counts())

# 2) Remove duplicates from training data
train = train.drop_duplicates()

# 3) Encode target
train["class"] = train["class"].map({
    "normal": 0,
    "anomaly": 1
})

# 4) One-hot encode categorical columns
categorical_cols = [
    "protocol_type",
    "service",
    "flag"
]

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

# 5) Make test features match training features
feature_cols = [
    col for col in train.columns
    if col != "class"
]

test = test.reindex(
    columns=feature_cols,
    fill_value=0
)

# 6) Final snapshot
print("\n=== FINAL TRAIN SHAPE ===")
print(train.shape)

print("\n=== FINAL TEST SHAPE ===")
print(test.shape)

print("\n=== FINAL MISSING VALUES ===")
print("Train:", train.isnull().sum().sum())
print("Test :", test.isnull().sum().sum())

# 7) Save clean datasets
train.to_csv(
    "./dataset/clean_train.csv",
    index=False
)

test.to_csv(
    "./dataset/clean_test.csv",
    index=False
)

print("\nSaved clean_train.csv and clean_test.csv")
