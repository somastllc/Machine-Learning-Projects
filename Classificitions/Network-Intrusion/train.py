import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "./dataset/clean_train.csv"
df = pd.read_csv(DATA_PATH)

# 1) Define features and target
X = df.drop(columns=["class"])
y = df["class"]

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3) Select numerical columns for scaling
scale_cols = [
    col for col in X.columns
    if not col.startswith(
        ("protocol_type_", "service_", "flag_")
    )
]

scaler = StandardScaler()

X_train_lr = X_train.copy()
X_test_lr = X_test.copy()

X_train_lr[scale_cols] = scaler.fit_transform(
    X_train[scale_cols]
)

X_test_lr[scale_cols] = scaler.transform(
    X_test[scale_cols]
)

# 4) Traiogistic Regressionn
lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train_lr, y_train)

# 5) Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# 6) Save models and preprocessing objects
joblib.dump(
    lr,
    "./models/logistic_regression.pkl"
)

joblib.dump(
    rf,
    "./models/random_forest.pkl"
)

joblib.dump(
    scaler,
    "./models/scaler.pkl"
)

joblib.dump(
    scale_cols,
    "./models/scale_columns.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "./models/feature_columns.pkl"
)

print("Training complete.")
print("Saved Logistic Regression, Random Forest, scaler, and feature columns.")
