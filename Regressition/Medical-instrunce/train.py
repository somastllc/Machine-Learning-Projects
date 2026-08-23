import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

DATA_PATH = './dataset/clean_insurance.csv'
MODEL_PATH = './models/linear_regression_model.pkl'

# 1) Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# 2) Define features and target
X = df.drop(columns='charges')
y = df['charges']

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4) Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5) Save model
os.makedirs('./models', exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f'Model saved to: {MODEL_PATH}')
print(f'Training samples: {len(X_train)}')
print(f'Testing samples: {len(X_test)}')
