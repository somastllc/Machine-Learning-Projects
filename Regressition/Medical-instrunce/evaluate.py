import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = './dataset/clean_insurance.csv'
MODEL_PATH = './models/linear_regression_model.pkl'

# 1) Load data and model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# 2) Define features and target
X = df.drop(columns='charges')
y = df['charges']

# 3) Recreate the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4) Make predictions
y_pred = model.predict(X_test)

# 5) Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('\n=== MODEL EVALUATION ===')
print(f'MAE:  {mae:.2f}')
print(f'MSE:  {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R²:   {r2:.4f}')

# 6) Single-row sanity check
# Pick one unseen row from X_test
# Use iloc[[i]] to keep it as a DataFrame with column names
i = 3
x_one_df = X_test.iloc[[i]]
y_true = y_test.iloc[i]

p_one = float(model.predict(x_one_df)[0])

print('\n=== SINGLE-ROW SANITY CHECK ===')
print(f'Actual Charges: ${y_true:,.2f}')
print(f'LR Prediction : ${p_one:,.2f}')

# 7) Actual vs Predicted plot
os.makedirs('./outputs', exist_ok=True)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted')
plt.savefig('./outputs/actual_vs_predicted.png', bbox_inches='tight')
plt.close()

# 8) Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0)
plt.xlabel('Predicted Charges')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('./outputs/residuals.png', bbox_inches='tight')
plt.close()

print('\nPlots saved in: ./outputs/')
