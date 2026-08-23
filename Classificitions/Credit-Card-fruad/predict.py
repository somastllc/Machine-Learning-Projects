import joblib
import pandas as pd

# 1) Load saved objects
lr = joblib.load('./models/logistic_regression.pkl')
rf = joblib.load('./models/random_forest.pkl')
scaler = joblib.load('./models/scaler.pkl')
feature_columns = joblib.load('./models/feature_columns.pkl')

# 2) Example transaction
new_data = pd.DataFrame([{
    'Time': 0.0,
    'V1': -1.359807,
    'V2': -0.072781,
    'V3': 2.536347,
    'V4': 1.378155,
    'V5': -0.338321,
    'V6': 0.462388,
    'V7': 0.239599,
    'V8': 0.098698,
    'V9': 0.363787,
    'V10': 0.090794,
    'V11': -0.551600,
    'V12': -0.617801,
    'V13': -0.991390,
    'V14': -0.311169,
    'V15': 1.468177,
    'V16': -0.470401,
    'V17': 0.207971,
    'V18': 0.025791,
    'V19': 0.403993,
    'V20': 0.251412,
    'V21': -0.018307,
    'V22': 0.277838,
    'V23': -0.110474,
    'V24': 0.066928,
    'V25': 0.128539,
    'V26': -0.189115,
    'V27': 0.133558,
    'V28': -0.021053,
    'Amount': 149.62
}])

new_data = new_data.reindex(columns=feature_columns, fill_value=0)

# 3) Prepare Logistic Regression input
scale_cols = ['Time', 'Amount']
new_data_lr = new_data.copy()
new_data_lr[scale_cols] = scaler.transform(new_data[scale_cols])

# 4) Predict
lr_prediction = int(lr.predict(new_data_lr)[0])
rf_prediction = int(rf.predict(new_data)[0])

lr_probability = float(lr.predict_proba(new_data_lr)[0][1])
rf_probability = float(rf.predict_proba(new_data)[0][1])

print('\n=== CREDIT CARD FRAUD PREDICTION ===')
print(f'Logistic Regression: {lr_prediction} ({lr_probability:.2%})')
print(f'Random Forest      : {rf_prediction} ({rf_probability:.2%})')
