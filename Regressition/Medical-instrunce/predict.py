import joblib
import pandas as pd

MODEL_PATH = './models/linear_regression_model.pkl'
model = joblib.load(MODEL_PATH)

# Example new person
age = 30
sex = 'male'
bmi = 23.5
children = 3
smoker = "yes"
region = 'southwest'

# Encode input the same way as preprocessing.py
sex_value = 1 if sex == 'male' else 0
smoker_value = 1 if smoker == 'yes' else 0

new_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_value],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker_value],
    'bmi_smoker': [bmi * smoker_value],
    'region_northwest': [1 if region == 'northwest' else 0],
    'region_southeast': [1 if region == 'southeast' else 0],
    'region_southwest': [1 if region == 'southwest' else 0]
})

prediction = model.predict(new_data)[0]
print(f'Predicted insurance cost: ${prediction:,.2f}')
