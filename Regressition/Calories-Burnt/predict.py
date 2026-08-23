import joblib
import pandas as pd

# 1) Load models and scaler
lr = joblib.load("./models/linear_regression.pkl")
rf = joblib.load("./models/random_forest.pkl")
scaler = joblib.load("./models/scaler.pkl")

# 2) New person/activity data
gender = "male"
age = 30
height = 175.0
weight = 75.0
duration = 20.0
heart_rate = 100.0
body_temp = 40.0

gender_value = 1 if gender.lower() == "male" else 0
bmi = weight / ((height / 100) ** 2)

new_data = pd.DataFrame({
    "Gender": [gender_value],
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
    "Duration": [duration],
    "Heart_Rate": [heart_rate],
    "Body_Temp": [body_temp],
    "BMI": [bmi]
})

# 3) Prepare data for Linear Regression
scale_cols = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "BMI"
]

new_data_lr = new_data.copy()
new_data_lr[scale_cols] = scaler.transform(new_data[scale_cols])

# 4) Predict
lr_prediction = float(lr.predict(new_data_lr)[0])
rf_prediction = float(rf.predict(new_data)[0])

print("\n=== CALORIES PREDICTION ===")
print(f"Linear Regression: {lr_prediction:.2f} calories")
print(f"Random Forest    : {rf_prediction:.2f} calories")
