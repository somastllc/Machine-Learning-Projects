import joblib
import pandas as pd

# 1) Load saved objects
lr = joblib.load(
    "./models/logistic_regression.pkl"
)

rf = joblib.load(
    "./models/random_forest.pkl"
)

scaler = joblib.load(
    "./models/scaler.pkl"
)

scale_cols = joblib.load(
    "./models/scale_columns.pkl"
)

feature_columns = joblib.load(
    "./models/feature_columns.pkl"
)

# 2) New student data
new_data = pd.DataFrame([{
    "Age": 21.0,
    "Gender": "Male",
    "Family_Income": 45000,
    "Internet_Access": "Yes",
    "Study_Hours_per_Day": 3.5,
    "Attendance_Rate": 80.0,
    "Assignment_Delay_Days": 2,
    "Travel_Time_Minutes": 25.0,
    "Part_Time_Job": "No",
    "Scholarship": "Yes",
    "Stress_Index": 5.0,
    "GPA": 3.0,
    "Semester_GPA": 3.1,
    "CGPA": 3.0,
    "Semester": "Year 2",
    "Department": "CS",
    "Parental_Education": "Bachelor"
}])

# 3) Encode binary categorical columns
new_data["Gender"] = new_data["Gender"].map({
    "Female": 0,
    "Male": 1
})

new_data["Internet_Access"] = new_data["Internet_Access"].map({
    "No": 0,
    "Yes": 1
})

new_data["Part_Time_Job"] = new_data["Part_Time_Job"].map({
    "No": 0,
    "Yes": 1
})

new_data["Scholarship"] = new_data["Scholarship"].map({
    "No": 0,
    "Yes": 1
})

# 4) One-hot encode categorical columns
categorical_cols = [
    "Semester",
    "Department",
    "Parental_Education"
]

new_data = pd.get_dummies(
    new_data,
    columns=categorical_cols,
    dtype=int
)

new_data = new_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# 5) Prepare Logistic Regression input
new_data_lr = new_data.copy()

new_data_lr[scale_cols] = scaler.transform(
    new_data[scale_cols]
)

# 6) Predict
lr_prediction = int(
    lr.predict(new_data_lr)[0]
)

rf_prediction = int(
    rf.predict(new_data)[0]
)

lr_probability = float(
    lr.predict_proba(new_data_lr)[0][1]
)

rf_probability = float(
    rf.predict_proba(new_data)[0][1]
)

print("\n=== STUDENT DROPOUT PREDICTION ===")
print(
    f"Logistic Regression: {lr_prediction} "
    f"({lr_probability:.2%})"
)

print(
    f"Random Forest      : {rf_prediction} "
    f"({rf_probability:.2%})"
)
