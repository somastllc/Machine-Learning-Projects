
import joblib
import pandas as pd

# 1) Load saved objects
lr = joblib.load("./models/linear_regression.pkl")
rf = joblib.load("./models/random_forest.pkl")
scaler = joblib.load("./models/scaler.pkl")
feature_columns = joblib.load("./models/feature_columns.pkl")
common_models = joblib.load("./models/common_models.pkl")

# 2) New car information
brand = "Toyota"
model = "Camry"
fuel_type = "Gasoline"
accident = "None reported"
clean_title = "Yes"
transmission_type = "Automatic"

model_year = 2020
mileage = 45000
horsepower = 203
engine_liters = 2.5

CURRENT_YEAR = 2026

car_age = CURRENT_YEAR - model_year
mileage_per_year = mileage / max(car_age, 1)

model_group = (
    model if model in common_models else "Other"
)

# 3) Create new row
new_data = pd.DataFrame(
    {
        "Mileage": [mileage],
        "Horsepower": [horsepower],
        "Engine_Liters": [engine_liters],
        "CarAge": [car_age],
        "MileagePerYear": [mileage_per_year],
        f"brand_{brand}": [1],
        f"Model_Group_{model_group}": [1],
        f"fuel_type_{fuel_type}": [1],
        f"accident_{accident}": [1],
        f"clean_title_{clean_title}": [1],
        f"Transmission_Type_{transmission_type}": [1],
    }
)

new_data = new_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# 4) Prepare Linear Regression input
scale_cols = [
    "Mileage",
    "Horsepower",
    "Engine_Liters",
    "CarAge",
    "MileagePerYear",
]

new_data_lr = new_data.copy()

new_data_lr[scale_cols] = scaler.transform(
    new_data[scale_cols]
)

# 5) Predict
lr_prediction = float(
    lr.predict(new_data_lr)[0]
)

rf_prediction = float(
    rf.predict(new_data)[0]
)

print("\n=== USED CAR PRICE PREDICTION ===")
print(f"Linear Regression: ${lr_prediction:,.2f}")
print(f"Random Forest    : ${rf_prediction:,.2f}")
