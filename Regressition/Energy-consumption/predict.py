import joblib
import pandas as pd

#Electricity price: replace these example values with your actual tariff. 
price_per_kwh = 0.50
currency = "USD"

# 1) Load saved objects
lr = joblib.load(
    "./models/linear_regression.pkl"
)

rf = joblib.load(
    "./models/random_forest.pkl"
)

scaler = joblib.load(
    "./models/scaler.pkl"
)

feature_columns = joblib.load(
    "./models/feature_columns.pkl"
)

# 2) New energy usage data
temperature = 25.0
humidity = 45.0
square_footage = 1500.0
occupancy = 5
hvac_usage = "On"
lighting_usage = "On"
renewable_energy = 15.0
day_of_week = "Monday"
holiday = "No"
hour = 14

hvac_value = 1 if hvac_usage == "On" else 0
lighting_value = 1 if lighting_usage == "On" else 0
holiday_value = 1 if holiday == "Yes" else 0

# 3) Create new row
new_data = pd.DataFrame({
    "Temperature": [temperature],
    "Humidity": [humidity],
    "SquareFootage": [square_footage],
    "Occupancy": [occupancy],
    "HVACUsage": [hvac_value],
    "LightingUsage": [lighting_value],
    "RenewableEnergy": [renewable_energy],
    "Holiday": [holiday_value],
    "Hour": [hour],
    f"DayOfWeek_{day_of_week}": [1]
})

new_data = new_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# 4) Prepare Linear Regression input
scale_cols = [
    "Temperature",
    "Humidity",
    "SquareFootage",
    "Occupancy",
    "RenewableEnergy",
    "Hour"
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

print("\n=== ENERGY CONSUMPTION PREDICTION ===")
print(f"Linear Regression: {lr_prediction:.2f} kWh")
print(f"Random Forest    : {rf_prediction:.2f} kWh")

# 6) Estimate usage cost for the same period as the energy prediction.
lr_cost = lr_prediction * price_per_kwh
rf_cost = rf_prediction * price_per_kwh

print("\n=== ESTIMATED ELECTRICITY COST ===")
print(f"Price per kWh    : {price_per_kwh} {currency}")
print(f"Linear Regression: {lr_cost:.2f} {currency}")
print(f"Random Forest    : {rf_cost:.2f} {currency}")
print("Assumes energy is in kWh; excludes fixed fees and taxes.")
