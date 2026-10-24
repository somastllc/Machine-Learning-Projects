import numpy as np
import pandas as pd

CSV_PATH = "./dataset/time_series_data_human_activities.csv"
df = pd.read_csv(CSV_PATH)

# 1) Initial snapshot
print("\n=== INITIAL HEAD ===")
print(df.head())

print("\n=== INITIAL INFO ===")
df.info()

print("\n=== INITIAL MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLICATES ===")
print(df.duplicated().sum())

# 2) Remove duplicates
df = df.drop_duplicates()

# 3) Create sensor magnitude
df["magnitude"] = np.sqrt(
    df["x-axis"] ** 2
    + df["y-axis"] ** 2
    + df["z-axis"] ** 2
)

# 4) Sort readings by user and time
df = df.sort_values(
    ["user", "timestamp"]
).reset_index(drop=True)

# 5) Create windows of 100 sensor readings
WINDOW_SIZE = 100

df["window_id"] = (
    df.groupby("user")
    .cumcount()
    // WINDOW_SIZE
)

# 6) Create window-level features
windows = df.groupby(
    ["user", "window_id"]
).agg(
    x_mean=("x-axis", "mean"),
    x_std=("x-axis", "std"),
    y_mean=("y-axis", "mean"),
    y_std=("y-axis", "std"),
    z_mean=("z-axis", "mean"),
    z_std=("z-axis", "std"),
    magnitude_mean=("magnitude", "mean"),
    magnitude_std=("magnitude", "std"),
    n_readings=("activity", "size")
).reset_index()

# 7) Keep the majority activity only for later interpretation
# The activity label is NOT used to train the clustering model.
activity_counts = (
    df.groupby(
        ["user", "window_id", "activity"]
    )
    .size()
    .rename("count")
    .reset_index()
)

largest = activity_counts.groupby(
    ["user", "window_id"]
)["count"].idxmax()

majority_activity = activity_counts.loc[
    largest,
    ["user", "window_id", "activity"]
]

windows = windows.merge(
    majority_activity,
    on=["user", "window_id"],
    how="left"
)

# 8) Remove very short final windows
windows = windows[
    windows["n_readings"] >= 50
].drop(
    columns=["n_readings"]
)

# 9) Final checks
print("\n=== CLEAN WINDOW DATA ===")
print(windows.head())

print("\nShape:")
print(windows.shape)

print("\nMissing values:")
print(windows.isnull().sum())

# 10) Save clean dataset
OUT_PATH = "./dataset/clean_human_activity_windows.csv"
windows.to_csv(
    OUT_PATH,
    index=False
)

print(f"\nSaved clean dataset to: {OUT_PATH}")
