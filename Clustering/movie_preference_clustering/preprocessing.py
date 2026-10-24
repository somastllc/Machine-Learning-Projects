import pandas as pd

# ==========================================
# 1) LOAD MOVIELENS FILES
# ==========================================

movies = pd.read_csv(
    "./dataset/movies.dat",
    sep="::",
    engine="python",
    names=[
        "MovieID",
        "Title",
        "Genres"
    ],
    encoding="latin-1"
)

ratings = pd.read_csv(
    "./dataset/ratings.dat",
    sep="::",
    engine="python",
    names=[
        "UserID",
        "MovieID",
        "Rating",
        "Timestamp"
    ],
    encoding="latin-1"
)

users = pd.read_csv(
    "./dataset/users.dat",
    sep="::",
    engine="python",
    names=[
        "UserID",
        "Gender",
        "Age",
        "Occupation",
        "ZipCode"
    ],
    encoding="latin-1"
)

print("\n=== MOVIES ===")
print(movies.head())

print("\n=== RATINGS ===")
print(ratings.head())

print("\n=== USERS ===")
print(users.head())

print("\n=== SHAPES ===")
print("Movies :", movies.shape)
print("Ratings:", ratings.shape)
print("Users  :", users.shape)

print("\n=== MISSING VALUES ===")
print("Movies :", movies.isnull().sum().sum())
print("Ratings:", ratings.isnull().sum().sum())
print("Users  :", users.isnull().sum().sum())

# ==========================================
# 2) CREATE GENRE COLUMNS
# ==========================================

genre_columns = (
    movies["Genres"]
    .str.get_dummies("|")
)

movies_with_genres = pd.concat(
    [
        movies[["MovieID", "Title"]],
        genre_columns
    ],
    axis=1
)

genres = genre_columns.columns.tolist()

# ==========================================
# 3) MERGE RATINGS WITH MOVIE GENRES
# ==========================================

data = ratings.merge(
    movies_with_genres,
    on="MovieID",
    how="left"
)

# ==========================================
# 4) USER MOVIE COUNTS
# ==========================================

user_rating_count = (
    ratings.groupby("UserID")
    .size()
    .rename("Rating_Count")
)

# ==========================================
# 5) GENRE WATCH COUNTS
# ==========================================

genre_counts = (
    data.groupby("UserID")[genres]
    .sum()
)

genre_share = genre_counts.div(
    user_rating_count,
    axis=0
)

# ==========================================
# 6) AVERAGE RATING PER GENRE
# ==========================================

for genre in genres:
    data[f"{genre}_rating_sum"] = (
        data[genre] * data["Rating"]
    )

rating_sum_columns = [
    f"{genre}_rating_sum"
    for genre in genres
]

genre_rating_sums = (
    data.groupby("UserID")[
        rating_sum_columns
    ]
    .sum()
)

genre_rating_sums.columns = genres

genre_avg_ratings = (
    genre_rating_sums
    / genre_counts.replace(0, pd.NA)
)

# If a user never watched a genre,
# use their overall average rating.
overall_avg_rating = (
    ratings.groupby("UserID")["Rating"]
    .mean()
)

for genre in genres:
    genre_avg_ratings[genre] = (
        genre_avg_ratings[genre]
        .fillna(overall_avg_rating)
    )

# ==========================================
# 7) CREATE PREFERENCE SCORE
# ==========================================

# Preference score combines:
# 1) how often the user watches a genre
# 2) how highly the user rates that genre
#
# Score = watch share * average rating / 5

preference_scores = (
    genre_share
    * (genre_avg_ratings / 5.0)
)

preference_scores.columns = [
    f"Preference_{genre}"
    for genre in genres
]

# ==========================================
# 8) CREATE CLEAN USER PROFILE DATASET
# ==========================================

user_profiles = preference_scores.copy()

user_profiles.insert(
    0,
    "Rating_Count",
    user_rating_count
)

user_profiles.insert(
    0,
    "UserID",
    user_profiles.index
)

user_profiles = user_profiles.reset_index(
    drop=True
)

# Add user metadata only for interpretation.
# These columns are NOT used to train KMeans.
user_profiles = user_profiles.merge(
    users,
    on="UserID",
    how="left"
)

print("\n=== CLEAN USER PROFILES ===")
print(user_profiles.head())

print("\nShape:")
print(user_profiles.shape)

print("\nMissing values:")
print(user_profiles.isnull().sum().sum())

# ==========================================
# 9) SAVE CLEAN DATASET
# ==========================================

OUT_PATH = (
    "./dataset/"
    "clean_movie_preference_profiles.csv"
)

user_profiles.to_csv(
    OUT_PATH,
    index=False
)

print(
    f"\nSaved clean dataset to: {OUT_PATH}"
)
