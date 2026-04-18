# Columns: https://github.com/devensinghbhagtani/Bollywood-Movie-Dataset.git   (minimum 5 questions)
# - Movie ID: Unique identifier for each movie
# - Name: Title of the movie
# - Year: Release year
# - Genre: Categories or types of the movie
# - Overview: Brief synopsis or plot summary
# - Director: Name(s) of the director(s)
# - Cast: Lead actors and actresses

# visulization, simple linear regression, eda, and 5 questions(3 from visualizatoins,1 from eda, 1 from simple linear regression)

# ==========================================
# IMDB MOVIE DATASET — FINAL CLEAN CODE
# ==========================================

# ==========================================
# IMDB MOVIE DATASET PROJECT — FINAL CODE
# ==========================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.linear_model import LinearRegression
from scipy import stats

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv("IMDB-Movie-Dataset(2023-1951).csv")
print("Dataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())


print("\n--- BEFORE CLEANING ---")
print("Shape:", df.shape)
print("\nMissing Values:\n")
print(df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nSample Data:")
print(df.head())

# ==========================================
# 2. DATA CLEANING
# ==========================================

# Remove unnecessary column
df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

# Convert year to numeric
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Keep valid years
df = df[df["year"].isna() | df["year"].between(1900, 2024)] #it explicitly keeps missing values(df["year"].isna()) and valid years(df["year"].between(1900, 2024))

# Check duplicates BEFORE list conversion
print("\nDuplicates BEFORE removal:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Split genres
df["genre_list"] = df["genre"].str.split(", ")
df_genres = df.explode("genre_list")

# ==========================================
# AFTER CLEANING
# ==========================================
print("\n--- AFTER CLEANING ---")
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSample Cleaned Data:")
print(df.head())

# ==========================================
# 3. HISTOGRAM
# ==========================================
year_vals = df["year"].dropna().astype(int)

plt.figure(figsize=(10, 5))
plt.hist(year_vals, bins=30, edgecolor="black")
plt.title("Distribution of Movies by Release Year")
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.show()

# ==========================================
# 4. SCATTER + TREND LINE
# ==========================================
movies_per_year = df.dropna(subset=["year"]).groupby("year").size().reset_index(name="count")

plt.figure(figsize=(10, 5))
plt.scatter(movies_per_year["year"], movies_per_year["count"])

m, b = np.polyfit(movies_per_year["year"], movies_per_year["count"], 1)
plt.plot(movies_per_year["year"], m * movies_per_year["year"] + b, linestyle="--")

plt.title("Year vs Number of Movies Released")
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.show()

# ==========================================
# 5. BOX PLOT
# ==========================================
plt.figure(figsize=(6, 5))
plt.boxplot(year_vals)
plt.title("Box Plot of Movie Release Years")
plt.ylabel("Year")
plt.show()

# ==========================================
# 6. Q1 — TOP GENRES
# ==========================================
genre_counts = df_genres["genre_list"].value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.barh(genre_counts.index[::-1], genre_counts.values[::-1])
plt.title("Top 10 Genres")
plt.xlabel("Count")
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(genre_counts.values, labels=genre_counts.index, autopct="%1.1f%%")
plt.title("Genre Share")
plt.show()

# ==========================================
# 7. Q2 — MOVIES PER DECADE
# ==========================================
df_year = df.dropna(subset=["year"]).copy()
df_year["decade"] = (df_year["year"] // 10) * 10

decade_counts = df_year.groupby("decade").size()

plt.figure(figsize=(10, 5))
decade_counts.plot(kind="bar")
plt.title("Movies per Decade")
plt.xlabel("Decade")
plt.ylabel("Number of Movies")
plt.show()

# ==========================================
# 8. Q3 — TOP DIRECTORS
# ==========================================
top_directors = df["director"].value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_directors.index[::-1], top_directors.values[::-1])
plt.title("Top Directors")
plt.xlabel("Movies Directed")
plt.show()

# ==========================================
# 9. Q4 — GENRE COMBINATIONS
# ==========================================
def genre_pairs(genre_str):
    genres = [g.strip() for g in str(genre_str).split(",")]
    if len(genres) < 2:
        return []
    return list(combinations(genres, 2))

pair_series = df["genre"].apply(genre_pairs).explode().dropna()
pair_counts = pair_series.value_counts().head(10)

labels = [f"{a} + {b}" for a, b in pair_counts.index]

plt.figure(figsize=(10, 6))
plt.barh(labels[::-1], pair_counts.values[::-1])
plt.title("Top Genre Combinations")
plt.xlabel("Count")
plt.show()

# Heatmap
top8 = genre_counts.index.tolist()
co_matrix = pd.DataFrame(0, index=top8, columns=top8)

for _, row in df.iterrows():
    genres = [g.strip() for g in str(row["genre"]).split(",")]
    for a, b in combinations(genres, 2):
        if a in top8 and b in top8:
            co_matrix.loc[a, b] += 1
            co_matrix.loc[b, a] += 1

sns.heatmap(co_matrix, annot=True, cmap="Blues")
plt.title("Genre Heatmap")
plt.show()

# ==========================================
# 10. Q5 — TOP ACTORS
# ==========================================
df["actor"] = df["cast"].str.split(", ")
df_cast = df.explode("actor")

actor_counts = df_cast["actor"].value_counts().head(12)

plt.figure(figsize=(10, 6))
plt.barh(actor_counts.index[::-1], actor_counts.values[::-1])
plt.title("Top Actors")
plt.xlabel("Number of Movies")
plt.show()

# ==========================================
# 11. LINEAR REGRESSION
# ==========================================
reg_data = df_year.groupby("decade").size().reset_index(name="count")

X = reg_data["decade"].values.reshape(-1, 1)
y = reg_data["count"].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, linestyle="--", label="Regression Line")
plt.title("Decade vs Movie Count")
plt.xlabel("Decade")
plt.ylabel("Movies")
plt.legend()
plt.show()

# ==========================================
# 12. T-TEST
# ==========================================
before_2000 = movies_per_year[movies_per_year["year"] < 2000]["count"]
after_2000  = movies_per_year[movies_per_year["year"] >= 2000]["count"]

t_stat, p_val = stats.ttest_ind(after_2000, before_2000)
print("\nT-test p-value:", p_val)

# ==========================================
# 13. CHI-SQUARE TEST
# ==========================================
df["primary_genre"] = df["genre"].str.split(", ").str[0]
df["decade"] = (df["year"] // 10) * 10

df["is_drama"] = df["primary_genre"].str.lower().str.contains("drama")

contingency = pd.crosstab(df["decade"], df["is_drama"])

chi2, p, dof, expected = stats.chi2_contingency(contingency)

print("\nChi-square:", chi2)
print("p-value:", p)
