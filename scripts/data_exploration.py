import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Loading the dataset
data = pd.read_csv('data/raw/movies.csv')

print(f"Number of ratings: {data.shape[0]}")
print(f"Number of unique users: {data['userId'].nunique()}")
print(f"Number of unique movies: {data['movieId'].nunique()}")

# Rating Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=data, palette='viridis')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# User Ratings Count
user_ratings_count = data.groupby('userId').size()
plt.figure(figsize=(10, 6))
sns.histplot(user_ratings_count, bins=30, kde=True)
plt.title('Distribution of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.show()

# Movie Ratings Count
movie_ratings_count = data.groupby('movieId').size()
plt.figure(figsize=(10, 6))
sns.histplot(movie_ratings_count, bins=30, kde=True)
plt.title('Distribution of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.show()
