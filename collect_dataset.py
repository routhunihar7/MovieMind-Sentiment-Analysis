
import nltk
import pandas as pd
import os

from nltk.corpus import movie_reviews

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

reviews = []
sentiments = []

# Collect positive and negative reviews
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):

        review = " ".join(movie_reviews.words(fileid))

        reviews.append(review)
        sentiments.append(category)

# Create DataFrame
df = pd.DataFrame({
    "review": reviews,
    "sentiment": sentiments
})

# Save dataset
df.to_csv("dataset/movie_reviews.csv", index=False)

# Display information
print("Dataset Created Successfully!")
print("Total Reviews:", len(df))

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())
