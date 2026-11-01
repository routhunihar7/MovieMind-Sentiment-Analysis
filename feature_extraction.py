
import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Create models folder
os.makedirs("models", exist_ok=True)

# Load processed dataset
df = pd.read_csv("dataset/processed_movie_reviews.csv")

# Separate features and labels
X = df["cleaned_review"]
y = df["sentiment"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

# Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data
X_test_tfidf = tfidf.transform(X_test)

# Save vectorizer
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

# Save training and testing data
data = {
    "X_train": X_train_tfidf,
    "X_test": X_test_tfidf,
    "y_train": y_train,
    "y_test": y_test
}

joblib.dump(data, "models/feature_data.pkl")

# Display results
print("\nTF-IDF Feature Extraction Completed!")

print("\nTraining Samples:", X_train_tfidf.shape[0])
print("Testing Samples:", X_test_tfidf.shape[0])

print("Number of TF-IDF Features:", X_train_tfidf.shape[1])

print("\nTraining Matrix Shape:", X_train_tfidf.shape)
print("Testing Matrix Shape:", X_test_tfidf.shape)

print("\nVectorizer and Feature Data Saved Successfully!")
