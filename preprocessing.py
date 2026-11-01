
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLP resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))

# Preserve negation words because they affect sentiment
stop_words = stop_words - {'not', 'no', 'nor'}

lemmatizer = WordNetLemmatizer()


# Text preprocessing function
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenization
    words = text.split()

    # Remove stopwords and apply lemmatization
    cleaned_words = []

    for word in words:

        if word not in stop_words and len(word) > 1:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    return " ".join(cleaned_words)


# Load original dataset
df = pd.read_csv("dataset/movie_reviews.csv")

# Apply preprocessing
df["cleaned_review"] = df["review"].apply(preprocess_text)

# Save processed dataset
df.to_csv(
    "dataset/processed_movie_reviews.csv",
    index=False
)

# Display results
print("\nText Preprocessing Completed!")

print("\nOriginal Review:")
print(df["review"].iloc[0])

print("\nCleaned Review:")
print(df["cleaned_review"].iloc[0])

print("\nDataset Shape:", df.shape)

print("\nProcessed dataset saved successfully!")
