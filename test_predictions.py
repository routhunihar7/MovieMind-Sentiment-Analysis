
import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load trained model and vectorizer
model = joblib.load("models/svm_model.pkl")

tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"not", "no", "nor"}

lemmatizer = WordNetLemmatizer()


# Preprocessing function
def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(
        r'http\S+|www\S+',
        '',
        text
    )

    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text
    )

    words = text.split()

    cleaned_words = []

    for word in words:

        if word not in stop_words and len(word) > 1:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    return " ".join(cleaned_words)


# Function to predict sentiment
def predict_sentiment(review):

    cleaned_review = preprocess_text(review)

    review_vector = tfidf.transform([cleaned_review])

    prediction = model.predict(review_vector)

    if prediction[0] == "pos":
        return "POSITIVE"

    else:
        return "NEGATIVE"


# Test reviews
test_reviews = [
    "This movie was absolutely fantastic and amazing",
    "The film was boring and completely disappointing",
    "Excellent acting and wonderful story",
    "Terrible movie with poor direction"
]

print("\nSENTIMENT ANALYSIS RESULTS")
print("=" * 50)

for review in test_reviews:

    result = predict_sentiment(review)

    print("\nReview:", review)

    print("Predicted Sentiment:", result)

    print("-" * 50)
