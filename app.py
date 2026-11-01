
import os
import re
import joblib
import pandas as pd

from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize Flask application
app = Flask(__name__)

# Load trained SVM model
model = joblib.load("models/svm_model.pkl")

# Load TF-IDF vectorizer
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# Initialize NLP preprocessing tools
stop_words = set(stopwords.words("english"))

# Preserve negation words
stop_words = stop_words - {"not", "no", "nor"}

lemmatizer = WordNetLemmatizer()


# Create results folder
os.makedirs("results", exist_ok=True)

# Prediction history file
HISTORY_FILE = "results/prediction_history.csv"


# --------------------------------
# TEXT PREPROCESSING
# --------------------------------

def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenization
    words = text.split()

    # Stopword removal and lemmatization
    cleaned_words = []

    for word in words:

        if word not in stop_words and len(word) > 1:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    return " ".join(cleaned_words)


# --------------------------------
# SENTIMENT PREDICTION
# --------------------------------

def predict_sentiment(review):

    cleaned_review = preprocess_text(review)

    if not cleaned_review:
        return None

    review_vector = tfidf.transform([cleaned_review])

    prediction = model.predict(review_vector)[0]

    if prediction == "pos":
        return "Positive"

    return "Negative"


# --------------------------------
# SAVE PREDICTION HISTORY
# --------------------------------

def save_prediction(review, sentiment):

    new_record = pd.DataFrame([{
        "review": review,
        "sentiment": sentiment
    }])

    if os.path.exists(HISTORY_FILE):

        new_record.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        new_record.to_csv(
            HISTORY_FILE,
            index=False
        )


# --------------------------------
# HOME PAGE
# --------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    review = ""
    error = None

    if request.method == "POST":

        review = request.form.get("review", "").strip()

        if not review:

            error = "Please enter a movie review."

        elif len(review) > 10000:

            error = "Review must contain fewer than 10,000 characters."

        else:

            sentiment = predict_sentiment(review)

            if sentiment is None:

                error = "Please enter meaningful review text."

            else:

                save_prediction(review, sentiment)

    return render_template(
        "index.html",
        sentiment=sentiment,
        review=review,
        error=error
    )


# --------------------------------
# ANALYTICS DASHBOARD
# --------------------------------

@app.route("/dashboard")
def dashboard():

    # Load model evaluation results
    metrics_path = "results/model_comparison.csv"

    if not os.path.exists(metrics_path):

        return "Model comparison file not found. Run train_model.py first.", 404

    metrics_df = pd.read_csv(metrics_path)

    metrics = metrics_df.to_dict("records")

    # Load prediction history
    if os.path.exists(HISTORY_FILE):

        history_df = pd.read_csv(HISTORY_FILE)

        history = (
            history_df.tail(10)
            .iloc[::-1]
            .to_dict("records")
        )

        positive_count = int(
            (history_df["sentiment"] == "Positive").sum()
        )

        negative_count = int(
            (history_df["sentiment"] == "Negative").sum()
        )

    else:

        history = []
        positive_count = 0
        negative_count = 0

    total_predictions = positive_count + negative_count

    # Find best model
    best_model = metrics_df.loc[
        metrics_df["Accuracy"].idxmax()
    ]

    best_accuracy = round(
        best_model["Accuracy"] * 100, 2
    )

    return render_template(
        "dashboard.html",
        metrics=metrics,
        history=history,
        positive_count=positive_count,
        negative_count=negative_count,
        total_predictions=total_predictions,
        best_model=best_model["Model"],
        best_accuracy=best_accuracy
    )


# --------------------------------
# RUN APPLICATION
# --------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
