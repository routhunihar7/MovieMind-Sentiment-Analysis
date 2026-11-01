
# MovieMind – Sentiment Analysis on Movie Reviews Using Machine Learning and NLP

## 1. Project Overview

MovieMind is a machine-learning-based web application that analyzes movie reviews and predicts whether the review expresses a Positive or Negative sentiment.

The project uses Natural Language Processing (NLP) techniques to preprocess text, TF-IDF to extract textual features, and machine learning algorithms to classify reviews.

## 2. Objectives

- Analyze movie reviews using NLP.
- Clean and preprocess textual data.
- Convert text into numerical features using TF-IDF.
- Train and compare multiple machine learning algorithms.
- Predict sentiment from user-entered reviews.
- Display model performance through an interactive dashboard.

## 3. Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK |
| Feature Extraction | TF-IDF |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Model Storage | Joblib |

## 4. Dataset

The project uses the NLTK Movie Reviews dataset.

- Total reviews: 2,000
- Positive reviews: 1,000
- Negative reviews: 1,000

Dataset source:
https://www.nltk.org/howto/corpus.html

## 5. Methodology

1. Dataset collection
2. Text preprocessing
3. Feature extraction using TF-IDF
4. Training and testing data split (80:20)
5. Machine learning model training
6. Model evaluation and comparison
7. Sentiment prediction
8. Flask web application development

## 6. NLP Preprocessing

The following operations are performed:

- Convert text to lowercase.
- Remove HTML tags and URLs.
- Remove unwanted special characters.
- Tokenize the review.
- Remove common stopwords while preserving negation words.
- Apply WordNet lemmatization.

## 7. Machine Learning Algorithms

Three classification algorithms are implemented:

- Multinomial Naive Bayes
- Support Vector Machine (LinearSVC)
- Logistic Regression

TF-IDF with unigrams and bigrams is used for feature extraction.

## 8. Model Performance

The following results were obtained from the test dataset:

| Algorithm | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Naive Bayes | 81.25% | 82.05% | 80.00% | 81.01% |
| SVM | 85.25% | 84.06% | 87.00% | 85.50% |
| Logistic Regression | 83.50% | 82.52% | 85.00% | 83.74% |

The Linear Support Vector Machine achieved the highest accuracy of 85.25% and is used as the prediction model in the web application.

## 9. Application Features

- User-friendly MovieMind interface.
- Real-time movie review sentiment prediction.
- Positive and negative sentiment results.
- Model performance dashboard.
- Accuracy and evaluation metric visualizations.
- Confusion matrix visualizations.
- Prediction history tracking.

## 10. Project Execution

### Step 1: Create a virtual environment

```bash
python -m venv venv
```

### Step 2: Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK resources

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 5: Run the application

```bash
python app.py
```

Open the application in your browser:

http://127.0.0.1:5000/

Dashboard:

http://127.0.0.1:5000/dashboard

## 11. Conclusion

MovieMind demonstrates how Natural Language Processing and machine learning can be combined to perform sentiment analysis on movie reviews. The comparison of three classification algorithms shows that Linear SVM provides the best performance among the implemented models, achieving 85.25% accuracy.

The project also demonstrates the integration of a trained machine learning model with a Flask web application to provide real-time sentiment predictions and performance visualization.

## 12. Future Enhancements

- Integrate deep learning models such as LSTM and BERT.
- Support larger movie review datasets.
- Improve prediction accuracy through hyperparameter tuning.
- Add movie review analysis from online sources.
- Deploy the application to a cloud platform.
