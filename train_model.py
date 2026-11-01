
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load TF-IDF feature data
data = joblib.load("models/feature_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Define ML models
models = {
    "Naive Bayes": MultinomialNB(),

    "SVM": LinearSVC(
        random_state=42,
        max_iter=5000
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

results = []

print("\nMACHINE LEARNING MODEL TRAINING")
print("=" * 45)

# Train and evaluate each model
for name, model in models.items():

    print("\nTraining:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label="pos"
    )
    recall = recall_score(
        y_test, y_pred, pos_label="pos"
    )
    f1 = f1_score(
        y_test, y_pred, pos_label="pos"
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Generate confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["neg", "pos"]
    )

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )

    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"results/{filename}_confusion_matrix.png"
    )
    plt.close()

    # Save trained model
    joblib.dump(
        model,
        f"models/{filename}_model.pkl"
    )

    print(name, "Accuracy:", round(accuracy * 100, 2), "%")

# Create comparison DataFrame
results_df = pd.DataFrame(results)

# Save evaluation results
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 45)

print(results_df.to_string(index=False))

# Plot model accuracy comparison
plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.title("Machine Learning Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Machine Learning Algorithms")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("results/model_accuracy_comparison.png")
plt.close()

# Plot all evaluation metrics
metrics_df = results_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=metrics_df,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Model Performance Evaluation")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("results/model_performance_comparison.png")
plt.close()

print("\nAll models trained successfully!")
print("Models saved in models folder.")
print("Graphs saved in results folder.")
