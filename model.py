import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('dataset/test_informative.csv')

# Features and target
X = df['review_text']
y = df['informativeness_level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(random_state=42))
])

# Alternative pipelines for comparison
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

pipeline2 = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())  # Best for text classification
])

pipeline3 = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC(random_state=42))  # Excellent for text
])

pipeline4 = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))  # Good baseline
])

# Save model
#joblib.dump(pipeline, 'informative_model.joblib')

# Train and compare all models
pipelines = {
    'Random Forest': pipeline,
    'Naive Bayes': pipeline2,
    'Linear SVM': pipeline3,
    'Logistic Regression': pipeline4
}

print("Training and comparing classifiers for text-based predictions...")
print("="*60)

results = {}
for name, pipe in pipelines.items():
    # Train model
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred = pipe.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Summary comparison
print("\n" + "="*60)
print("SUMMARY - Best Classifiers for Text:")
print("="*60)
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for i, (name, accuracy) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {accuracy:.4f}")

print(f"\n🏆 Best performer: {sorted_results[0][0]} ({sorted_results[0][1]:.4f})")