import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# Correct relative path from ml/ folder
file_path = "data/fake_job_postings.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print(df.head())

print("\nColumn info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nFraudulent value counts:")
print(df['fraudulent'].value_counts())

# Combine important text fields into one "text_fused" column
text_cols = [
    "title", 
    "company_profile", 
    "description", 
    "requirements", 
    "benefits"
]

df[text_cols] = df[text_cols].fillna("")

df["text_fused"] = df[text_cols].agg(" ".join, axis=1)

print("\nSample fused text:")
print(df["text_fused"].iloc[0][:500])  # first 500 chars



df["clean_text"] = df["text_fused"].apply(clean_text)

print("\nSample cleaned text:")
print(df["clean_text"].iloc[0][:500])


#Implement TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
model.fit(X_train, y_train)

ypred = model.predict(X_test)

print(confusion_matrix(y_test, ypred))
print(classification_report(y_test, ypred))

y_probs = model.predict_proba(X_test)[:, 1]

for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    y_pred_threshold = (y_probs > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_threshold, average='binary')
    print(f"Threshold: {threshold}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

THRESHOLD = 0.7
y_pred = (y_probs >= THRESHOLD).astype(int)

import joblib

joblib.dump(model, "ml/model.joblib")
joblib.dump(vectorizer, "ml/vectorizer.joblib")
print("Model and vectorizer saved.")