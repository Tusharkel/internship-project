"""
Run this script once to train and save the model:
    python classifier/ml/train.py

Expects a CSV at data/emails.csv with columns:
    text   → raw email body
    label  → "spam" or "ham"
"""

import sys
import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from classifier.ml.preprocess import clean_text

DATA_PATH      = Path("data/emails.csv")
MODEL_PATH     = Path("models/naive_bayes_spam.joblib")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")


def train():
    print("📂  Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    assert {"text", "label"}.issubset(df.columns), "CSV must have 'text' and 'label' columns"

    df["clean"] = df["text"].apply(clean_text)
    X, y = df["clean"], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🔧  Training TF-IDF + Naive Bayes pipeline...")
    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
    model      = MultinomialNB(alpha=0.1)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    model.fit(X_train_vec, y_train)

    print("\n📊  Evaluation on test set:")
    preds = model.predict(X_test_vec)
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    # Persist
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"✅  Model saved to {MODEL_PATH}")
    print(f"✅  Vectorizer saved to {VECTORIZER_PATH}")


if __name__ == "__main__":
    train()