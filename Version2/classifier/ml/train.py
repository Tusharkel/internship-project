"""
Trains Naive Bayes on the Kaggle email spam dataset.
The Kaggle dataset has word-frequency columns + a 'Prediction' label column.
  Prediction = 1 → spam
  Prediction = 0 → ham
"""

import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import numpy as np

DATA_PATH       = Path("data/emails.csv")
MODEL_PATH      = Path("models/naive_bayes.joblib")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")


def train():
    print("📂  Loading Kaggle dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"    Shape: {df.shape}")
    print(f"    Columns sample: {list(df.columns[:5])} ...")

    # ── Kaggle dataset uses word-frequency columns + 'Prediction' label ──
    # Drop the 'Email No.' column if present
    if "Email No." in df.columns:
        df = df.drop(columns=["Email No."])

    # Label column is 'Prediction' (1=spam, 0=ham)
    assert "Prediction" in df.columns, "Expected a 'Prediction' column in the CSV"

    y = df["Prediction"].map({1: "spam", 0: "ham"})
    X_raw = df.drop(columns=["Prediction"])

    # The Kaggle dataset is already numeric (word frequencies)
    # We scale and use directly — no TF-IDF needed for this format
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🔧  Training Naive Bayes...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)

    print("\n📊  Evaluation:")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, VECTORIZER_PATH)   # saving scaler in vectorizer slot
    print(f"✅  Model saved    → {MODEL_PATH}")
    print(f"✅  Scaler saved   → {VECTORIZER_PATH}")