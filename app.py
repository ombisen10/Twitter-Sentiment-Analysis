"""
Interactive Flask Web Application for Twitter Sentiment Analysis (Topic 18)
Provides a real-time web interface for live tweet classification, model comparisons,
EDA visualizations, and academic report inspection.
"""

import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Local imports
from src.preprocess import TweetPreprocessor
from src.naive_bayes_scratch import MultinomialNaiveBayesScratch
from data_loader import ensure_dataset

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global state for loaded models and data
state = {
    "preprocessor": None,
    "vectorizer": None,
    "nb_scratch": None,
    "nb_sklearn": None,
    "lr_model": None,
    "dataset_stats": {},
    "is_ready": False
}


def initialize_models():
    """Initializes and trains models for real-time inference."""
    print("[APP] Initializing NLP components and training models...")
    df = ensure_dataset()

    preprocessor = TweetPreprocessor(use_lemmatizer=True, preserve_negations=True)

    # Check if preprocessed text is cached
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"].apply(preprocessor.preprocess)
        df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].values,
        df["sentiment"].values,
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"].values
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 1. Custom Scratch Naive Bayes
    nb_scratch = MultinomialNaiveBayesScratch(alpha=1.0)
    nb_scratch.fit(X_train_vec, y_train)

    # 2. Sklearn Multinomial NB
    nb_sklearn = MultinomialNB(alpha=1.0)
    nb_sklearn.fit(X_train_vec, y_train)

    # 3. Logistic Regression
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X_train_vec, y_train)

    state["preprocessor"] = preprocessor
    state["vectorizer"] = vectorizer
    state["nb_scratch"] = nb_scratch
    state["nb_sklearn"] = nb_sklearn
    state["lr_model"] = lr_model

    state["dataset_stats"] = {
        "total_records": len(df),
        "pos_count": int((df["sentiment"] == 1).sum()),
        "neg_count": int((df["sentiment"] == 0).sum()),
        "vocab_size": len(vectorizer.vocabulary_),
        "scratch_acc": round(float(nb_scratch.score(X_test_vec, y_test)) * 100, 2),
        "sklearn_acc": round(float(nb_sklearn.score(X_test_vec, y_test)) * 100, 2),
        "lr_acc": round(float(lr_model.score(X_test_vec, y_test)) * 100, 2)
    }

    state["is_ready"] = True
    print("[APP] All models ready for inference!")


@app.route("/")
def index():
    return render_template("index.html", stats=state["dataset_stats"])


@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Serves generated visualization images."""
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    return send_from_directory(fig_dir, filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predicts sentiment for a user-submitted tweet."""
    if not state["is_ready"]:
        return jsonify({"error": "Models are still initializing."}), 503

    data = request.get_json(force=True)
    raw_text = data.get("text", "").strip()

    if not raw_text:
        return jsonify({"error": "Tweet text cannot be empty."}), 400

    # 1. Preprocess
    cleaned_text = state["preprocessor"].preprocess(raw_text)
    tokens = cleaned_text.split()

    if not cleaned_text:
        # Fallback if text was purely non-alphanumeric noise
        cleaned_text = raw_text.lower()
        tokens = cleaned_text.split()

    # 2. Vectorize
    vec = state["vectorizer"].transform([cleaned_text])

    # 3. Custom Scratch Naive Bayes prediction
    scratch_probs = state["nb_scratch"].predict_proba(vec)[0]  # [prob_neg, prob_pos]
    scratch_pred = int(state["nb_scratch"].predict(vec)[0])

    # 4. Sklearn Naive Bayes prediction
    sklearn_probs = state["nb_sklearn"].predict_proba(vec)[0]

    # 5. Logistic Regression prediction
    lr_probs = state["lr_model"].predict_proba(vec)[0]
    lr_pred = int(state["lr_model"].predict(vec)[0])

    # Ensemble consensus (majority voting among Scratch NB, Sklearn NB, and Logistic Regression)
    votes = [scratch_pred, int(sklearn_probs[1] >= 0.5), lr_pred]
    main_pred = 1 if sum(votes) >= 2 else 0
    main_confidence = float(np.mean([
        scratch_probs[main_pred],
        sklearn_probs[main_pred],
        lr_probs[main_pred]
    ])) * 100

    return jsonify({
        "raw_text": raw_text,
        "clean_text": cleaned_text,
        "tokens": tokens,
        "sentiment": "Positive" if main_pred == 1 else "Negative",
        "polarity": main_pred,
        "confidence": round(main_confidence, 2),
        "scratch_model": {
            "prediction": "Positive" if scratch_pred == 1 else "Negative",
            "pos_prob": round(float(scratch_probs[1]) * 100, 2),
            "neg_prob": round(float(scratch_probs[0]) * 100, 2)
        },
        "sklearn_model": {
            "prediction": "Positive" if sklearn_probs[1] >= 0.5 else "Negative",
            "pos_prob": round(float(sklearn_probs[1]) * 100, 2),
            "neg_prob": round(float(sklearn_probs[0]) * 100, 2)
        },
        "logistic_regression": {
            "prediction": "Positive" if lr_pred == 1 else "Negative",
            "pos_prob": round(float(lr_probs[1]) * 100, 2),
            "neg_prob": round(float(lr_probs[0]) * 100, 2)
        }
    })


# Automatically initialize models so WSGI servers (Gunicorn/Render/Railway) are ready immediately
initialize_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================================")
    print(f"[START] TWITTER SENTIMENT WEB APP RUNNING AT: http://0.0.0.0:{port}")
    print(f"==================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
