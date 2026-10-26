"""
Interactive Flask Web Application for Twitter Sentiment Analysis (Topic 18)
Provides a real-time web interface for live tweet classification, model comparisons,
EDA visualizations, and academic report inspection.
"""

import os
import time
import traceback
import numpy as np
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
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024  # guard oversized bodies

MAX_TWEET_CHARS = 2000
ALLOWED_FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".csv"}

# Global state for loaded models and data
state = {
    "preprocessor": None,
    "vectorizer": None,
    "nb_scratch": None,
    "nb_sklearn": None,
    "lr_model": None,
    "dataset_stats": {},
    "is_ready": False,
    "init_error": None,
}


def _probs_for_labels(model, vec):
    """Return (neg_prob, pos_prob) using the model's own classes_ ordering.

    Never assumes probability index 0 is negative and 1 is positive.
    """
    proba = model.predict_proba(vec)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    try:
        neg_idx = classes.index(0)
        pos_idx = classes.index(1)
    except ValueError:
        neg_idx, pos_idx = 0, 1 if len(proba) > 1 else 0
    return float(proba[neg_idx]), float(proba[pos_idx])


def initialize_models():
    """Initializes and trains models for real-time inference."""
    print("[APP] Initializing NLP components and training models...")
    t0 = time.time()
    df = ensure_dataset()

    preprocessor = TweetPreprocessor(use_lemmatizer=True, preserve_negations=True)

    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"].apply(preprocessor.preprocess)
        n_before = len(df)
        df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
        n_removed = n_before - len(df)
        if n_removed:
            print(f"[APP] Removed {n_removed} empty-after-cleaning rows.")

    # Reproducible stratified 80/20 split. TF-IDF is fit on train only.
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].values,
        df["sentiment"].values,
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"].values,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb_scratch = MultinomialNaiveBayesScratch(alpha=1.0)
    nb_scratch.fit(X_train_vec, y_train)

    nb_sklearn = MultinomialNB(alpha=1.0)
    nb_sklearn.fit(X_train_vec, y_train)

    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X_train_vec, y_train)

    state["preprocessor"] = preprocessor
    state["vectorizer"] = vectorizer
    state["nb_scratch"] = nb_scratch
    state["nb_sklearn"] = nb_sklearn
    state["lr_model"] = lr_model

    state["dataset_stats"] = {
        "total_records": int(len(df)),
        "pos_count": int((df["sentiment"] == 1).sum()),
        "neg_count": int((df["sentiment"] == 0).sum()),
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "test_size": int(len(y_test)),
        "scratch_acc": round(float(nb_scratch.score(X_test_vec, y_test)) * 100, 2),
        "sklearn_acc": round(float(nb_sklearn.score(X_test_vec, y_test)) * 100, 2),
        "lr_acc": round(float(lr_model.score(X_test_vec, y_test)) * 100, 2),
    }

    state["is_ready"] = True
    print(f"[APP] All models ready for inference in {time.time()-t0:.2f}s!")


@app.route("/")
def index():
    return render_template("index.html", stats=state["dataset_stats"])


@app.route("/health")
def health():
    return jsonify({"ready": state["is_ready"], "stats": state["dataset_stats"]})


@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Serves generated visualization images (safe extension allowlist)."""
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_FIGURE_EXTS or ".." in filename:
        return jsonify({"error": "File not found."}), 404
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    return send_from_directory(fig_dir, filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predicts sentiment for a user-submitted tweet.

    Contract (all on 200):
      raw_text, clean_text, tokens, sentiment, polarity,
      confidence (mean winning-class probability across 3 models, uncalibrated),
      confidence_definition, agreement (e.g. "2/3"), agreement_count,
      avg_pos_prob, avg_neg_prob, inference_ms,
      scratch_model/sklearn_model/logistic_regression {prediction, pos_prob, neg_prob},
      warning (optional, for low-confidence or empty-after-cleaning inputs).
    Errors: {error: str} with 400/413/503.
    """
    if not state["is_ready"]:
        return jsonify({"error": "Models are still initializing. Please retry."}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON with a 'text' field."}), 400
    try:
        data = request.get_json()
    except Exception:
        app.logger.exception("Malformed JSON in /api/predict")
        return jsonify({"error": "Malformed JSON body."}), 400
    if not isinstance(data, dict) or "text" not in data:
        return jsonify({"error": "Missing required field: 'text'."}), 400

    raw = data.get("text")
    if not isinstance(raw, str):
        return jsonify({"error": "'text' must be a string."}), 400
    raw_text = raw.strip()
    if not raw_text:
        return jsonify({"error": "Tweet text cannot be empty."}), 400
    if len(raw_text) > MAX_TWEET_CHARS:
        return jsonify({"error": f"Tweet exceeds {MAX_TWEET_CHARS} characters."}), 413

    try:
        t0 = time.perf_counter()
        cleaned_text = state["preprocessor"].preprocess(raw_text)
        tokens = cleaned_text.split() if cleaned_text else []
        warning = None
        if not cleaned_text:
            warning = "Input had no informative tokens after cleaning; prediction is low-signal."
            cleaned_text = raw_text.lower()[:200]
            tokens = []

        vec = state["vectorizer"].transform([cleaned_text if tokens else "emptyfallbacktokenxyz"])

        # Each model's ACTUAL predicted class (majority vote uses these, not thresholds).
        scratch_pred = int(state["nb_scratch"].predict(vec)[0])
        sklearn_pred = int(state["nb_sklearn"].predict(vec)[0])
        lr_pred = int(state["lr_model"].predict(vec)[0])

        s_neg, s_pos = _probs_for_labels(state["nb_scratch"], vec)
        k_neg, k_pos = _probs_for_labels(state["nb_sklearn"], vec)
        l_neg, l_pos = _probs_for_labels(state["lr_model"], vec)

        votes = [scratch_pred, sklearn_pred, lr_pred]
        main_pred = 1 if sum(votes) >= 2 else 0
        agree = sum(1 for v in votes if v == main_pred)

        # Ensemble confidence: mean of the three winning-class probabilities.
        # Documented as an uncalibrated average, NOT a calibrated certainty.
        win_probs = [
            s_pos if main_pred == 1 else s_neg,
            k_pos if main_pred == 1 else k_neg,
            l_pos if main_pred == 1 else l_neg,
        ]
        main_confidence = float(np.mean(win_probs)) * 100
        avg_pos = round(float(np.mean([s_pos, k_pos, l_pos])) * 100, 2)
        avg_neg = round(float(np.mean([s_neg, k_neg, l_neg])) * 100, 2)
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        if main_confidence < 60.0:
            low = "Low confidence (<60%): models disagree or the tweet is ambiguous."
            warning = f"{warning} {low}" if warning else low

        resp = {
            "raw_text": raw_text,
            "clean_text": cleaned_text,
            "tokens": tokens,
            "sentiment": "Positive" if main_pred == 1 else "Negative",
            "polarity": main_pred,
            "confidence": round(float(main_confidence), 2),
            "confidence_definition": "Mean winning-class probability across 3 models (uncalibrated average).",
            "agreement": f"{agree}/3",
            "agreement_count": agree,
            "avg_pos_prob": avg_pos,
            "avg_neg_prob": avg_neg,
            "inference_ms": inference_ms,
            "scratch_model": {
                "prediction": "Positive" if scratch_pred == 1 else "Negative",
                "pos_prob": round(s_pos * 100, 2),
                "neg_prob": round(s_neg * 100, 2),
            },
            "sklearn_model": {
                "prediction": "Positive" if sklearn_pred == 1 else "Negative",
                "pos_prob": round(k_pos * 100, 2),
                "neg_prob": round(k_neg * 100, 2),
            },
            "logistic_regression": {
                "prediction": "Positive" if lr_pred == 1 else "Negative",
                "pos_prob": round(l_pos * 100, 2),
                "neg_prob": round(l_neg * 100, 2),
            },
        }
        if warning:
            resp["warning"] = warning
        return jsonify(resp)
    except Exception:
        app.logger.error("Prediction failed:\n%s", traceback.format_exc())
        return jsonify({"error": "Prediction failed. Please try again."}), 500


# Eager init so WSGI servers (Gunicorn/Render) are ready immediately.
# Failures are captured with an actionable message instead of a raw traceback.
# Set T18_SKIP_INIT=1 to skip (used by the fast unit-test suite).
if os.environ.get("T18_SKIP_INIT") != "1":
    try:
        initialize_models()
    except Exception as exc:  # pragma: no cover - startup path
        state["init_error"] = str(exc)
        print(f"[APP][ERROR] Model initialization failed: {exc}")
        print("[APP][HINT] Check requirements are installed and data/ is readable.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n==================================================================")
    print(f"[START] TWITTER SENTIMENT WEB APP RUNNING AT: http://0.0.0.0:{port}")
    print("==================================================================\n")
    if state["init_error"]:
        print(f"[FATAL] Not starting: {state['init_error']}")
    else:
        app.run(host="0.0.0.0", port=port, debug=False)
