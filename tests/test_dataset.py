"""Dataset + TF-IDF boundary tests (no full retraining)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from data_loader import ensure_dataset
from src.preprocess import TweetPreprocessor


def test_label_mapping_consistent():
    df = ensure_dataset()
    assert {"text", "sentiment"}.issubset(df.columns)
    assert set(df["sentiment"].dropna().unique()).issubset({0, 1})
    for _, r in df.head(50).iterrows():
        if r["sentiment"] == 1:
            assert str(r["sentiment_label"]).lower() == "positive"
        else:
            assert str(r["sentiment_label"]).lower() == "negative"


def test_no_missing_targets():
    df = ensure_dataset()
    assert df["sentiment"].isnull().sum() == 0
    assert (df["text"].astype(str).str.strip() != "").sum() > 0


def test_tfidf_fit_only_on_train():
    prep = TweetPreprocessor(use_lemmatizer=False)
    texts = [prep.preprocess(t) for t in [
        "I love this movie :)", "Terrible service :(", "not bad at all",
        "worst flight ever", "so happy today", "awful delay again",
    ]]
    tr, te = train_test_split(texts, test_size=0.33, random_state=0)
    vec = TfidfVectorizer(max_features=50)
    Xtr = vec.fit_transform(tr)
    Xte = vec.transform(te)
    assert Xtr.shape[1] == Xte.shape[1] == len(vec.vocabulary_)
    # Unseen word yields all-zero row (no leakage of test vocab into fit).
    zv = vec.transform(["zzzunseenwordqqq"])
    assert zv.nnz == 0
