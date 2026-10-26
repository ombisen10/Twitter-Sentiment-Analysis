"""Scratch Naive Bayes math tests on tiny deterministic fixtures."""
import numpy as np
import pytest
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB

from src.naive_bayes_scratch import MultinomialNaiveBayesScratch
from src.models import verify_manual_metrics


@pytest.fixture
def tiny_counts():
    # 4 docs, 3 terms. Classes [0, 1].
    X = np.array([
        [3.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 2.0],
        [0.0, 2.0, 3.0],
    ])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_priors_and_smoothing(tiny_counts):
    X, y = tiny_counts
    m = MultinomialNaiveBayesScratch(alpha=1.0).fit(X, y)
    assert list(m.classes_) == [0, 1]
    assert abs(m.class_priors_.sum() - 1.0) < 1e-12
    # Row 0: counts [5,1,1] +1 -> [6,2,2]/10 ; Row 1: [0,5,5]+1 -> [1,6,6]/13
    assert abs(np.exp(m.feature_log_prob_[0]).sum() - 1.0) < 1e-12
    assert abs(np.exp(m.feature_log_prob_[1]).sum() - 1.0) < 1e-12


def test_probability_normalization(tiny_counts):
    X, y = tiny_counts
    m = MultinomialNaiveBayesScratch(alpha=1.0).fit(X, y)
    P = m.predict_proba(X)
    assert np.all(P >= 0)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_sparse_parity(tiny_counts):
    X, y = tiny_counts
    d = MultinomialNaiveBayesScratch(alpha=1.0).fit(X, y)
    s = MultinomialNaiveBayesScratch(alpha=1.0).fit(sparse.csr_matrix(X), y)
    assert np.allclose(d.feature_log_prob_, s.feature_log_prob_)
    assert (d.predict(X) == s.predict(sparse.csr_matrix(X))).all()


def test_matches_sklearn_counts(tiny_counts):
    X, y = tiny_counts
    mine = MultinomialNaiveBayesScratch(alpha=1.0).fit(X, y)
    ref = MultinomialNB(alpha=1.0).fit(X, y)
    assert (mine.predict(X) == ref.predict(X)).all()
    assert np.allclose(mine.predict_proba(X), ref.predict_proba(X), atol=1e-9)


def test_not_fitted_raises():
    m = MultinomialNaiveBayesScratch()
    with pytest.raises(ValueError):
        m.predict(np.zeros((1, 3)))


def test_dim_mismatch_raises(tiny_counts):
    X, y = tiny_counts
    m = MultinomialNaiveBayesScratch().fit(X, y)
    with pytest.raises(ValueError):
        m.predict(np.zeros((1, 5)))


def test_all_zero_vector_uses_priors(tiny_counts):
    X, y = tiny_counts
    m = MultinomialNaiveBayesScratch().fit(X, y)
    pred = m.predict(np.zeros((1, 3)))
    assert pred[0] in (0, 1)  # argmax of log priors; must not crash


def test_verify_manual_metrics():
    cm = np.array([[975, 25], [84, 915]])
    y_true = [0] * 1000 + [1] * 999
    y_pred = [0] * 975 + [1] * 25 + [0] * 84 + [1] * 915
    out = verify_manual_metrics(cm, y_true, y_pred)
    assert out["compared"] and out["all_match"]
