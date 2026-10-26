"""Flask API + voting/confidence semantics (lightweight stubbed models)."""
import numpy as np
import pytest
from scipy import sparse

import app as appmod


class _StubModel:
    def __init__(self, classes, proba_rows, preds):
        self.classes_ = np.array(classes)
        self._proba = np.array(proba_rows, dtype=float)
        self._preds = np.array(preds)

    def predict_proba(self, X):
        n = X.shape[0]
        return np.tile(self._proba[:1], (n, 1))

    def predict(self, X):
        return np.tile(self._preds[:1], (X.shape[0],))


class _StubPreprocessor:
    def preprocess(self, t):
        return t.lower().strip()


class _StubVec:
    def transform(self, texts):
        return sparse.csr_matrix(np.ones((len(texts), 2)))


@pytest.fixture
def client():
    appmod.state["preprocessor"] = _StubPreprocessor()
    appmod.state["vectorizer"] = _StubVec()
    # classes_ deliberately ordered [1, 0] to prove we don't assume index 1 = positive
    appmod.state["nb_scratch"] = _StubModel([1, 0], [[0.8, 0.2]], [1])
    appmod.state["nb_sklearn"] = _StubModel([1, 0], [[0.6, 0.4]], [1])
    appmod.state["lr_model"] = _StubModel([0, 1], [[0.3, 0.7]], [0])
    appmod.state["is_ready"] = True
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_index_route(client):
    r = client.get("/")
    assert r.status_code == 200


def test_predict_schema_and_voting(client):
    r = client.post("/api/predict", json={"text": "Loving this!"})
    assert r.status_code == 200
    d = r.get_json()
    for k in ["raw_text", "clean_text", "tokens", "sentiment", "polarity",
              "confidence", "agreement", "agreement_count",
              "scratch_model", "sklearn_model", "logistic_regression"]:
        assert k in d
    # votes 1,1,0 -> Positive 2/3
    assert d["sentiment"] == "Positive" and d["agreement"] == "2/3"
    # confidence = mean winning-class (positive) prob = mean(0.8,0.6,0.7)=0.70 -> 70.0
    assert abs(d["confidence"] - 70.0) < 0.01
    # stub sklearn classes [1,0] with row [0.6,0.4]: pos=0.6 (index 0), neg=0.4
    assert d["sklearn_model"]["pos_prob"] == 60.0


def test_empty_rejected(client):
    assert client.post("/api/predict", json={"text": "   "}).status_code == 400


def test_missing_field(client):
    assert client.post("/api/predict", json={}).status_code == 400


def test_non_string_rejected(client):
    assert client.post("/api/predict", json={"text": 123}).status_code == 400


def test_too_long(client):
    assert client.post("/api/predict", json={"text": "x" * 2001}).status_code == 413


def test_non_json(client):
    assert client.post("/api/predict", data="hello",
                       content_type="text/plain").status_code == 400
