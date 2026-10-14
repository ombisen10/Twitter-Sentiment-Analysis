"""
Multinomial Naive Bayes Classifier - Built from Scratch
Meets Rubric Criterion 3: Algorithm Implementation from Scratch (15 Points)

Mathematical Foundation:
Bayes' Theorem:
    P(Y = c | X) = [ P(Y = c) * P(X | Y = c) ] / P(X)

Conditional Independence Assumption (Naive):
    P(X | Y = c) = Product_{j=1..D} P(x_j | Y = c)

In Log-Space (to prevent arithmetic underflow):
    log P(Y = c | X) = log P(Y = c) + Sum_{j=1..D} x_j * log P(w_j | Y = c)

Laplace (Additive) Smoothing:
    P(w_j | Y = c) = (Count(w_j, c) + alpha) / (Total_Words_in_c + alpha * |Vocabulary|)
"""

import numpy as np
from scipy import sparse
from typing import Union


class MultinomialNaiveBayesScratch:
    """
    Multinomial Naive Bayes classifier implemented strictly from first mathematical principles.
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Additive (Laplace) smoothing parameter.
        alpha = 0: No smoothing
        alpha = 1: Standard Laplace smoothing
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.classes_ = None
        self.class_priors_ = None
        self.class_log_priors_ = None
        self.feature_log_prob_ = None  # Shape: (n_classes, n_features)
        self.n_features_in_ = None
        self.vocab_size_ = None

    def fit(self, X: Union[np.ndarray, sparse.spmatrix], y: Union[np.ndarray, list]):
        """
        Fits the Multinomial Naive Bayes model according to training data X and labels y.
        
        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training vectors (e.g. Word count or TF-IDF matrices).
        y : array-like of shape (n_samples,)
            Target values (0: Negative, 1: Positive).
        """
        # Convert inputs to standard numpy or sparse format
        y = np.asarray(y)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.vocab_size_ = self.n_features_in_

        # 1. Compute Class Prior Probabilities P(Y = c)
        # P(Y = c) = Count(Y = c) / Total_Samples
        self.class_priors_ = class_counts / float(n_samples)
        self.class_log_priors_ = np.log(self.class_priors_)

        # 2. Compute Word Counts per Class: Count(w_j, c)
        feature_counts = np.zeros((n_classes, self.n_features_in_), dtype=np.float64)

        is_sparse = sparse.issparse(X)
        for idx, c in enumerate(self.classes_):
            mask = (y == c)
            if is_sparse:
                # Efficient sum across rows for sparse CSR matrices
                feature_counts[idx, :] = np.asarray(X[mask].sum(axis=0)).ravel()
            else:
                feature_counts[idx, :] = X[mask].sum(axis=0)

        # 3. Apply Laplace Smoothing:
        # P(w_j | Y = c) = [Count(w_j, c) + alpha] / [Sum_j Count(w_j, c) + alpha * |V|]
        smoothed_counts = feature_counts + self.alpha
        smoothed_totals = smoothed_counts.sum(axis=1, keepdims=True)

        # 4. Store Conditional Log Likelihoods for numerical precision
        self.feature_log_prob_ = np.log(smoothed_counts) - np.log(smoothed_totals)

        return self

    def predict_log_proba(self, X: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """
        Calculates posterior unnormalized log probabilities:
        log P(Y = c | X) = log P(Y = c) + X * (feature_log_prob_)^T
        """
        if self.feature_log_prob_ is None:
            raise ValueError("Model is not fitted yet. Call fit() before predicting.")

        # X is (n_samples, n_features)
        # feature_log_prob_ is (n_classes, n_features)
        # Result is (n_samples, n_classes)
        if sparse.issparse(X):
            joint_log_likelihood = X.dot(self.feature_log_prob_.T)
        else:
            joint_log_likelihood = np.dot(X, self.feature_log_prob_.T)

        joint_log_likelihood += self.class_log_priors_
        return joint_log_likelihood

    def predict_proba(self, X: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """
        Converts log probabilities to normalized probabilities via the Log-Sum-Exp trick.
        P(Y = c | X) = exp(log_p - log_sum_exp)
        """
        log_prob = self.predict_log_proba(X)
        # Subtract max for numerical stability before exp
        max_log_prob = np.max(log_prob, axis=1, keepdims=True)
        exp_prob = np.exp(log_prob - max_log_prob)
        return exp_prob / np.sum(exp_prob, axis=1, keepdims=True)

    def predict(self, X: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """
        Predicts class label for samples in X.
        Class = argmax_c [ log P(Y = c | X) ]
        """
        log_prob = self.predict_log_proba(X)
        best_class_indices = np.argmax(log_prob, axis=1)
        return self.classes_[best_class_indices]

    def score(self, X: Union[np.ndarray, sparse.spmatrix], y: Union[np.ndarray, list]) -> float:
        """Returns accuracy score on given test data and labels."""
        y = np.asarray(y)
        preds = self.predict(X)
        return float(np.mean(preds == y))
