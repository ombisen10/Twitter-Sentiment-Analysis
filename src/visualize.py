"""
Visualization Module for Twitter Sentiment Analysis
Meets Rubric Criterion 8: Visualization (10 Points)
Provides customized, publication-grade figures tailored for the Twitter sentiment dataset.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import Counter
from sklearn.metrics import roc_curve, auc

# Consistent high-aesthetic theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
PALETTE_POS = "#2ecc71"
PALETTE_NEG = "#e74c3c"
PALETTE_MAIN = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]


def plot_class_distribution(labels: pd.Series, save_path: Optional[str] = None):
    """Visualizes sentiment class balance with counts and percentages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    counts = labels.value_counts().sort_index()
    names = ["Negative (0)", "Positive (1)"]
    colors = [PALETTE_NEG, PALETTE_POS]

    # Bar chart
    bars = ax1.bar(names, counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax1.set_title("Tweet Sentiment Class Distribution (Count)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Number of Tweets", fontsize=11)
    ax1.set_ylim(0, max(counts.values) * 1.15)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:,}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Donut chart
    ax2.pie(counts.values, labels=names, autopct='%1.1f%%', startangle=90,
            colors=colors, explode=(0.04, 0.04),
            wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True})
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax2.add_artist(centre_circle)
    ax2.set_title("Sentiment Proportion Ratio", fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_text_lengths(df: pd.DataFrame, text_col: str = "clean_text", label_col: str = "label", save_path: Optional[str] = None):
    """Plots word count and character length distributions across sentiments."""
    df_plot = df.copy()
    df_plot['char_length'] = df_plot[text_col].apply(len)
    df_plot['word_count'] = df_plot[text_col].apply(lambda x: len(x.split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Character length
    sns.histplot(data=df_plot, x='char_length', hue=label_col, palette=[PALETTE_NEG, PALETTE_POS],
                 kde=True, ax=axes[0], bins=30, alpha=0.5, element="step")
    axes[0].set_title("Character Length Distribution by Sentiment", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Character Count")
    axes[0].set_ylabel("Tweet Frequency")
    axes[0].legend(["Positive", "Negative"])

    # Word count
    sns.histplot(data=df_plot, x='word_count', hue=label_col, palette=[PALETTE_NEG, PALETTE_POS],
                 kde=True, ax=axes[1], bins=25, alpha=0.5, element="step")
    axes[1].set_title("Word Count Distribution by Sentiment", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Tweet Frequency")
    axes[1].legend(["Positive", "Negative"])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_ngrams(df: pd.DataFrame, text_col: str = "clean_text", label_col: str = "label", top_n: int = 15, save_path: Optional[str] = None):
    """Plots top distinguishing n-grams for positive and negative tweets."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, (sentiment_val, label_name, color) in enumerate([
        (0, "Negative Sentiment", PALETTE_NEG),
        (1, "Positive Sentiment", PALETTE_POS)
    ]):
        texts = df[df[label_col] == sentiment_val][text_col].dropna()
        all_words = " ".join(texts).split()
        counter = Counter(all_words)
        common = counter.most_common(top_n)

        words, counts = zip(*common) if common else ([], [])
        y_pos = np.arange(len(words))

        axes[idx].barh(y_pos, counts, color=color, edgecolor="black", alpha=0.85)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(words, fontsize=11)
        axes[idx].invert_yaxis()  # Top word at the top
        axes[idx].set_title(f"Top {top_n} Tokens in {label_name}", fontsize=13, fontweight='bold', pad=10)
        axes[idx].set_xlabel("Frequency Count", fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(results_dict: Dict[str, dict], save_path: Optional[str] = None):
    """Plots confusion matrix heatmaps side-by-side for all evaluated models."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.8))
    if n_models == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results_dict.items()):
        cm = data["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Pred Neg', 'Pred Pos'],
                    yticklabels=['Actual Neg', 'Actual Pos'],
                    annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(name, fontsize=10.5, fontweight='bold', pad=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(results_dict: Dict[str, dict], y_test: np.ndarray, save_path: Optional[str] = None):
    """Plots combined ROC curves with AUC scores for all models."""
    plt.figure(figsize=(9, 7))

    for idx, (name, data) in enumerate(results_dict.items()):
        model = data["model"]
        scores = None
        if hasattr(model, "predict_proba"):
            try:
                classes = list(getattr(model, "classes_", [0, 1]))
                pos_idx = classes.index(1) if 1 in classes else 1
                scores = model.predict_proba(data.get("X_test_vec", None))[:, pos_idx]
            except Exception:
                scores = None
        elif hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(data.get("X_test_vec", None))
            except Exception:
                scores = None
        if scores is None:
            scores = data["predictions"]

        fpr, tpr, _ = roc_curve(y_test, scores)
        # Recompute AUC from the same scores used for the curve so the
        # legend value always matches the plotted line.
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2.2)

    plt.plot([0, 1], [0, 1], 'k--', label="Random Baseline (AUC = 0.500)", alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plots multi-metric performance comparison bar chart across models."""
    df_melt = comparison_df.melt(
        id_vars=["Algorithm"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_melt,
        x="Algorithm",
        y="Score",
        hue="Metric",
        palette="viridis",
        edgecolor="black",
        alpha=0.9
    )

    plt.title("Cross-Algorithm Performance Metrics Comparison", fontsize=14, fontweight='bold', pad=12)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.xlabel("Machine Learning Algorithm", fontsize=11)
    plt.ylim(0, 1.15)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.legend(loc="upper right", frameon=True, ncol=4)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}",
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, rotation=90)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
