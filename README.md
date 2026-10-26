# Twitter Sentiment Analysis (Mini-Project: Topic 18)

## 📌 Project Overview
**Course / Academic Year:** Machine Learning Practical Mini Project (A.Y. 2025-26)  
**Topic 18:** Classify tweets as positive/negative sentiment using Twitter Sentiment dataset.  
**Domain:** Natural Language Processing (NLP) & Supervised Machine Learning  

---

## 🎯 Objectives
1. Perform in-depth **Exploratory Data Analysis (EDA)** and understanding of Twitter sentiment text data.
2. Build an end-to-end **NLP Preprocessing Pipeline** (noise cleaning, handle removal, URL scrubbing, contraction expansion, tokenization, stopword filtering, lemmatization).
3. **Implement Multinomial Naive Bayes completely from scratch** using first principles, Bayes' theorem, Laplace smoothing, and log-space computations.
4. Train and benchmark multiple ML models:
   - **Custom Naive Bayes (From Scratch)**
   - **Scikit-Learn Multinomial Naive Bayes**
   - **Logistic Regression**
   - **Support Vector Machine (Linear SVC)**
   - **Random Forest Classifier**
5. Perform **Hyperparameter Tuning** with `GridSearchCV` (TF-IDF parameters, smoothing `alpha`, regularization `C`).
6. Comprehensive **Model Evaluation**:
   - Classification Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
   - **Step-by-step Manual Calculation** of metrics from confusion matrix sample to satisfy rubric requirements.
7. Generate rich, publication-grade **Visualizations** (word clouds, n-gram distributions, confusion matrices, ROC curves).
8. Provide actionable **Business Interpretation & Error Analysis** (sarcasm, ambiguous sentiment, false positives/negatives).

---

## 📊 Grading Rubric Mapping (100 Points)

| Criteria | Points | Where It is Covered |
|---|---|---|
| **1. Dataset Understanding** | 10 | `twitter_sentiment_analysis.ipynb` (Section 1) & `PROJECT_REPORT.md` |
| **2. Data Preprocessing** | 10 | `src/preprocess.py`, Notebook Section 2 |
| **3. Algorithm Implementation (Scratch)** | 15 | `src/naive_bayes_scratch.py`, Notebook Section 3 |
| **4. Algorithm Application / Modeling** | 15 | `src/models.py`, Notebook Section 4 |
| **5. Parameter Tuning / Customization** | 10 | `src/models.py`, Notebook Section 5 |
| **6. Model Evaluation / Metrics** | 10 | Notebook Section 6 (Includes **Manual Calculation**) |
| **7. Comparison / Analysis** | 10 | Notebook Section 7 & `PROJECT_REPORT.md` |
| **8. Visualization** | 10 | `src/visualize.py`, Notebook Section 8 & `figures/` |
| **9. Interpretation & Discussion** | 10 | Notebook Section 9 & `PROJECT_REPORT.md` |
| **10. Code Documentation & Readability** | 5 | Clean code structure, comments, docstrings, type hints |
| **11. Report / Presentation** | 5 | `PROJECT_REPORT.md` |
| **TOTAL** | **100** | Full compliance with A.Y. 2025-26 Rubric |

---

## 📂 Project Directory Structure

```text
Twitter-Sentiment-Analysis/
│
├── data/
│   └── twitter_sentiment_dataset.csv   # Curated balanced dataset (10,000 tweets)
├── figures/                             # Generated plots & visual charts
│   ├── class_distribution.png
│   ├── wordcloud_positive.png
│   ├── wordcloud_negative.png
│   ├── top_ngrams.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── model_comparison.png
├── src/
│   ├── __init__.py
│   ├── preprocess.py                   # Complete NLP cleaning & tokenization
│   ├── naive_bayes_scratch.py          # Custom Multinomial Naive Bayes from scratch
│   ├── models.py                       # Sklearn models, pipelines & tuning
│   └── visualize.py                    # Plotting utilities
├── data_loader.py                      # Dataset preparation and download script
├── main.py                             # Automated complete end-to-end execution
├── twitter_sentiment_analysis.ipynb    # Interactive Jupyter Notebook for submission
├── PROJECT_REPORT.md                   # Comprehensive academic project report
├── requirements.txt                    # Project dependencies
└── README.md                           # Project guide (this file)
```

---

## 🚀 Getting Started

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
```bash
python main.py
```

### 3. Open Interactive Jupyter Notebook
```bash
jupyter notebook twitter_sentiment_analysis.ipynb
```

### 4. Run the Web App
```bash
python app.py
# open http://127.0.0.1:5000/
```

### 5. Run Tests
```bash
python -m pytest tests/ -q
```

## API Contract (`POST /api/predict`)

Request: `{ "text": "I love this!" }` (string, 1–2000 chars).
Success returns `sentiment` (Positive/Negative), `polarity` (1/0),
`confidence` = mean winning-class probability across the 3 live models
(uncalibrated average, not a calibrated certainty), `agreement` (e.g. `2/3`),
per-model predictions + `pos_prob`/`neg_prob`, `avg_pos_prob`/`avg_neg_prob`,
`inference_ms` (server-side model time), and optional `warning` for
low-confidence (<60%) or empty-after-cleaning inputs.
Errors return `{ "error": "..." }` with 400/413/503.

## Validity Note

Held-out test (80/20, seed 42, N=1,999): Scratch NB 94.55%, LR 99.50%,
Linear SVC / RF 99.60%. Accuracy is inflated by emoticon tokens
(`sentimentpositive`/`sentimentnegative`) that are strongly label-correlated
in the NLTK corpus. 10,000 → 9,994 rows after cleaning; 988 cleaned duplicates
(118 overlap train/test); one cleaned text (`true`) has both labels.
Documented, not hidden.
