"""
Preprocessing Module for Twitter Sentiment Analysis
Handles text cleaning, tokenization, stopword removal, and normalization.
"""

import re
import string
from typing import List, Optional

# Pre-defined contraction mapping for social media text
CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
    "mightn't": "might not", "must've": "must have", "mustn't": "must not",
    "needn't": "need not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that'd": "that would", "that's": "that is",
    "there'd": "there would", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is", "where'd": "where did",
    "where's": "where is", "who'll": "who will", "who's": "who is", "won't": "will not",
    "would've": "would have", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have", "u": "you",
    "ur": "your", "plz": "please", "thx": "thanks", "omg": "oh my god",
    "btw": "by the way", "idk": "i do not know", "tbh": "to be honest"
}

# Standard English sentiment-neutral stopwords (excluding negation terms)
NEGATION_WORDS = {"not", "no", "nor", "never", "neither", "barely", "hardly", "scarcely"}

DEFAULT_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now"
} - NEGATION_WORDS


class TweetPreprocessor:
    """
    Robust Tweet Preprocessor designed to handle noisy social media text.
    
    Steps:
    1. URL Removal (e.g. https://t.co/...)
    2. User Mention Removal (e.g. @user)
    3. Hashtag normalization (#happy -> happy)
    4. Contraction Expansion (don't -> do not)
    5. Emoji / Emoticon translation to sentiment tokens
    6. Special character & punctuation stripping
    7. Lowercasing
    8. Tokenization
    9. Sentiment-aware Stopword Filtering (preserves negations)
    10. Lemmatization / Stemming
    """

    def __init__(self, use_lemmatizer: bool = True, preserve_negations: bool = True):
        self.use_lemmatizer = use_lemmatizer
        self.preserve_negations = preserve_negations
        self.stopwords = DEFAULT_STOPWORDS
        self.lemmatizer = None
        self._init_nltk_tools()

    def _init_nltk_tools(self):
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Try loading NLTK stopwords
            try:
                sw = set(stopwords.words('english'))
                if self.preserve_negations:
                    sw = sw - NEGATION_WORDS
                self.stopwords = sw
            except LookupError:
                pass  # Fall back to default internal set
            
            # Try loading Lemmatizer
            try:
                self.lemmatizer = WordNetLemmatizer()
            except LookupError:
                self.lemmatizer = None
        except ImportError:
            self.lemmatizer = None

    def expand_contractions(self, text: str) -> str:
        """Expands informal English contractions and Twitter slang."""
        words = text.split()
        expanded = [CONTRACTIONS.get(w.lower(), w) for w in words]
        return " ".join(expanded)

    def clean_text(self, text: str) -> str:
        """Removes social media artifacts and sanitizes string."""
        if not isinstance(text, str):
            return ""

        # Lowercase first for uniformity
        text = text.lower()

        # 1. Remove URLs (http, https, www)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # 2. Remove HTML entities (e.g., &amp;, &lt;, &gt;)
        text = re.sub(r"&[a-z]+;", " ", text)

        # 3. Remove User Mentions (@username)
        text = re.sub(r"@\w+", " ", text)

        # 4. Remove Hashtag symbol but keep tag word (#positive -> positive)
        text = re.sub(r"#(\w+)", r"\1", text)

        # 5. Expand contractions (can't -> cannot)
        text = self.expand_contractions(text)

        # 6. Map common emoticons to sentiment markers
        text = re.sub(r":\)|:-\)|:\^\)|:D|:-D|<3", " sentimentpositive ", text)
        text = re.sub(r":\(|:-\(|:'\(|:\[", " sentimentnegative ", text)

        # 7. Remove non-alphabetic characters (digits, punctuation)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # 8. Collapse repeated characters (e.g., 'sooooo gooood' -> 'soo good')
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # 9. Normalize multiple whitespaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Splits cleaned text into individual word tokens."""
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filters out high-frequency stopwords while retaining sentiment negations."""
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Reduces words to their base dictionary lemma."""
        if self.lemmatizer:
            try:
                return [self.lemmatizer.lemmatize(t) for t in tokens]
            except Exception:
                return tokens
        return tokens

    def preprocess(self, text: str) -> str:
        """
        Full end-to-end preprocessing pipeline for a single tweet string.
        Returns cleaned, normalized, space-delimited text ready for vectorization.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        lemmatized = self.lemmatize_tokens(filtered)
        return " ".join(lemmatized)


def preprocess_corpus(corpus: List[str], use_lemmatizer: bool = True) -> List[str]:
    """Applies preprocessing to an entire collection of tweets."""
    preprocessor = TweetPreprocessor(use_lemmatizer=use_lemmatizer)
    return [preprocessor.preprocess(doc) for doc in corpus]
