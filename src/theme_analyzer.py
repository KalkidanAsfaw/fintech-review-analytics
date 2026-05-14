import re
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# 5 business-relevant themes and their keyword seeds
THEME_KEYWORDS = {
    "Transaction Performance": [
        "transfer", "transaction", "payment", "send", "receive",
        "slow", "fast", "speed", "delay", "loading", "process",
    ],
    "Account Access": [
        "login", "log in", "password", "otp", "sign in", "open",
        "access", "fingerprint", "biometric", "unlock", "verify",
        "register", "account",
    ],
    "UI & Design": [
        "interface", "design", "easy", "simple", "update", "button",
        "screen", "user friendly", "ui", "look", "layout", "navigate",
        "display", "theme",
    ],
    "Customer Support": [
        "support", "service", "help", "response", "staff", "call",
        "customer care", "agent", "contact", "helpline", "complaint",
        "resolve", "feedback",
    ],
    "Feature Requests": [
        "add", "feature", "option", "wish", "need", "improve",
        "please", "request", "would like", "suggestion", "missing",
        "allow", "enable",
    ],
}


def _stopwords() -> set:
    try:
        from nltk.corpus import stopwords
        import nltk
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except Exception:
        return {"the", "a", "an", "is", "it", "in", "on", "and", "or", "to", "of"}


def preprocess_text(text: str, stops: set = None) -> str:
    """Lowercase, remove punctuation, strip stopwords."""
    if not isinstance(text, str):
        return ""
    if stops is None:
        stops = _stopwords()
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in stops and len(t) > 2]
    return " ".join(tokens)


def assign_theme(text: str) -> str:
    """Assign the theme whose keywords match most in the review text."""
    if not isinstance(text, str) or not text.strip():
        return "Other"
    text_lower = text.lower()
    scores = {
        theme: sum(1 for kw in keywords if kw in text_lower)
        for theme, keywords in THEME_KEYWORDS.items()
    }
    best_theme = max(scores, key=scores.get)
    return best_theme if scores[best_theme] > 0 else "Other"


def get_top_keywords(
    df: pd.DataFrame,
    text_col: str = "review",
    bank: str = None,
    n: int = 20,
) -> list:
    """Return top n TF-IDF keywords (optionally filtered by bank)."""
    subset = df[df["bank"] == bank] if bank else df
    stops = _stopwords()
    corpus = subset[text_col].fillna("").apply(lambda t: preprocess_text(t, stops))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    tfidf = vectorizer.fit_transform(corpus)
    mean_scores = tfidf.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    top_idx = mean_scores.argsort()[::-1][:n]
    return [(feature_names[i], round(float(mean_scores[i]), 4)) for i in top_idx]


def analyze_themes(df: pd.DataFrame, text_col: str = "review") -> pd.DataFrame:
    df = df.copy()
    df["identified_theme"] = df[text_col].apply(assign_theme)
    dist = df["identified_theme"].value_counts()
    logger.info(f"Theme distribution:\n{dist.to_string()}")
    return df
