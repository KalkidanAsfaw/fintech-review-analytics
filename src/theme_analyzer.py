import re
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Themes are checked in priority order — more specific themes take precedence.
# "UI & General Experience" is intentionally last: its broad positive vocabulary
# acts as a catch-all for short, generic quality reviews ("good", "nice", "best app")
# that contain no signal for a more specific theme.
THEME_KEYWORDS = {
    "Transaction Performance": [
        # money operations
        "transfer", "transaction", "payment", "send money", "receive",
        "money", "cash", "fund", "top up", "topup", "bill", "pay",
        "airtime", "debit", "credit", "charge", "fee", "balance",
        "withdraw", "deposit", "pending", "process", "processing",
        # app stability during operations
        "crash", "crashes", "hang", "freeze", "stuck", "not working",
        "doesn't work", "wont open", "won't open", "force close",
        "not open", "crush", "lag", "restart", "stopped",
        # speed/loading
        "slow", "takes forever", "takes long", "loading", "delay",
        "delayed", "timeout", "network error", "connection",
    ],
    "Account Access": [
        "login", "log in", "log out", "logout", "password", "otp",
        "one time password", "sign in", "sign up", "open account",
        "fingerprint", "biometric", "face id", "unlock", "verify",
        "verification", "register", "registration", "forgot password",
        "reset", "blocked", "suspended", "pin", "authentication",
        "token", "session", "expire", "account number", "wrong password",
        "invalid", "cannot login", "can't login",
    ],
    "Customer Support": [
        "support", "customer service", "customer care", "help",
        "response", "respond", "staff", "call center", "call",
        "agent", "contact", "helpline", "hotline", "complaint",
        "complain", "resolve", "poor service", "bad service",
        "terrible service", "rude", "unhelpful", "no response",
        "never respond", "useless", "waste of time",
    ],
    "Feature Requests": [
        "please add", "add feature", "would like", "wish",
        "suggestion", "suggest", "request", "improve", "improvement",
        "missing feature", "allow", "enable", "option", "dark mode",
        "statement", "mini statement", "history", "notification",
        "virtual card", "card", "international", "dollar", "usd",
        "forex", "schedule", "recurring", "auto pay", "widget",
        "apple pay", "google pay", "qr", "scan",
    ],
    "UI & General Experience": [
        # interface & usability
        "interface", "design", "ui", "ux", "layout", "screen",
        "button", "display", "navigate", "navigation", "user friendly",
        "easy to use", "easy to navigate", "smooth", "clean", "modern",
        "simple", "intuitive", "responsive", "fast", "quick", "speed",
        # generic positive sentiment (catch-all for short reviews)
        "good", "great", "nice", "best", "excellent", "amazing",
        "awesome", "wonderful", "fantastic", "perfect", "love",
        "like", "appreciate", "thank", "well", "superb", "brilliant",
        "super", "convenient", "useful", "helpful", "recommend",
        "impressive", "satisfied", "happy", "enjoy", "comfortable",
        "good app", "great app", "nice app", "best app", "good work",
        "keep up", "well done", "go ahead", "one step ahead",
        # generic negative (short complaints)
        "bad", "worst", "terrible", "awful", "horrible", "boring",
        "disappointed", "poor", "useless app", "waste",
    ],
}

# Priority order: specific themes are evaluated before the broad catch-all.
THEME_PRIORITY = [
    "Transaction Performance",
    "Account Access",
    "Customer Support",
    "Feature Requests",
    "UI & General Experience",
]


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
    """Assign theme using priority-based first-match across THEME_PRIORITY order.

    More specific themes are checked first. 'UI & General Experience' is last
    and uses broad vocabulary to catch short generic reviews that carry no
    signal for a more specific theme.
    """
    if not isinstance(text, str) or not text.strip():
        return "Other"
    text_lower = text.lower()
    for theme in THEME_PRIORITY:
        if any(kw in text_lower for kw in THEME_KEYWORDS[theme]):
            return theme
    return "Other"


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
