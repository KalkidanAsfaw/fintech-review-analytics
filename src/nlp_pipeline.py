"""
Modular NLP preprocessing pipeline.

Steps (each callable independently):
  1. tokenize       — split text into word tokens
  2. remove_stopwords — filter out common English stop words
  3. lemmatize      — reduce tokens to base form (optional, uses NLTK WordNet)
  4. run_pipeline   — apply all steps to a single string
  5. process_dataframe — apply the pipeline to a DataFrame column
"""

import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ── NLTK resource bootstrap ───────────────────────────────────────────────────

def _ensure_nltk_resources() -> None:
    import nltk
    for resource in ("stopwords", "wordnet", "omw-1.4", "punkt_tab"):
        nltk.download(resource, quiet=True)


def _get_stopwords() -> set:
    _ensure_nltk_resources()
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))


def _get_lemmatizer():
    _ensure_nltk_resources()
    from nltk.stem import WordNetLemmatizer
    return WordNetLemmatizer()


# ── Pipeline steps ────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens, stripping punctuation."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def remove_stopwords(tokens: list[str], stops: set = None) -> list[str]:
    """Remove English stop words and single-character tokens."""
    if stops is None:
        stops = _get_stopwords()
    return [t for t in tokens if t not in stops and len(t) > 1]


def lemmatize(tokens: list[str], lemmatizer=None) -> list[str]:
    """Reduce each token to its base (lemma) form using NLTK WordNetLemmatizer."""
    if lemmatizer is None:
        lemmatizer = _get_lemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def run_pipeline(
    text: str,
    use_lemmatization: bool = True,
    stops: set = None,
    lemmatizer=None,
) -> str:
    """Full pipeline: tokenize → remove stopwords → (optional) lemmatize.

    Returns a single clean string suitable for TF-IDF or keyword matching.
    The original text is kept intact separately for sentiment models, which
    perform better on unprocessed input.
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, stops=stops)
    if use_lemmatization:
        tokens = lemmatize(tokens, lemmatizer=lemmatizer)
    return " ".join(tokens)


def process_dataframe(
    df: pd.DataFrame,
    text_col: str = "review_text",
    output_col: str = "processed_text",
    use_lemmatization: bool = True,
) -> pd.DataFrame:
    """Apply the NLP pipeline to every row of a DataFrame column.

    Adds `output_col` with the cleaned tokens joined as a string.
    The original `text_col` is preserved — sentiment models (distilbert)
    run on it directly; only theme assignment uses the processed text.
    """
    logger.info(
        f"Running NLP pipeline on '{text_col}' "
        f"(lemmatization={'on' if use_lemmatization else 'off'})"
    )
    stops = _get_stopwords()
    lemmatizer = _get_lemmatizer() if use_lemmatization else None

    df = df.copy()
    df[output_col] = df[text_col].apply(
        lambda t: run_pipeline(t, use_lemmatization=use_lemmatization,
                               stops=stops, lemmatizer=lemmatizer)
    )
    logger.info(f"Pipeline complete. Added column '{output_col}'.")
    return df
