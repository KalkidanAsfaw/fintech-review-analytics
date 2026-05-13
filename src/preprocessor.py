import re
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["reviewId", "content", "score", "at", "app_name", "app_id"]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["reviewId"])
    logger.info(f"Removed {before - len(df)} duplicate reviews")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["content", "score"]).copy()
    df["content"] = df["content"].fillna("")
    logger.info(f"Dropped {before - len(df)} rows with missing content/score")
    return df


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)          # strip HTML tags
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)   # remove special chars
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    return text.strip().lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "reviewId": "review_id",
        "content": "review_text",
        "score": "rating",
        "at": "date",
        "app_name": "bank",
        "app_id": "app_id",
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["review_length"] = df["review_text"].str.len()
    df["word_count"] = df["review_text"].str.split().str.len()
    return df


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[REQUIRED_COLUMNS].copy()
    df = remove_duplicates(df)
    df = handle_missing(df)
    df["content"] = df["content"].apply(clean_text)
    df = normalize_columns(df)
    df = df.reset_index(drop=True)
    logger.info(f"Preprocessing complete. {len(df)} clean reviews ready.")
    return df


def save_clean_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} clean reviews to {filepath}")
