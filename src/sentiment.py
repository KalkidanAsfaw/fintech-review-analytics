import logging
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
NEUTRAL_THRESHOLD = 0.75  # confidence below this is labelled neutral


def load_pipeline(model: str = MODEL_NAME):
    from transformers import pipeline
    logger.info(f"Loading sentiment model: {model}")
    return pipeline("sentiment-analysis", model=model, truncation=True, max_length=512)


def _to_label(raw_label: str, score: float) -> tuple:
    """Map distilbert POSITIVE/NEGATIVE + confidence to a 3-class label."""
    if score < NEUTRAL_THRESHOLD:
        return "neutral", score
    return raw_label.lower(), score


def analyze_sentiment(
    df: pd.DataFrame,
    text_col: str = "review",
    batch_size: int = 64,
    pipe=None,
) -> pd.DataFrame:
    if pipe is None:
        pipe = load_pipeline()

    texts = df[text_col].fillna("").tolist()
    labels, scores = [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = pipe(batch, truncation=True, max_length=512)
        for r in results:
            label, score = _to_label(r["label"], round(r["score"], 4))
            labels.append(label)
            scores.append(score)

        if i % (batch_size * 10) == 0:
            logger.info(f"Sentiment: processed {min(i + batch_size, len(texts))}/{len(texts)}")

    df = df.copy()
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    return df
