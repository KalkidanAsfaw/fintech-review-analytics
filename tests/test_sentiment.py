from unittest.mock import MagicMock, patch
import pandas as pd
from src.sentiment import _to_label, analyze_sentiment, NEUTRAL_THRESHOLD


def test_to_label_positive_high_confidence():
    label, score = _to_label("POSITIVE", 0.95)
    assert label == "positive"
    assert score == 0.95


def test_to_label_negative_high_confidence():
    label, score = _to_label("NEGATIVE", 0.88)
    assert label == "negative"
    assert score == 0.88


def test_to_label_low_confidence_is_neutral():
    label, _ = _to_label("POSITIVE", NEUTRAL_THRESHOLD - 0.01)
    assert label == "neutral"
    label, _ = _to_label("NEGATIVE", NEUTRAL_THRESHOLD - 0.01)
    assert label == "neutral"


def _make_mock_pipe(raw_label: str = "POSITIVE", score: float = 0.95):
    """Return a callable that mimics the HuggingFace pipeline interface."""
    def pipe(batch, **kwargs):
        return [{"label": raw_label, "score": score} for _ in batch]
    return pipe


def test_analyze_sentiment_adds_columns():
    df = pd.DataFrame({"review": ["great app", "terrible experience", "ok"]})
    pipe = _make_mock_pipe("POSITIVE", 0.92)
    result = analyze_sentiment(df, text_col="review", pipe=pipe)
    assert "sentiment_label" in result.columns
    assert "sentiment_score" in result.columns
    assert len(result) == 3


def test_analyze_sentiment_labels_positive():
    df = pd.DataFrame({"review": ["love it"]})
    pipe = _make_mock_pipe("POSITIVE", 0.96)
    result = analyze_sentiment(df, pipe=pipe)
    assert result["sentiment_label"].iloc[0] == "positive"


def test_analyze_sentiment_labels_negative():
    df = pd.DataFrame({"review": ["crashes every time"]})
    pipe = _make_mock_pipe("NEGATIVE", 0.91)
    result = analyze_sentiment(df, pipe=pipe)
    assert result["sentiment_label"].iloc[0] == "negative"


def test_analyze_sentiment_labels_neutral_for_low_confidence():
    df = pd.DataFrame({"review": ["it is okay"]})
    pipe = _make_mock_pipe("POSITIVE", 0.60)
    result = analyze_sentiment(df, pipe=pipe)
    assert result["sentiment_label"].iloc[0] == "neutral"


def test_analyze_sentiment_handles_empty_text():
    df = pd.DataFrame({"review": [None, "", "good"]})
    pipe = _make_mock_pipe("POSITIVE", 0.90)
    result = analyze_sentiment(df, pipe=pipe)
    assert len(result) == 3
    assert result["sentiment_label"].notna().all()


def test_analyze_sentiment_does_not_mutate_input():
    df = pd.DataFrame({"review": ["nice app"]})
    pipe = _make_mock_pipe()
    _ = analyze_sentiment(df, pipe=pipe)
    assert "sentiment_label" not in df.columns
