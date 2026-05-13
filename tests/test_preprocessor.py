import pandas as pd
import pytest
from src.preprocessor import (
    clean_text,
    remove_duplicates,
    handle_missing,
    preprocess_reviews,
)

SAMPLE_DATA = {
    "reviewId": ["r1", "r2", "r1", "r4"],
    "content": ["Great app!", None, "Great app!", "<b>Love it</b>  "],
    "score": [5, 4, 5, None],
    "at": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"],
    "app_name": ["Bank A", "Bank A", "Bank A", "Bank A"],
    "app_id": ["com.a", "com.a", "com.a", "com.a"],
}


def make_df():
    return pd.DataFrame(SAMPLE_DATA)


def test_clean_text_strips_html():
    assert clean_text("<b>Hello</b>") == "hello"


def test_clean_text_collapses_whitespace():
    assert clean_text("too   many   spaces") == "too many spaces"


def test_clean_text_lowercases():
    assert clean_text("UPPER CASE") == "upper case"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_remove_duplicates():
    df = make_df()
    result = remove_duplicates(df)
    assert len(result) == 3
    assert result["reviewId"].nunique() == 3


def test_handle_missing_drops_null_content_and_score():
    df = make_df()
    df = remove_duplicates(df)
    result = handle_missing(df)
    assert result["content"].isnull().sum() == 0
    assert result["score"].isnull().sum() == 0


def test_preprocess_reviews_full_pipeline():
    df = make_df()
    result = preprocess_reviews(df)
    assert "review_id" in result.columns
    assert "review_text" in result.columns
    assert "rating" in result.columns
    assert "date" in result.columns
    assert "bank" in result.columns
    assert result["review_text"].str.isupper().sum() == 0  # all lowercased
    assert result["review_id"].nunique() == len(result)    # no duplicates


def test_preprocess_reviews_raises_on_missing_columns():
    df = pd.DataFrame({"content": ["ok"], "score": [5]})
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_reviews(df)


def test_preprocess_reviews_empty_input():
    df = pd.DataFrame()
    result = preprocess_reviews(df)
    assert result.empty
