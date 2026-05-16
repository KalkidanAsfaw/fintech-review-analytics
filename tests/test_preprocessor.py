import pandas as pd
import pytest
from src.preprocessor import (
    clean_text,
    remove_duplicates,
    handle_missing,
    preprocess_reviews,
)

SAMPLE_DATA = {
    "reviewId": ["r1", "r2", "r1", "r4", "r5"],
    "content":  ["Great app!", None, "Great app!", "<b>Love it</b>  ", "   "],
    "score":    [5, 4, 5, None, 3],
    "at":       ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-04"],
    "app_name": ["Bank A"] * 5,
    "source":   ["Google Play"] * 5,
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
    assert len(result) == 4
    assert result["reviewId"].nunique() == 4


def test_handle_missing_drops_null_and_empty():
    df = make_df()
    df = remove_duplicates(df)
    result = handle_missing(df)
    assert result["content"].isnull().sum() == 0
    assert (result["content"].str.strip() == "").sum() == 0
    assert result["score"].isnull().sum() == 0


def test_preprocess_reviews_output_columns():
    df = make_df()
    result = preprocess_reviews(df)
    for col in ["review", "rating", "date", "bank", "source"]:
        assert col in result.columns


def test_preprocess_reviews_date_format():
    df = make_df()
    result = preprocess_reviews(df)
    assert result["date"].str.match(r"\d{4}-\d{2}-\d{2}").all()


def test_preprocess_reviews_no_duplicates():
    df = make_df()
    result = preprocess_reviews(df)
    assert result["review_id"].nunique() == len(result)


def test_preprocess_reviews_all_lowercase():
    df = make_df()
    result = preprocess_reviews(df)
    assert result["review"].str.isupper().sum() == 0


def test_preprocess_reviews_raises_on_missing_columns():
    df = pd.DataFrame({"content": ["ok"], "score": [5]})
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_reviews(df)


def test_preprocess_reviews_empty_input():
    df = pd.DataFrame()
    result = preprocess_reviews(df)
    assert result.empty
