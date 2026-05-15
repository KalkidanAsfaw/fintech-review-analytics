import pandas as pd
from src.nlp_pipeline import tokenize, remove_stopwords, lemmatize, run_pipeline, process_dataframe


def test_tokenize_lowercases():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_strips_punctuation():
    result = tokenize("login, failed!")
    assert "," not in result
    assert "!" not in " ".join(result)


def test_tokenize_returns_empty_for_non_string():
    assert tokenize(None) == []
    assert tokenize(123) == []


def test_remove_stopwords_filters_common_words():
    tokens = ["the", "app", "is", "good"]
    result = remove_stopwords(tokens)
    assert "the" not in result
    assert "is" not in result
    assert "app" in result
    assert "good" in result


def test_remove_stopwords_filters_single_chars():
    tokens = ["a", "b", "good"]
    result = remove_stopwords(tokens)
    assert "a" not in result
    assert "good" in result


def test_lemmatize_reduces_to_base_form():
    result = lemmatize(["crashes", "transfers", "loading"])
    assert "crash" in result
    assert "transfer" in result


def test_lemmatize_preserves_non_inflected_words():
    result = lemmatize(["login", "bank", "fast"])
    assert "login" in result
    assert "bank" in result


def test_run_pipeline_returns_string():
    result = run_pipeline("The app is crashing often")
    assert isinstance(result, str)
    assert len(result) > 0


def test_run_pipeline_removes_stopwords():
    result = run_pipeline("the app is very good")
    tokens = result.split()
    assert "the" not in tokens
    assert "is" not in tokens


def test_run_pipeline_lemmatizes_when_enabled():
    result = run_pipeline("crashes transfers", use_lemmatization=True)
    assert "crash" in result or "transfer" in result


def test_run_pipeline_skips_lemmatization_when_disabled():
    result = run_pipeline("crashes transfers", use_lemmatization=False)
    assert "crashes" in result or "transfers" in result


def test_run_pipeline_handles_empty_string():
    assert run_pipeline("") == ""


def test_run_pipeline_handles_none():
    assert run_pipeline(None) == ""


def test_process_dataframe_adds_output_column():
    df = pd.DataFrame({"review_text": ["great app", "login failed", "slow transfer"]})
    result = process_dataframe(df, text_col="review_text")
    assert "processed_text" in result.columns
    assert len(result) == 3


def test_process_dataframe_does_not_mutate_input():
    df = pd.DataFrame({"review_text": ["good app"]})
    _ = process_dataframe(df)
    assert "processed_text" not in df.columns


def test_process_dataframe_preserves_original_column():
    df = pd.DataFrame({"review_text": ["good app"]})
    result = process_dataframe(df)
    assert "review_text" in result.columns
    assert result["review_text"].iloc[0] == "good app"
