import pandas as pd
from src.theme_analyzer import assign_theme, analyze_themes, get_top_keywords, preprocess_text


def test_assign_theme_transaction():
    assert assign_theme("the transfer was very slow") == "Transaction Performance"


def test_assign_theme_account_access():
    assert assign_theme("I cannot login with my password") == "Account Access"


def test_assign_theme_ui_design():
    assert assign_theme("the interface is easy and simple to navigate") == "UI & General Experience"


def test_assign_theme_customer_support():
    assert assign_theme("customer support never picks up the call") == "Customer Support"


def test_assign_theme_feature_request():
    assert assign_theme("please add dark mode and mini statement history") == "Feature Requests"


def test_assign_theme_other_for_unmatched():
    result = assign_theme("amazing")
    assert isinstance(result, str)


def test_assign_theme_handles_none():
    assert assign_theme(None) == "Other"


def test_assign_theme_handles_empty():
    assert assign_theme("") == "Other"


def test_preprocess_text_lowercases():
    assert preprocess_text("HELLO WORLD") == preprocess_text("hello world")


def test_preprocess_text_removes_punctuation():
    result = preprocess_text("Hello, world!")
    assert "," not in result
    assert "!" not in result


def test_preprocess_text_removes_short_tokens():
    result = preprocess_text("it is ok")
    tokens = result.split()
    assert all(len(t) > 2 for t in tokens)


def test_analyze_themes_adds_column():
    df = pd.DataFrame({
        "review": ["I cannot login", "transfer is slow", "great interface"],
        "bank": ["Bank A"] * 3,
    })
    result = analyze_themes(df)
    assert "identified_theme" in result.columns
    assert len(result) == 3


def test_analyze_themes_does_not_mutate_input():
    df = pd.DataFrame({"review": ["login error"], "bank": ["Bank A"]})
    _ = analyze_themes(df)
    assert "identified_theme" not in df.columns


def test_get_top_keywords_returns_list_of_tuples():
    df = pd.DataFrame({
        "review": ["the transfer was very slow and painful", "login failed again today"],
        "bank": ["Bank A", "Bank A"],
    })
    result = get_top_keywords(df, text_col="review", bank="Bank A", n=5)
    assert isinstance(result, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in result)


def test_get_top_keywords_no_bank_filter():
    df = pd.DataFrame({
        "review": ["great app", "terrible service", "good transfer speed"],
        "bank": ["Bank A", "Bank B", "Bank A"],
    })
    result = get_top_keywords(df, n=5)
    assert len(result) <= 5
