from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from src.db_connector import insert_banks, insert_reviews, BANK_METADATA


def make_mock_conn(fetchall_returns=None):
    cur = MagicMock()
    cur.fetchall.return_value = fetchall_returns or []
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cur


def sample_df(bank="CBE Mobile Banking"):
    return pd.DataFrame({
        "review_id":        ["r1"],
        "bank":             [bank],
        "review_text":      ["great app"],
        "rating":           [5],
        "date":             ["2024-01-01"],
        "distilbert_label": ["positive"],
        "distilbert_score": [0.98],
        "identified_theme": ["UI & General Experience"],
        "source":           ["Google Play"],
    })


# ── insert_banks ──────────────────────────────────────────────────────────────

def test_insert_banks_executes_insert_for_each_bank():
    fetchall = [(i+1, name) for i, name in enumerate(BANK_METADATA)]
    conn, cur = make_mock_conn(fetchall_returns=fetchall)
    insert_banks(conn)
    assert cur.execute.call_count == len(BANK_METADATA) + 1  # N inserts + 1 SELECT


def test_insert_banks_returns_name_to_id_mapping():
    fetchall = [(1, "CBE Mobile Banking"), (2, "Bank of Abyssinia"), (3, "Dashen Bank")]
    conn, cur = make_mock_conn(fetchall_returns=fetchall)
    mapping = insert_banks(conn)
    assert mapping == {"CBE Mobile Banking": 1, "Bank of Abyssinia": 2, "Dashen Bank": 3}


def test_insert_banks_commits():
    conn, cur = make_mock_conn(fetchall_returns=[(1, "CBE Mobile Banking")])
    insert_banks(conn)
    conn.commit.assert_called_once()


# ── insert_reviews ────────────────────────────────────────────────────────────

@patch("src.db_connector.execute_values")
def test_insert_reviews_returns_correct_count(mock_exec):
    conn, cur = make_mock_conn()
    df = sample_df()
    count = insert_reviews(conn, df, {"CBE Mobile Banking": 1})
    assert count == 1
    mock_exec.assert_called_once()


@patch("src.db_connector.execute_values")
def test_insert_reviews_skips_unknown_bank(mock_exec):
    conn, _ = make_mock_conn()
    df = sample_df(bank="Unknown Bank")
    count = insert_reviews(conn, df, {"CBE Mobile Banking": 1})
    assert count == 0
    mock_exec.assert_not_called()


@patch("src.db_connector.execute_values")
def test_insert_reviews_commits(mock_exec):
    conn, _ = make_mock_conn()
    insert_reviews(conn, sample_df(), {"CBE Mobile Banking": 1})
    conn.commit.assert_called_once()


@patch("src.db_connector.execute_values")
def test_insert_reviews_uses_distilbert_columns(mock_exec):
    conn, _ = make_mock_conn()
    df = pd.DataFrame({
        "review_id":        ["r1"],
        "bank":             ["Dashen Bank"],
        "review_text":      ["slow transfer"],
        "rating":           [2],
        "date":             ["2024-01-01"],
        "distilbert_label": ["negative"],
        "distilbert_score": [0.91],
        "identified_theme": ["Transaction Performance"],
        "source":           ["Google Play"],
    })
    insert_reviews(conn, df, {"Dashen Bank": 3})
    args = mock_exec.call_args[0][2]  # the records list
    assert args[0][5] == "negative"   # sentiment_label
    assert args[0][6] == 0.91         # sentiment_score


@patch("src.db_connector.execute_values")
def test_insert_reviews_handles_null_date_and_sentiment(mock_exec):
    conn, _ = make_mock_conn()
    df = pd.DataFrame({
        "review_id":        ["r1"],
        "bank":             ["CBE Mobile Banking"],
        "review_text":      ["ok"],
        "rating":           [3],
        "date":             [None],
        "distilbert_label": [None],
        "distilbert_score": [float("nan")],
        "identified_theme": [None],
        "source":           ["Google Play"],
    })
    count = insert_reviews(conn, df, {"CBE Mobile Banking": 1})
    assert count == 1
    args = mock_exec.call_args[0][2]
    assert args[0][4] is None   # review_date
    assert args[0][5] is None   # sentiment_label
    assert args[0][6] is None   # sentiment_score
