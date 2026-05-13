from unittest.mock import patch
import pandas as pd
import pytest
from src.scraper import scrape_app_reviews, scrape_all_apps, save_raw_reviews

MOCK_REVIEWS = [
    {
        "reviewId": "abc123",
        "content": "Great app!",
        "score": 5,
        "at": "2024-01-01",
        "userName": "User1",
        "thumbsUpCount": 10,
        "replyContent": None,
        "repliedAt": None,
    },
    {
        "reviewId": "def456",
        "content": "Needs improvement.",
        "score": 2,
        "at": "2024-01-02",
        "userName": "User2",
        "thumbsUpCount": 3,
        "replyContent": None,
        "repliedAt": None,
    },
]


@patch("src.scraper.reviews", return_value=(MOCK_REVIEWS, None))
def test_scrape_app_reviews_returns_dataframe(mock_reviews):
    df = scrape_app_reviews("com.test.app", "Test App", count=2)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "app_name" in df.columns
    assert "app_id" in df.columns
    assert "source" in df.columns
    assert df["app_name"].iloc[0] == "Test App"
    assert df["source"].iloc[0] == "Google Play"


@patch("src.scraper.reviews", return_value=([], None))
def test_scrape_app_reviews_returns_empty_when_no_reviews(mock_reviews):
    df = scrape_app_reviews("com.no.reviews", "No Reviews App", count=10)
    assert df.empty


@patch("src.scraper.reviews", side_effect=Exception("Network error"))
def test_scrape_app_reviews_returns_empty_on_error(mock_reviews):
    df = scrape_app_reviews("com.bad.app", "Bad App", count=10)
    assert df.empty


@patch("src.scraper.reviews", return_value=(MOCK_REVIEWS, None))
def test_scrape_all_apps_combines_results(mock_reviews):
    apps = {"App A": "com.a.app", "App B": "com.b.app"}
    df = scrape_all_apps(apps=apps, count=2)
    assert len(df) == 4  # 2 reviews × 2 apps
    assert set(df["app_name"]) == {"App A", "App B"}


@patch("src.scraper.reviews", return_value=([], None))
def test_scrape_all_apps_empty_when_no_results(mock_reviews):
    df = scrape_all_apps(apps={"App": "com.empty.app"}, count=10)
    assert df.empty


def test_save_raw_reviews(tmp_path):
    df = pd.DataFrame(MOCK_REVIEWS)
    df["app_name"] = "Test App"
    df["app_id"] = "com.test.app"
    path = str(tmp_path / "raw.csv")
    save_raw_reviews(df, path)
    loaded = pd.read_csv(path)
    assert len(loaded) == 2
