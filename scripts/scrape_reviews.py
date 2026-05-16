"""
Scrape Google Play Store reviews for Ethiopian fintech apps and save
a clean, analysis-ready CSV to data/processed/reviews_clean.csv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scraper import scrape_all_apps, save_raw_reviews
from src.preprocessor import preprocess_reviews, save_clean_data

RAW_PATH = "data/raw/reviews_raw.csv"
CLEAN_PATH = "data/processed/reviews_clean.csv"


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    raw_df = scrape_all_apps(count=400)
    if raw_df.empty:
        print("No reviews collected. Check app IDs and network access.")
        sys.exit(1)

    save_raw_reviews(raw_df, RAW_PATH)

    clean_df = preprocess_reviews(raw_df)
    save_clean_data(clean_df, CLEAN_PATH)

    print(f"\nDone. {len(clean_df)} clean reviews saved to {CLEAN_PATH}")
    print(clean_df.groupby("bank")["review_id"].count().rename("review_count"))


if __name__ == "__main__":
    main()
