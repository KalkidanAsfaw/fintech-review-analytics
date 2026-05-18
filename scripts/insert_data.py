"""
Insert cleaned and analysed review data into PostgreSQL bank_reviews database.

Input : data/processed/reviews_with_sentiment.csv (full analysis file)
Tables: banks, reviews
"""
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.db_connector import get_connection, insert_banks, insert_reviews, run_verification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/reviews_with_sentiment.csv"


def main():
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found: {DATA_PATH}. Run scripts/run_analysis.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    if "review_id" not in df.columns:
        df.insert(0, "review_id", [f"review_{i+1}" for i in range(len(df))])
    logger.info(f"Loaded {len(df)} rows from {DATA_PATH}")

    conn = get_connection()

    # insert banks
    bank_mapping = insert_banks(conn)

    # insert reviews
    inserted = insert_reviews(conn, df, bank_mapping)
    logger.info(f"Insertion complete: {inserted} rows")

    # verification
    print("\n=== Verification Queries ===")
    results = run_verification(conn)

    print("\nReviews per bank:")
    for row in results["reviews_per_bank"]:
        print(f"  {row[0]}: {row[1]}")

    print("\nAverage rating per bank:")
    for row in results["avg_rating_per_bank"]:
        print(f"  {row[0]}: {row[1]}")

    print("\nNulls in key columns:")
    nulls = results["nulls_in_key_columns"][0]
    print(f"  review_text={nulls[0]}  rating={nulls[1]}  sentiment_label={nulls[2]}")

    print("\nSentiment distribution:")
    for row in results["sentiment_distribution"]:
        print(f"  {row[0]}: {row[1]}")

    conn.close()


if __name__ == "__main__":
    main()
