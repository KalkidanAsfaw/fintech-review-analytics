"""
Run sentiment and thematic analysis on the cleaned review dataset.

Input : data/processed/reviews_clean.csv
Output: data/processed/reviews_with_sentiment.csv
        columns: review_id, review_text, sentiment_label, sentiment_score,
                 identified_theme, bank, rating, date, source
"""
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.sentiment import analyze_sentiment, load_pipeline
from src.theme_analyzer import analyze_themes, get_top_keywords

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLEAN_PATH = "data/processed/reviews_clean.csv"
OUTPUT_PATH = "data/processed/reviews_with_sentiment.csv"


def main():
    if not os.path.exists(CLEAN_PATH):
        logger.error(f"Clean data not found at {CLEAN_PATH}. Run scripts/scrape_reviews.py first.")
        sys.exit(1)

    df = pd.read_csv(CLEAN_PATH)
    logger.info(f"Loaded {len(df)} reviews from {CLEAN_PATH}")

    # generate stable review IDs
    df.insert(0, "review_id", [f"review_{i+1}" for i in range(len(df))])

    # sentiment analysis
    logger.info("Running sentiment analysis (distilbert)...")
    pipe = load_pipeline()
    df = analyze_sentiment(df, text_col="review", pipe=pipe)

    # thematic analysis
    logger.info("Running thematic analysis...")
    df = analyze_themes(df, text_col="review")

    # log top keywords per bank
    for bank in df["bank"].unique():
        keywords = get_top_keywords(df, text_col="review", bank=bank, n=10)
        kw_str = ", ".join(f"{k}({s})" for k, s in keywords)
        logger.info(f"Top keywords [{bank}]: {kw_str}")

    # save output
    output_cols = [
        "review_id", "review", "sentiment_label", "sentiment_score",
        "identified_theme", "bank", "rating", "date", "source",
    ]
    df = df.rename(columns={"review": "review_text"})
    output_cols = [c if c != "review" else "review_text" for c in output_cols]
    df[output_cols].to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {len(df)} analysed reviews to {OUTPUT_PATH}")

    # summary
    print("\n=== Sentiment by bank ===")
    print(df.groupby(["bank", "sentiment_label"])["review_id"].count().unstack(fill_value=0).to_string())
    print("\n=== Themes by bank ===")
    print(df.groupby(["bank", "identified_theme"])["review_id"].count().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
