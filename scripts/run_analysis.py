"""
End-to-end sentiment and thematic analysis pipeline.

Input : data/processed/reviews_clean.csv
Output: data/processed/reviews_with_sentiment.csv  (full analysis)
        data/processed/reviews_final.csv            (5-column spec output)

Final CSV columns: review_id, review_text, sentiment_label,
                   sentiment_score, identified_theme
"""
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.nlp_pipeline import process_dataframe
from src.sentiment import analyze_sentiment, load_pipeline
from src.theme_analyzer import analyze_themes, get_top_keywords

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLEAN_PATH   = "data/processed/reviews_clean.csv"
FULL_PATH    = "data/processed/reviews_with_sentiment.csv"
FINAL_PATH   = "data/processed/reviews_final.csv"

FINAL_COLS   = ["review_id", "review_text", "sentiment_label",
                "sentiment_score", "identified_theme"]


def main():
    if not os.path.exists(CLEAN_PATH):
        logger.error(f"Clean data not found at {CLEAN_PATH}. Run scripts/scrape_reviews.py first.")
        sys.exit(1)

    df = pd.read_csv(CLEAN_PATH)
    logger.info(f"Loaded {len(df)} reviews from {CLEAN_PATH}")
    df.insert(0, "review_id", [f"review_{i+1}" for i in range(len(df))])

    # ── Step 1: NLP pipeline (tokenize → stopwords → lemmatize) ──────────────
    # Produces 'processed_text' used by theme assignment and TF-IDF.
    # Sentiment models run on the original 'review' text (unprocessed).
    df = process_dataframe(df, text_col="review", output_col="processed_text",
                           use_lemmatization=True)

    # ── Step 2: Sentiment analysis (distilbert, primary) ─────────────────────
    logger.info("Running distilbert sentiment analysis...")
    pipe = load_pipeline()
    df = analyze_sentiment(df, text_col="review", pipe=pipe)

    # ── Step 3: Thematic analysis (keyword matching on processed text) ────────
    logger.info("Running thematic analysis...")
    df = analyze_themes(df, text_col="processed_text")

    # ── Step 4: Log TF-IDF top keywords per bank ─────────────────────────────
    for bank in sorted(df["bank"].unique()):
        kws = get_top_keywords(df, text_col="processed_text", bank=bank, n=10)
        logger.info(f"Top keywords [{bank}]: " +
                    ", ".join(f"{k}({s})" for k, s in kws))

    # ── Step 5: Save full analysis file ──────────────────────────────────────
    full = df.rename(columns={"review": "review_text"})
    full.to_csv(FULL_PATH, index=False)
    logger.info(f"Full analysis saved → {FULL_PATH} ({len(full)} rows)")

    # ── Step 6: Save 5-column spec output ────────────────────────────────────
    # distilbert_label / distilbert_score → spec names sentiment_label / sentiment_score
    final = df.rename(columns={
        "review":            "review_text",
        "distilbert_label":  "sentiment_label",
        "distilbert_score":  "sentiment_score",
    })[FINAL_COLS]
    final.to_csv(FINAL_PATH, index=False)
    logger.info(f"Final spec CSV saved → {FINAL_PATH} ({len(final)} rows)")
    logger.info(f"Columns: {FINAL_COLS}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Sentiment by bank ===")
    print(df.groupby(["bank", "sentiment_label"])["review_id"]
            .count().unstack(fill_value=0).to_string())

    print("\n=== Themes by bank ===")
    print(df.groupby(["bank", "identified_theme"])["review_id"]
            .count().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
