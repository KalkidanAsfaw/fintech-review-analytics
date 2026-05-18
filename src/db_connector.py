import os
import logging
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "bank_reviews"),
    "user":   os.getenv("DB_USER", os.getenv("USER", "postgres")),
    # No host → psycopg2 connects via Unix socket (peer auth, no password needed)
    # Set DB_HOST env var to override for remote/Docker connections
    **({
        "host":     os.environ["DB_HOST"],
        "port":     os.getenv("DB_PORT", "5432"),
        "password": os.getenv("DB_PASSWORD", ""),
    } if os.getenv("DB_HOST") else {}),
}

BANK_METADATA = {
    "CBE Mobile Banking": "com.combanketh.mobilebanking",
    "Bank of Abyssinia":  "com.boa.boaMobileBanking",
    "Dashen Bank":        "com.dashen.dashensuperapp",
}


def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    logger.info(f"Connected to {DB_CONFIG['dbname']} on {DB_CONFIG.get('host', 'unix socket')}")
    return conn


def insert_banks(conn) -> dict:
    """Insert bank rows and return {bank_name: bank_id} mapping."""
    with conn.cursor() as cur:
        for bank_name, app_name in BANK_METADATA.items():
            cur.execute(
                """
                INSERT INTO banks (bank_name, app_name)
                VALUES (%s, %s)
                ON CONFLICT (bank_name) DO NOTHING
                """,
                (bank_name, app_name),
            )
        conn.commit()

        cur.execute("SELECT bank_id, bank_name FROM banks")
        mapping = {row[1]: row[0] for row in cur.fetchall()}
    logger.info(f"Banks table: {mapping}")
    return mapping


def insert_reviews(conn, df, bank_mapping: dict) -> int:
    """Bulk-insert reviews DataFrame into the reviews table. Returns row count."""
    import pandas as pd

    records = []
    for _, row in df.iterrows():
        bank_id = bank_mapping.get(row["bank"])
        if bank_id is None:
            continue
        # support both spec column names and full-analysis column names
        sentiment_label = (
            row.get("sentiment_label") or row.get("distilbert_label")
        )
        sentiment_score_raw = (
            row.get("sentiment_score") if pd.notna(row.get("sentiment_score", float("nan")))
            else row.get("distilbert_score")
        )
        records.append((
            row["review_id"],
            bank_id,
            row.get("review_text") or row.get("review"),
            int(row["rating"]),
            row["date"] if pd.notna(row["date"]) else None,
            sentiment_label,
            float(sentiment_score_raw) if pd.notna(sentiment_score_raw) else None,
            row.get("identified_theme"),
            row.get("source", "Google Play"),
        ))

    if not records:
        logger.warning("No records to insert — check bank mapping")
        return 0

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO reviews
              (review_id, bank_id, review_text, rating, review_date,
               sentiment_label, sentiment_score, identified_theme, source)
            VALUES %s
            ON CONFLICT (review_id) DO NOTHING
            """,
            records,
        )
        conn.commit()

    logger.info(f"Inserted {len(records)} reviews")
    return len(records)


def run_verification(conn) -> dict:
    """Run integrity checks and return results as a dict."""
    queries = {
        "reviews_per_bank": """
            SELECT b.bank_name, COUNT(r.review_id) AS review_count
            FROM banks b
            LEFT JOIN reviews r USING (bank_id)
            GROUP BY b.bank_name
            ORDER BY review_count DESC
        """,
        "avg_rating_per_bank": """
            SELECT b.bank_name, ROUND(AVG(r.rating), 2) AS avg_rating
            FROM banks b
            JOIN reviews r USING (bank_id)
            GROUP BY b.bank_name
            ORDER BY avg_rating DESC
        """,
        "nulls_in_key_columns": """
            SELECT
                COUNT(*) FILTER (WHERE review_text IS NULL)     AS null_review_text,
                COUNT(*) FILTER (WHERE rating IS NULL)          AS null_rating,
                COUNT(*) FILTER (WHERE sentiment_label IS NULL) AS null_sentiment_label
            FROM reviews
        """,
        "sentiment_distribution": """
            SELECT sentiment_label, COUNT(*) AS count
            FROM reviews
            GROUP BY sentiment_label
            ORDER BY count DESC
        """,
    }

    results = {}
    with conn.cursor() as cur:
        for name, sql in queries.items():
            cur.execute(sql)
            results[name] = cur.fetchall()
    return results
