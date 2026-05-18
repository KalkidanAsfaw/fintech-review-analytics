-- bank_reviews database schema
-- Run with: psql -d bank_reviews -f sql/schema.sql

CREATE TABLE IF NOT EXISTS banks (
    bank_id   SERIAL       PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL UNIQUE,
    app_name  VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id        VARCHAR(50)  PRIMARY KEY,
    bank_id          INTEGER      NOT NULL REFERENCES banks(bank_id),
    review_text      TEXT,
    rating           SMALLINT     NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_date      DATE,
    sentiment_label  VARCHAR(10)  CHECK (sentiment_label IN ('positive', 'negative', 'neutral')),
    sentiment_score  NUMERIC(6,4),
    identified_theme VARCHAR(50),
    source           VARCHAR(50)  DEFAULT 'Google Play'
);
