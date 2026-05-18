# Fintech Review Analytics

Analysis of Google Play Store reviews for three Ethiopian banks — Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank — to extract insights on customer satisfaction, sentiment, and product feedback.

## Project Structure

```
fintech-review-analytics/
├── .github/workflows/    # CI/CD (GitHub Actions)
├── data/
│   ├── raw/              # Raw scraped reviews (gitignored)
│   └── processed/        # Cleaned dataset (gitignored)
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── scraper.py        # Google Play Store scraping logic
│   ├── preprocessor.py   # Data cleaning and normalization pipeline
│   ├── sentiment.py      # DistilBERT sentiment analysis pipeline
│   ├── theme_analyzer.py # TF-IDF keyword extraction and theme assignment
│   ├── nlp_pipeline.py   # Tokenization, stopword removal, lemmatization
│   └── db_connector.py   # PostgreSQL connection and insertion logic
├── sql/
│   └── schema.sql        # Database schema (banks + reviews tables)
├── tests/                # Unit tests (pytest)
├── scripts/
│   ├── scrape_reviews.py # Scrape + preprocess runner
│   ├── run_analysis.py   # Sentiment + thematic analysis runner
│   └── insert_data.py    # Insert cleaned data into PostgreSQL
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Database Setup (Task 3)

### Prerequisites
- PostgreSQL 14+

### 1. Install PostgreSQL
```bash
sudo apt-get install postgresql postgresql-contrib
```

### 2. Create database and user
```bash
sudo -u postgres createdb bank_reviews
sudo -u postgres psql -c "CREATE USER $USER WITH SUPERUSER;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE bank_reviews TO $USER;"
```

### 3. Apply schema
```bash
psql -d bank_reviews -f sql/schema.sql
```

### 4. Insert data
```bash
python scripts/insert_data.py
```

The script reads `data/processed/reviews_with_sentiment.csv`, populates the `banks` and `reviews` tables, and prints verification queries confirming row counts, average ratings, null checks, and sentiment distribution.

### Verification Query Results

**Reviews per bank**

| bank_name | review_count |
|---|---|
| CBE Mobile Banking | 496 |
| Dashen Bank | 495 |
| Bank of Abyssinia | 489 |
| **Total** | **1,480** |

**Average rating per bank**

| bank_name | avg_rating |
|---|---|
| CBE Mobile Banking | 4.11 |
| Dashen Bank | 3.91 |
| Bank of Abyssinia | 3.53 |

**Null check on key columns**

| null_review_text | null_rating | null_sentiment_label |
|---|---|---|
| 0 | 0 | 0 |

**Sentiment distribution**

| sentiment_label | count |
|---|---|
| positive | 912 |
| negative | 523 |
| neutral | 45 |

### Schema

**banks**

| Column | Type | Description |
|---|---|---|
| bank_id | SERIAL PK | Auto-increment identifier |
| bank_name | VARCHAR(100) UNIQUE | e.g. "CBE Mobile Banking" |
| app_name | VARCHAR(100) | Google Play app identifier |

**reviews**

| Column | Type | Description |
|---|---|---|
| review_id | VARCHAR(50) PK | e.g. "review_1" |
| bank_id | INTEGER FK | References banks.bank_id |
| review_text | TEXT | Cleaned review content |
| rating | SMALLINT | 1–5 stars |
| review_date | DATE | YYYY-MM-DD |
| sentiment_label | VARCHAR(10) | positive / negative / neutral |
| sentiment_score | NUMERIC(6,4) | DistilBERT confidence score |
| identified_theme | VARCHAR(50) | Assigned theme category |
| source | VARCHAR(50) | Default: "Google Play" |

## Usage

Run the full scraping and preprocessing pipeline:

```bash
python scripts/scrape_reviews.py
```

Output is saved to `data/processed/reviews_clean.csv` with columns: `review, rating, date, bank, source`.

---

## Data Collection Methodology

### Tool

Reviews were collected using [`google-play-scraper`](https://github.com/JoMingyu/google-play-scraper), a Python library that scrapes publicly available review data from the Google Play Store without requiring an API key.

### Target Apps

| Bank | App Name | Google Play App ID |
|---|---|---|
| Commercial Bank of Ethiopia | CBE Mobile Banking | `com.combanketh.mobilebanking` |
| Bank of Abyssinia | BoA Mobile | `com.boa.boaMobileBanking` |
| Dashen Bank | Dashen Mobile | `com.dashen.dashensuperapp` |

### Scraping Parameters

- **Language:** English (`lang="en"`)
- **Country:** United States (`country="us"`)
- **Sort order:** Newest first (`Sort.NEWEST`)
- **Reviews per bank:** 500 (target), resulting in 500 collected per bank

### Date Range

Reviews were collected on **13 May 2026**. The resulting dataset spans:

- **Earliest review:** 12 February 2025
- **Latest review:** 12 May 2026
- **Range:** approximately 15 months

### Dataset Summary

| Bank | Reviews Collected | Avg Rating | Reviews After Cleaning |
|---|---|---|---|
| CBE Mobile Banking | 500 | 4.12 | 496 |
| Bank of Abyssinia | 500 | 3.56 | 489 |
| Dashen Bank | 500 | 3.92 | 495 |
| **Total** | **1,500** | | **1,480** |

Missing data rate: **1.3%** (20 rows dropped — empty review text after cleaning).

---

## Preprocessing Steps

1. **Deduplication** — removed duplicate reviews by `reviewId` (0 duplicates found).
2. **Missing value handling** — dropped rows where review text or rating was null or empty (20 rows dropped, logged).
3. **Text cleaning** — stripped HTML tags, removed special characters, collapsed whitespace, lowercased.
4. **Date normalization** — converted all dates to `YYYY-MM-DD` format.
5. **Column standardization** — final CSV contains exactly: `review, rating, date, bank, source`.

The cleaned CSV is listed in `.gitignore` and is never committed to the repository.

---

## Limitations

1. **English-only reviews:** `google-play-scraper` was queried with `lang="en"`, so reviews written in Amharic or other languages are excluded. This may under-represent the Ethiopian user base, who may prefer to write in Amharic.

2. **Initial Dashen Bank app ID:** The originally identified app ID (`com.dashen.mobilebankingapp`) returned 0 reviews across all language and country combinations tested. The correct app — **Dashen Mobile** (`com.dashen.dashensuperapp`) — was identified via Play Store search and used instead.

3. **No historical date filtering:** `google-play-scraper` does not support filtering by date range directly. The 500 newest reviews per bank were collected; older reviews beyond this window are not represented.

4. **Rate limiting:** A 1-second delay was added between requests per app to avoid hitting rate limits. For very large collection runs, this may need to be increased.

5. **Review count cap:** The library caps a single request at 200 reviews internally and paginates automatically up to the requested `count`. Availability depends on how many reviews Google has indexed for the given language/country combination.

---

## CI/CD

GitHub Actions runs on every push to `main`, `dev`, and `task-1`:

1. Installs dependencies via `pip install -r requirements.txt`
2. Runs the full test suite via `pytest tests/`
