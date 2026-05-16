import time
import logging
import pandas as pd
from google_play_scraper import reviews, Sort

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

APPS = {
    "CBE Mobile Banking": "com.combanketh.mobilebanking",
    "Bank of Abyssinia": "com.boa.boaMobileBanking",
    "Dashen Bank": "com.dashen.dashensuperapp",
}


def scrape_app_reviews(
    app_id: str,
    app_name: str,
    count: int = 400,
    lang: str = "en",
    country: str = "us",
) -> pd.DataFrame:
    logger.info(f"Scraping {count} reviews for {app_name} ({app_id})")
    try:
        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=count,
        )
    except Exception as exc:
        logger.error(f"Failed to scrape {app_name}: {exc}")
        return pd.DataFrame()

    if not result:
        logger.warning(
            f"No reviews returned for {app_name}. "
            "The app may have no reviews in the selected language/country. "
            "Limitation: google-play-scraper only surfaces reviews indexed by Google Play "
            "for the given lang/country combination."
        )
        return pd.DataFrame()

    df = pd.DataFrame(result)
    df["app_name"] = app_name
    df["app_id"] = app_id
    df["source"] = "Google Play"
    logger.info(f"Collected {len(df)} reviews for {app_name}")
    return df


def scrape_all_apps(apps: dict = None, count: int = 400) -> pd.DataFrame:
    if apps is None:
        apps = APPS

    frames = []
    for app_name, app_id in apps.items():
        df = scrape_app_reviews(app_id, app_name, count=count)
        if not df.empty:
            frames.append(df)
        time.sleep(1)  # polite delay between requests

    if not frames:
        logger.warning("No reviews collected for any app.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total reviews collected: {len(combined)}")
    return combined


def save_raw_reviews(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} raw reviews to {filepath}")
