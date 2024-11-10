import logging
import uuid
import random
import re
import json
import tldextract
import numpy as np
import pandas as pd
import asyncio
from typing import List, Dict
from urllib.parse import urlparse

from courlan import get_base_url, is_external
import newspaper
import trafilatura
from sqlalchemy.orm import Session
from lingua import Language, LanguageDetectorBuilder


import src.models as models
from src.factuality.schemas import ArticleCreate
from src.factuality_model.client import AsyncClient
from src.utils import logger
import src.db.task as db_task
import src.db.site as db_site
import src.db.article as db_article

MAX_FEED_ARTICLES = 10
SUPPORTED_LANGS = ["en", "ru", "hi", "zh-cn", "kk", "it", "de", "ko", "es", "fr"]


def get_most_matching_row(df, base_url):
    hostname = tldextract.extract(base_url)
    search_pattern = hostname.domain
    matching_rows = df[df["source"].str.contains(search_pattern, case=False)]

    if matching_rows.empty:
        return None

    closest_match = matching_rows["source"].str.len().sort_values().index[0]
    most_matching_row = matching_rows.loc[closest_match]

    return most_matching_row


def transform_scores(scores, category):
    del scores["source"]
    if category == "manipulation":
        return {
            key.upper().replace("LABELS.", ""): value for key, value in scores.items()
        }
    else:
        return {key.upper(): value for key, value in scores.items()}


def process_task(db: Session, task_id: uuid.UUID):
    task = db_task.get_task(db, task_id=task_id)

    if task is None:
        return

    try:
        task.status = models.TaskStatus.IN_PROGRESS
        task = db_task.update_task(db, task)

        response = score(db, task.request["url"])

        task.status = models.TaskStatus.COMPLETED
        task.response = response
        task = db_task.update_task(db, task)

    except Exception as e:
        logging.error(f"Error processing task {task_id}: {e}")
        task.status = models.TaskStatus.FAILED
        task.error = {"message": str(e)}
        db_task.update_task(db, task)


def load_framing_scores(base_url: str) -> List[Dict[str, float]]:
    framing_df = pd.read_parquet("framing.parquet")
    framing_df["source"] = framing_df.index
    most_matching_framing = get_most_matching_row(framing_df, base_url)
    if isinstance(most_matching_framing, pd.Series):
        return [
            {"label": label, "score": most_matching_framing[label]}
            for label in most_matching_framing.index
            if label != "source"
        ]
    return []


# Helper function to load manipulation scores similarly
def load_manipulation_scores(base_url: str) -> List[Dict[str, float]]:
    manipulation_df = pd.read_parquet("manipulation.parquet")
    manipulation_df["source"] = manipulation_df.index
    most_matching_manipulation = get_most_matching_row(manipulation_df, base_url)
    if isinstance(most_matching_manipulation, pd.Series):
        return [
            {"label": label, "score": most_matching_manipulation[label]}
            for label in most_matching_manipulation.index
            if label != "source"
        ]
    return []


def load_corpus_scores(base_url: str) -> Dict[str, List[Dict[str, float]]]:
    corpus_df = pd.read_csv("corpus.tsv", delimiter="\t")

    hostname = tldextract.extract(base_url)
    search_pattern = hostname.domain

    matching_rows = corpus_df[
        corpus_df["source_url"].str.contains(search_pattern, case=False, na=False)
        | corpus_df["source_url_normalized"].str.contains(
            search_pattern, case=False, na=False
        )
    ]

    if matching_rows.empty:
        return {"factuality": [], "bias": []}

    closest_match_index = (
        matching_rows[["source_url", "source_url_normalized"]]
        .apply(
            lambda row: min(len(row["source_url"]), len(row["source_url_normalized"])),
            axis=1,
        )
        .idxmin()
    )

    most_matching_row = matching_rows.loc[closest_match_index]

    fact_score = [{"label": most_matching_row["fact"], "score": 1.0}]
    bias_score = [{"label": most_matching_row["bias"], "score": 1.0}]

    return {"factuality": fact_score, "bias": bias_score}


def ensure_json_serializable(data):
    """Recursively convert any int64 or float64 to standard int or float."""
    if isinstance(data, list):
        return [ensure_json_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: ensure_json_serializable(value) for key, value in data.items()}
    else:
        return data


def softmax(logits: List[float]) -> List[float]:
    """Convert logits to probabilities using the softmax function."""
    exp_values = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return (exp_values / exp_values.sum()).tolist()


def process_scores(scores: Dict[str, List[Dict[str, float]]]) -> Dict[str, List[Dict[str, float]]]:
    """Process scores to select top 5 and convert logits to probabilities for persuasion."""
    
    processed_scores = {}

    for category, score_list in scores.items():
        # Ensure score_list is a list; if not, skip this category
        if not isinstance(score_list, list):
            logger.error(f"Expected list for category '{category}', but got {type(score_list)}.")
            processed_scores[category] = []
            continue

        # Convert persuasion logits to probabilities and get top 5
        if category == "persuasion" and score_list:
            logits = [item["score"] for item in score_list]
            probabilities = softmax(logits)
            score_list = [
                {"label": score_list[i]["label"], "score": probabilities[i]}
                for i in range(len(score_list))
            ]
            processed_scores[category] = sorted(score_list, key=lambda x: x["score"], reverse=True)[:5]
        
        # For other categories, just get the top 5 scores by value
        else:
            processed_scores[category] = sorted(score_list, key=lambda x: x["score"], reverse=True)[:5]

    return processed_scores



def score(db: Session, url: str):
    logger.info(f"Scoring article at URL: {url}")
    base_url = get_base_url(url)

    # Check if site and article already exist
    site = db_site.get_site(db, url=base_url)
    if site:
        logger.info(f"Found existing site with ID: {site.id}")

    article = db_article.get_article(db, url)
    if article and article.is_scored:
        logger.info("Article already scored.")
        return {"article": article.scores}

    # Parse article content
    parsed_article = parse_article(url)
    if site is None:
        site = db_site.create_site(db, url=base_url)

    # Initialize AsyncClient and fetch scores
    model = AsyncClient()
    try:
        scores = asyncio.run(model.get_all_scores(parsed_article["text"]))
        if not scores:
            raise ValueError("No scores returned from API.")
    except Exception as e:
        logger.error(f"Error fetching scores: {e}")
        scores = {}  # Fallback to loading from local files

    # Load scores from local files if API scores are missing
    if not scores.get("factuality"):
        scores["factuality"] = load_corpus_scores(base_url).get("factuality", [])
    if not scores.get("bias"):
        scores["bias"] = load_corpus_scores(base_url).get("bias", [])
    if not scores.get("framing"):
        scores["framing"] = load_framing_scores(base_url)
    if not scores.get("persuasion"):
        scores["persuasion"] = load_manipulation_scores(base_url)
    if not scores.get("genre"):
        scores["genre"] = []  # Ensure missing category is empty

    logger.info("Article scoring completed.")

    # Convert scores to JSON-serializable format and process top scores
    scores = ensure_json_serializable(scores)
    scores = process_scores(scores)

    # Prepare article data with scores
    article_data = {
        "site_id": site.id,
        "url": parsed_article["url"],
        "lang": parsed_article["lang"],
        "title": parsed_article["title"],
        "author": parsed_article["authors"],
        "content": parsed_article["text"],
        "library": parsed_article["library"],
        "is_scored": True,
        "scores": scores,  # Now ensured to be JSON-serializable and processed
    }

    # Save the article in the database
    article = db_article.create_articles(db, [article_data])[0]
    return {"article": scores}


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def parse_article(url: str):
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        text = clean_text(article.text)
        lang = detect_language(text)
        if lang not in SUPPORTED_LANGS:
            raise Exception(f"Language {lang} not supported")
        return {
            "library": "newspaper3k",
            "title": article.title or "",
            "text": text,
            "url": url,
            "authors": ", ".join(article.authors),
            "lang": lang,
        }
    except Exception as e:
        logger.error(f"Error with newspaper3k for URL {url}: {e}")

    try:
        downloaded = trafilatura.fetch_url(url)
        data = json.loads(
            trafilatura.extract(
                downloaded, output_format="json", include_comments=False
            )
        )
        text = clean_text(data["text"])
        lang = detect_language(text)
        if lang not in SUPPORTED_LANGS:
            raise Exception(f"Language {lang} not supported")
        return {
            "library": "trafilatura",
            "title": data.get("title", ""),
            "text": text,
            "url": url,
            "authors": data.get("author", ""),
            "lang": lang,
        }
    except Exception as e:
        logger.error(f"Error with trafilatura for URL {url}: {e}")

    raise ValueError("Failed to parse the article with both libraries.")


def parse_feed(url: str, base_url: str = None) -> list:
    articles = []
    urls_set = set()
    feed = newspaper.build(url, memoize_articles=False)

    articles_list = list(feed.articles)
    random.shuffle(articles_list)

    for article in articles_list:
        if len(articles) >= MAX_FEED_ARTICLES:
            break

        if (
            article.is_valid_url()
            and not is_external(url=article.url, reference=url, ignore_suffix=True)
            and article.url not in urls_set
        ):
            urls_set.add(article.url)

        try:
            parsed_article = parse_article(article.url)
            if parsed_article and parsed_article["text"]:
                articles.append(parsed_article)
        except ValueError as e:
            logger.error(f"Error parsing article {article.url}: {e}")

    if base_url and len(articles) < 10:
        articles.extend(parse_feed(url=base_url))

    return articles


LANGUAGE_MAP = {
    "en": Language.ENGLISH,
    "ru": Language.RUSSIAN,
    "hi": Language.HINDI,
    "zh-cn": Language.CHINESE,
    "kk": Language.KAZAKH,
    "it": Language.ITALIAN,
    "de": Language.GERMAN,
    "ko": Language.KOREAN,
    "es": Language.SPANISH,
    "fr": Language.FRENCH,
}

supported_languages = [LANGUAGE_MAP[lang] for lang in SUPPORTED_LANGS]
detector = LanguageDetectorBuilder.from_languages(*supported_languages).build()


def detect_language(text):
    try:
        sample_text = text[:500]

        detected_language = detector.detect_language_of(sample_text)

        for code, lang in LANGUAGE_MAP.items():
            if lang == detected_language:
                return code
        return None
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return None
