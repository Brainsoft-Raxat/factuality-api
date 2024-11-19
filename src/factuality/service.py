import asyncio
import json
import logging
import random
import re
import uuid
from typing import Dict, List

import newspaper
import numpy as np
import pandas as pd
import trafilatura
from courlan import get_base_url, is_external
from lingua import Language, LanguageDetectorBuilder
from sqlalchemy.orm import Session

import src.db.article as db_article
import src.db.site as db_site
import src.db.task as db_task
import src.models as models
from src.factuality.parser import (
    normalize_source_url_for_corpus,
    normalize_url_for_framing_and_manipulation,
)
from src.factuality_model.client import AsyncClient
from src.utils import logger

MAX_FEED_ARTICLES = 3
SUPPORTED_LANGS = ["en", "ru", "hi", "zh-cn", "kk", "it", "de", "ko", "es", "fr"]


def softmax(scores):
    """Apply softmax to convert logits to probabilities"""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def get_top_k_scores(scores_list, k=5):
    """Get top k scores after normalizing and sorting"""

    labels = [item["label"] for item in scores_list]
    scores = np.array([item["score"] for item in scores_list])

    probabilities = softmax(scores)

    scored_labels = list(zip(labels, probabilities))
    sorted_scores = sorted(scored_labels, key=lambda x: x[1], reverse=True)

    return [
        {"label": label, "score": float(score)} for label, score in sorted_scores[:k]
    ]


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
    """Load framing scores for a given URL by matching the exact source."""
    # Load framing data
    framing_df = pd.read_parquet("framing.parquet")

    # Normalize the base URL to find the corresponding source
    source = normalize_url_for_framing_and_manipulation(base_url)

    # Find the exact match for the source in the DataFrame
    matching_rows = framing_df[framing_df["source"] == source]

    # If an exact match is found, extract the scores
    if not matching_rows.empty:
        most_matching_framing = matching_rows.iloc[0]
        scores = [
            {"label": normalize_label(label), "score": most_matching_framing[label]}
            for label in most_matching_framing.index
            if label != "source"
        ]
        return get_top_k_scores(scores, k=5)

    # Return empty list if no match is found
    return []


def load_manipulation_scores(base_url: str) -> List[Dict[str, float]]:
    """Load manipulation scores for a given URL by matching the exact source."""
    # Load manipulation data
    manipulation_df = pd.read_parquet("manipulation.parquet")

    # Normalize the base URL to find the corresponding source
    source = normalize_url_for_framing_and_manipulation(base_url)

    # Find the exact match for the source in the DataFrame
    matching_rows = manipulation_df[manipulation_df["source"] == source]

    # If an exact match is found, extract the scores
    if not matching_rows.empty:
        most_matching_manipulation = matching_rows.iloc[0]
        scores = [
            {
                "label": normalize_label(label),
                "score": most_matching_manipulation[label],
            }
            for label in most_matching_manipulation.index
            if label != "source"
        ]
        return get_top_k_scores(scores, k=5)

    # Return empty list if no match is found
    return []


def load_corpus_scores(source_url: str) -> Dict[str, List[Dict[str, float]]]:
    """Load corpus scores by matching exact source_url or source_url_normalized."""

    corpus_df = pd.read_csv("corpus.tsv", delimiter="\t")

    source_url_normalized = normalize_source_url_for_corpus(source_url)

    matching_rows = corpus_df[
        (corpus_df["source_url"] == source_url)
        | (corpus_df["source_url_normalized"] == source_url_normalized)
    ]

    if matching_rows.empty:
        return {"factuality": [], "bias": []}

    most_matching_row = matching_rows.iloc[0]

    fact_score = [{"label": normalize_label(most_matching_row["fact"]), "score": 1.0}]
    bias_score = [{"label": normalize_label(most_matching_row["bias"]), "score": 1.0}]

    return {"factuality": fact_score, "bias": bias_score}


def process_scores(
    scores: Dict[str, List[Dict[str, float]]],
) -> Dict[str, List[Dict[str, float]]]:
    """Process scores to select top 5 and convert logits to probabilities for persuasion."""

    processed_scores = {}

    for category, score_list in scores.items():
        if not isinstance(score_list, list):
            logger.error(
                f"Expected list for category '{category}', but got {type(score_list)}."
            )
            processed_scores[category] = []
            continue

        if category == "persuasion" and score_list:
            logits = [item["score"] for item in score_list]
            probabilities = softmax(logits)
            score_list = [
                {"label": score_list[i]["label"], "score": probabilities[i]}
                for i in range(len(score_list))
            ]
            processed_scores[category] = sorted(
                score_list, key=lambda x: x["score"], reverse=True
            )[:5]

        else:
            processed_scores[category] = sorted(
                score_list, key=lambda x: x["score"], reverse=True
            )[:5]

    return processed_scores


def ensure_json_serializable(obj):
    """Convert numpy types to standard Python types"""
    if isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def normalize_label(label: str) -> str:
    """Normalize label to match article format: lowercase with underscores"""
    label = label.replace("labels.", "")
    return label.lower().replace(" ", "_").replace("-", "_")


def merge_duplicate_labels(scores_list):
    """Merge scores for duplicate labels after normalization"""
    label_scores = {}

    for item in scores_list:
        label = normalize_label(item["label"])
        if label not in label_scores:
            label_scores[label] = []
        label_scores[label].append(float(item["score"]))

    return [
        {"label": label, "score": float(sum(scores) / len(scores))}
        for label, scores in label_scores.items()
    ]


def score(db: Session, url: str):
    logger.info(f"=== Starting scoring process for URL: {url} ===")
    base_url = get_base_url(url)
    logger.info(f"Base URL extracted: {base_url}")

    default_site_scores = {
        "factuality": [],
        "bias": [],
        "framing": [],
        "persuasion": [],
        "genre": [],
    }

    site = db_site.get_site(db, url=base_url)
    if site:
        logger.info(f"Found existing site in database with ID: {site.id}")

        if site.scores is None:
            site.scores = default_site_scores
        else:
            for category in default_site_scores:
                if category not in site.scores:
                    site.scores[category] = []

    article = db_article.get_article(db, url)
    if article and article.is_scored:
        logger.info("Article already exists and is scored - returning cached scores")
        return {
            "article": article.scores or {},
            "site": site.scores or default_site_scores,
        }

    logger.info("Starting article parsing...")
    parsed_article = parse_article(url)
    logger.info(
        f"Article parsed successfully using {parsed_article['library']} library"
    )
    parsed_articles = [parsed_article]

    if site is None:
        logger.info("Creating new site record in database")
        site = db_site.create_site(db, url=base_url)
        site.scores = default_site_scores.copy()
        logger.info(f"Created new site with ID: {site.id}")

    logger.info("Loading pre-computed scores from local data sources...")
    site_scores = {
        "factuality": load_corpus_scores(base_url).get("factuality", []),
        "bias": load_corpus_scores(base_url).get("bias", []),
        "framing": load_framing_scores(base_url),
        "persuasion": load_manipulation_scores(base_url),
        "genre": [],
    }

    for category, scores in site_scores.items():
        if scores:
            logger.info(f"Found pre-computed {category} scores: {len(scores)} labels")
        else:
            logger.info(f"No pre-computed scores found for {category}")

    logger.info("Normalizing and merging duplicate labels in site scores...")
    for category in site_scores:
        if site_scores[category]:
            site_scores[category] = merge_duplicate_labels(site_scores[category])

    missing_categories = [
        category
        for category, scores in site_scores.items()
        if not scores or scores is None
    ]

    if missing_categories:
        logger.info(
            f"Need to calculate scores for missing categories: {missing_categories}"
        )
        logger.info("Starting feed parsing...")
        feed_articles = parse_feed(url=base_url)
        logger.info(f"Found {len(feed_articles)} articles in feed")
        parsed_articles.extend(feed_articles[:5])
        logger.info(f"Using {len(parsed_articles)} articles for analysis")

        logger.info("Starting batch scoring of articles...")
        model = AsyncClient()
        all_article_scores = []

        for i, parsed_article in enumerate(parsed_articles, 1):
            try:
                logger.info(f"Scoring article {i}/{len(parsed_articles)}")
                scores = asyncio.run(model.get_all_scores(parsed_article["text"]))
                if scores and isinstance(scores, dict):
                    scores.pop("execution_time", None)
                    scores = ensure_json_serializable(scores)
                    for category in scores:
                        scores[category] = merge_duplicate_labels(scores[category])
                    all_article_scores.append(scores)
                    logger.info(f"Successfully scored article {i}")
            except Exception as e:
                logger.error(f"Error scoring article {i}: {str(e)}")

        if all_article_scores:
            logger.info("Calculating aggregated scores for missing categories...")
            aggregated_scores = {}

            first_scores = all_article_scores[0]
            for category in missing_categories:
                if category in first_scores:
                    logger.info(f"Aggregating scores for {category}")
                    aggregated_scores[category] = {}
                    for score_item in first_scores[category]:
                        label = normalize_label(score_item["label"])
                        aggregated_scores[category][label] = []

            for article_scores in all_article_scores:
                for category in missing_categories:
                    if category in article_scores:
                        for score_item in article_scores[category]:
                            label = normalize_label(score_item["label"])
                            score = float(score_item["score"])
                            if (
                                category in aggregated_scores
                                and label in aggregated_scores[category]
                            ):
                                aggregated_scores[category][label].append(score)

            for category in aggregated_scores:
                logger.info(f"Finalizing aggregated scores for {category}")
                site_scores[category] = []
                for label, scores in aggregated_scores[category].items():
                    if scores:
                        avg_score = float(sum(scores) / len(scores))
                        site_scores[category].append(
                            {"label": label, "score": avg_score}
                        )

    logger.info("Processing final scores...")
    site_scores = ensure_json_serializable(site_scores)

    if site_scores.get("framing"):
        logger.info("Applying top-k and softmax to framing scores")
        site_scores["framing"] = get_top_k_scores(
            merge_duplicate_labels(site_scores["framing"]), k=5
        )
    if site_scores.get("persuasion"):
        logger.info("Applying top-k and softmax to persuasion scores")
        site_scores["persuasion"] = get_top_k_scores(
            merge_duplicate_labels(site_scores["persuasion"]), k=5
        )

    logger.info("Scoring individual article...")
    model = AsyncClient()
    article_scores = {}
    try:
        scores = asyncio.run(model.get_all_scores(parsed_article["text"]))
        if scores and isinstance(scores, dict):
            scores.pop("execution_time", None)
            article_scores = ensure_json_serializable(scores)
            logger.info("Successfully scored individual article")
    except Exception as e:
        logger.error(f"Error scoring individual article: {str(e)}")

    logger.info("Updating site scores in database...")
    site = db_site.update_site(db, site.id, new_scores=site_scores)
    logger.info("Site scores updated successfully")

    logger.info("Creating article record in database...")
    article_data = {
        "site_id": site.id,
        "url": parsed_article["url"],
        "lang": parsed_article["lang"],
        "title": parsed_article["title"],
        "author": parsed_article["authors"],
        "content": parsed_article["text"],
        "library": parsed_article["library"],
        "is_scored": True,
        "scores": article_scores,
    }

    article = db_article.create_articles(db, [article_data])[0]
    logger.info(f"Created article record with ID: {article.id}")

    final_site_scores = site_scores or default_site_scores
    for category in default_site_scores:
        if category not in final_site_scores:
            final_site_scores[category] = []

    final_article_scores = article_scores if article_scores else {}

    logger.info("=== Scoring process completed ===")
    return {"article": final_article_scores, "site": final_site_scores}


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
