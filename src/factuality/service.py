import logging
import uuid
import random
import re
import json
import tldextract
import pandas as pd
from typing import List
from urllib.parse import urlparse


from courlan import get_base_url, is_external
import newspaper
import trafilatura
from sqlalchemy.orm import Session
from lingua import Language, LanguageDetectorBuilder


import src.models as models
from src.factuality.schemas import ArticleCreate
from src.factuality_model.client import Client
from src.utils import logger
import src.db.task as db_task
import src.db.site as db_site
import src.db.article as db_article

MAX_FEED_ARTICLES = 10
SUPPORTED_LANGS = ['en', 'ru', 'hi', 'zh-cn',
                   'kk', 'it', 'de', 'ko', 'es', 'fr']


def get_most_matching_row(df, base_url):
    hostname = tldextract.extract(base_url)
    search_pattern = hostname.domain
    matching_rows = df[df['source'].str.contains(search_pattern, case=False)]

    if matching_rows.empty:
        return None

    closest_match = matching_rows['source'].str.len().sort_values().index[0]
    most_matching_row = matching_rows.loc[closest_match]

    return most_matching_row


def transform_scores(scores, category):
    del scores['source']
    if category == 'manipulation':
        return {key.upper().replace('LABELS.', ''): value for key, value in scores.items()}
    else:
        return {key.upper(): value for key, value in scores.items()}


def process_task(db: Session, task_id: uuid.UUID):
    task = db_task.get_task(db, task_id=task_id)

    if task is None:
        return

    try:
        task.status = models.TaskStatus.IN_PROGRESS
        task = db_task.update_task(db, task)
        response = score(db, task.request['url'])

        task.status = models.TaskStatus.COMPLETED
        task.response = response
        task = db_task.update_task(db, task)

    except Exception as e:
        task.status = models.TaskStatus.FAILED
        task.error = {
            "message": str(e)
        }
        db_task.update_task(db, task)


def score(db: Session, url: str):
    logger.info(f"Scoring article at URL: {url}")
    base_url = get_base_url(url)

    site = db_site.get_site(db, url=base_url)
    if site:
        logger.info(f"Found existing site with ID: {site.id}")

    article = db_article.get_article(db, url)

    if article and article.is_scored:
        logger.info("Article already scored.")
        return {"article": article.scores}

    parsed_article = parse_article(url)

    if site is None:
        site = db_site.create_site(db, url=base_url)

    model = Client()
    scores = model.score_articles_concurrently([parsed_article['text']])[0]

    logger.info("Article scoring completed.")

    article_data = {
        'site_id': site.id,
        'url': parsed_article['url'],
        'lang': parsed_article['lang'],
        'title': parsed_article['title'],
        'author': parsed_article['authors'],
        'content': parsed_article['text'],
        'library': parsed_article['library'],
        'is_scored': True,
        'scores': scores
    }

    article = db_article.create_articles(db, [article_data])[0]

    framing_df = pd.read_parquet('framing.parquet')
    manipulation_df = pd.read_parquet('manipulation.parquet')

    framing_df['source'] = framing_df.index
    manipulation_df['source'] = manipulation_df.index

    most_matching_framing = get_most_matching_row(framing_df, base_url)
    most_matching_manipulation = get_most_matching_row(
        manipulation_df, base_url)

    if most_matching_framing is not None:
        framing_scores_transformed = transform_scores(
            most_matching_framing.to_dict(), 'framing')
        site.scores['framing'] = framing_scores_transformed

    if most_matching_manipulation is not None:
        manipulation_scores_transformed = transform_scores(
            most_matching_manipulation.to_dict(), 'manipulation')
        site.scores['manipulation'] = manipulation_scores_transformed

    return {
        "article": scores
    }


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


logger = logging.getLogger(__name__)


def parse_article(url: str):
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        text = clean_text(article.text)
        lang = detect_language(text)
        if lang not in SUPPORTED_LANGS:
            raise Exception(f'Language {lang} not supported')
        return {"library": "newspaper3k", 'title': article.title or "",
                "text": text, "url": url, "authors": ', '.join(article.authors), "lang": lang}
    except Exception as e:
        logger.error(f"Error with newspaper3k for URL {url}: {e}")

    try:
        downloaded = trafilatura.fetch_url(url)
        data = json.loads(trafilatura.extract(
            downloaded, output_format="json", include_comments=False))
        text = clean_text(data['text'])
        lang = detect_language(text)
        if lang not in SUPPORTED_LANGS:
            raise Exception(f'Language {lang} not supported')
        return {"library": "trafilatura", 'title': data.get('title', ""),
                "text": text, "url": url, "authors": data.get('author', ""), "lang": lang}
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

        if article.is_valid_url() and not is_external(url=article.url, reference=url, ignore_suffix=True) and article.url not in urls_set:
            urls_set.add(article.url)

        try:
            parsed_article = parse_article(article.url)
            if parsed_article and parsed_article['text']:
                articles.append(parsed_article)
        except ValueError as e:
            logger.error(f"Error parsing article {article.url}: {e}")

    if base_url and len(articles) < 10:
        articles.extend(parse_feed(url=base_url))

    return articles


LANGUAGE_MAP = {
    'en': Language.ENGLISH,
    'ru': Language.RUSSIAN,
    'hi': Language.HINDI,
    'zh-cn': Language.CHINESE,
    'kk': Language.KAZAKH,
    "it": Language.ITALIAN,
    "de": Language.GERMAN,
    "ko": Language.KOREAN,
    "es": Language.SPANISH,
    "fr": Language.FRENCH
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
