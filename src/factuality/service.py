import logging
import uuid
import random
import re
import json
from typing import List


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
SUPPORTED_LANGS = ['en', 'ru', 'hi', 'zh-cn', 'kk']


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

    if site and article and site.scores is not None and article.is_scored:
        logger.info("Article already scored.")
        return {"site": site.scores, "article": article.scores}

    parsed_article = parse_article(url)
    parsed_articles = [parsed_article]

    if site is None:
        site = db_site.create_site(db, url=base_url)

    if site and site.scores is None:
        feed_articles = parse_feed(base_url=base_url, url=url)

        parsed_articles.extend(feed_articles)

    input_texts: List[str] = [parsed_article['text']
                              for parsed_article in parsed_articles]

    model = Client()

    resulting_scores = []

    for i, input_text in enumerate(input_texts):
        payload = {"inputs": input_text}

        try:
            response = model.score_articles(payload)
            logger.info(f"Successfully scored article {i+1}/{len(input_texts)}")
            if response:
                score_dict = {"label0": 0.0, "label1": 0.0, "label2": 0.0}
                for item in response[0]:
                    if item['label'] == 'contradiction':
                        score_dict['label0'] = item['score']
                    elif item['label'] == 'neutral':
                        score_dict['label1'] = item['score']
                    elif item['label'] == 'entailment':
                        score_dict['label2'] = item['score']
                resulting_scores.append(score_dict)
            else:
                resulting_scores.append(
                    {"label0": 0.0, "label1": 0.0, "label2": 0.0})
        except Exception as e:
            if i == 0:
                logger.error(f"Error scoring article {i+1}/{len(input_texts)}: {str(e)}")
                raise Exception(f"Failed to get results from model: {str(e)}")
            else:
                logger.error(f"Error scoring article {i+1}/{len(input_texts)}: {str(e)}")
                resulting_scores.append(
                    {"label0": 0.0, "label1": 0.0, "label2": 0.0})

    logger.info("Article scoring completed.")

    articles_data: List[ArticleCreate] = []
    for i, parsed_article in enumerate(parsed_articles):
        scores = resulting_scores[i]
        if any(score > 0.0 for score in scores.values()):
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
            articles_data.append(article_data)

    if len(resulting_scores) == 0:
        raise Exception(
            "failed to get model responses for articles. len(resulting_scores) = 0")

    articles = db_article.create_articles(db, articles_data)

    sums = {label: sum(score[label] for score in resulting_scores) for label in [
        'label0', 'label1', 'label2']}

    if site.scores:
        for label in sums:
            sums[label] += site.scores[label]

    total = sum(sums.values())

    site_score = {label: sums[label] / total for label in sums}

    site = db_site.update_site(db, site.id, new_scores=site_score)

    return {
        "article": resulting_scores[0],
        "site": site_score
    }


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


logger = logging.getLogger(__name__)
SUPPORTED_LANGS = ['en', 'ru', 'hi', 'zh-cn', 'kk']


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
    'kk': Language.KAZAKH
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
