import math
import time
import uuid
import random
from typing import List

from fastapi import HTTPException, status
from courlan import get_base_url, is_external
import newspaper
from sqlalchemy.orm import Session


import src.models as models
from src.factuality.schemas import ArticleCreate
from src.factuality_model.client import Client
from src.utils import logger

MAX_FEED_ARTICLES = 10


def create_task(db: Session, request: dict) -> models.Task:
    try:
        task = models.Task(
            id=uuid.uuid4(),
            status=models.TaskStatus.PENDING,
            request=request,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return task
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def update_task(db: Session, task: models.Task) -> models.Task:
    try:
        db.add(task)
        db.commit()
        db.refresh(task)
        return task
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def get_task(db: Session, task_id: uuid.UUID) -> models.Task:
    return db.query(models.Task).filter(models.Task.id == task_id).first()


def process_task(db: Session, task_id: uuid.UUID):
    db_task = get_task(db, task_id=task_id)

    if db_task is None:
        return

    try:
        db_task.status = models.TaskStatus.IN_PROGRESS
        db_task = update_task(db, db_task)
        response = score(db, db_task.request['url'])

        db_task.status = models.TaskStatus.COMPLETED
        db_task.response = response
        db_task = update_task(db, db_task)

    except HTTPException as e:
        db_task.status = models.TaskStatus.FAILED
        db_task.error = {
            "message": e.detail
        }
        update_task(db, db_task)

    except Exception as e:
        db_task.status = models.TaskStatus.FAILED
        db_task.error = {
            "message": str(e)
        }
        update_task(db, db_task)


def score(db: Session, url: str):
    base_url = get_base_url(url)

    db_site = get_site(db, url=base_url)
    db_article = get_article(db, url)

    if db_site and db_article and db_site.scores is not None and db_article.is_scored:
        return {"site": db_site.scores, "article": db_article.scores}

    article = parse_article(url)

    articles = [article]

    if db_site is None:
        db_site = create_site(db, url=base_url)

    if db_site and db_site.scores is None:
        feed_urls = parse_feed(url=base_url)
        if len(feed_urls) == 0:
            feed_urls = parse_feed(url)

        random.shuffle(feed_urls)

        i = 0
        for feed_url in feed_urls:
            if i >= MAX_FEED_ARTICLES:
                break
            try:
                feed_article = parse_article(feed_url)
            except HTTPException as e:
                logger.error(
                    f"Failed to parse feed article {feed_url}: {e.detail}")
                continue
            except Exception as e:
                logger.error(
                    f"Failed to parse feed article {feed_url}: {str(e)}")
                continue

            i += 1
            articles.append(feed_article)

    input_texts: List[str] = [article.text for article in articles]

    model = Client()

    max_chunk_length = 500
    chunked_texts = []

    for text in input_texts:
        if len(text) > max_chunk_length:
            chunks = [text[i:i + max_chunk_length]
                      for i in range(0, len(text), max_chunk_length)]
            chunked_texts.extend(chunks)
        else:
            chunked_texts.append(text)

    max_batch_size = 100
    resulting_scores = []

    for i in range(0, len(chunked_texts), max_batch_size):
        chunk = chunked_texts[i:i + max_batch_size]
        payload = {
            "inputs": chunk,
            "parameters": {
                'padding': True,
                'truncation': True,
                'max_length': 512
            }
        }

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = model.score_articles(payload)
                if response.status_code != 200:
                    err = response.json()
                    if 'estimated_time' in err:
                        time.sleep(err['estimated_time'])
                        raise HTTPException(
                            status_code=500,
                            detail=err
                        )

                response = response.json()
                resulting_scores.extend(response)
                break
            except HTTPException as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed after {max_retries} retries. Last error: {str(e)}")

    average_scores = []
    current_index = 0

    for text in input_texts:
        if len(text) > max_chunk_length:
            num_chunks = math.ceil(len(text) / max_chunk_length)
            scores_chunked = resulting_scores[current_index: current_index + num_chunks]

            average_score = {
                "label0": 0.0,
                "label1": 0.0,
                "label2": 0.0,
            }

            for score_chunked in scores_chunked:
                average_score["label0"] += score_chunked[0]['score']
                average_score["label1"] += score_chunked[1]['score']
                average_score["label2"] += score_chunked[2]['score']

            sum_of_scores = average_score["label0"] + \
                average_score["label1"] + average_score["label2"]

            average_score["label0"] = average_score['label0']/sum_of_scores
            average_score["label1"] = average_score['label1']/sum_of_scores
            average_score["label2"] = average_score['label2']/sum_of_scores

            average_scores.append(average_score)
            current_index += num_chunks
        else:
            average_score = {
                "label0": resulting_scores[current_index][2]['score'],
                "label1": resulting_scores[current_index][1]['score'],
                "label2": resulting_scores[current_index][0]['score'],
            }
            average_scores.append(resulting_scores[current_index])
            current_index += 1

    if len(articles) != len(average_scores):
        min_length = min(len(articles), len(average_scores))
        articles = articles[:min_length]
        average_scores = average_scores[:min_length]

    articles_data: List[ArticleCreate] = []
    for i, article in enumerate(articles):
        article_data = ArticleCreate(
            site_id=db_site.id,
            url=article.url,
            title=article.title,
            author=", ".join(article.authors),
            content=article.text,
            is_scored=True,
            scores=average_scores[i]
        )

        articles_data.append(article_data)

    if len(average_scores) == 0:
        raise HTTPException(
            status_code=500, detail="failed to get model responses for articles. len(average_scores) = 0")

    db_articles = create_articles(db, articles_data)

    sums = {label: sum(average_score[label] for average_score in average_scores) for label in [
        'label0', 'label1', 'label2']}

    if db_site.scores:
        for label in sums:
            sums[label] += db_site.scores[label]

    total = sum(sums.values())

    site_score = {label: sums[label] / total for label in sums}

    db_site = update_site(db, db_site.id, new_scores=site_score)

    return {
        "article": average_scores[0],
        "site": site_score
    }


def get_site(db: Session, url: str) -> models.Site:
    return db.query(models.Site).filter(models.Site.url == url).first()


def create_site(db: Session, url: str):
    try:
        db_site = models.Site(
            id=uuid.uuid4(),
            url=url,
        )
        db.add(db_site)
        db.commit()
        db.refresh(db_site)
        return db_site
    except Exception as e:
        db.rollback()
        raise e


def update_site(db: Session, site_id: str, new_scores: dict):
    try:
        db_site = db.query(models.Site).filter(
            models.Site.id == site_id).first()
        if db_site:
            db_site.scores = new_scores
            db.commit()
            db.refresh(db_site)
            return db_site
        else:
            return None
    except Exception as e:
        db.rollback()
        raise e


def create_articles(db: Session, articles: List[ArticleCreate]):
    try:
        db_articles = []
        for article in articles:
            db_article = models.Article(
                **article.model_dump(exclude_unset=True, exclude_none=True)
            )
            db_article.id = uuid.uuid4()
            db.add(db_article)
            db_articles.append(db_article)
        db.commit()
        return db_articles
    except Exception as e:
        db.rollback()
        raise e


def get_article(db: Session, url: str) -> models.Article:
    return db.query(models.Article).filter(models.Article.url == url).first()


def get_articles_by_urls(db: Session, urls: List[str]) -> List[str]:
    return db.query(models.Article).filter(models.Article.url.in_(urls)).all()


def create_article(db: Session, article: ArticleCreate) -> models.Article:
    try:
        db_article = models.Article(
            **article.model_dump(exclude_unset=True, exclude_none=True)
        )
        db_article.id = uuid.uuid4()
        db.add(db_article)
        db.commit()
        db.refresh(db_article)
        return db_article
    except Exception as e:
        db.rollback()
        raise e


def update_article(db: Session, article_id: str, article_data: ArticleCreate):
    try:
        db_article = db.query(models.Article).filter(
            models.Article.id == article_id).first()
        if db_article:
            for key, value in article_data.model_dump(exclude_unset=True, exclude_none=True).items():
                setattr(db_article, key, value)
            db.commit()
            db.refresh(db_article)
            return db_article
        else:
            return None
    except Exception as e:
        db.rollback()
        raise e


def parse_article(url: str):
    article = newspaper.Article(url=url)
    article.download()
    article.parse()
    
    # if not article.is_valid_url():
    #     raise HTTPException(status_code=500, detail="Invalid URL")

    if not article.is_valid_body():
        raise HTTPException(
            status_code=500, detail=f'article content is not valid. this is what was parsed: {article.text}')

    return article


def parse_feed(url: str) -> list:
    articles = []
    urls_set = set()
    feed = newspaper.build(url, memoize_articles=False)
    for article in feed.articles:
        raw_article = newspaper.Article(url=article.url)
        if raw_article.is_valid_url() and not is_external(url=article.url, reference=url, ignore_suffix=True) and raw_article.url not in urls_set:
            urls_set.add(raw_article.url)
            articles.append(raw_article.url)

    return articles
