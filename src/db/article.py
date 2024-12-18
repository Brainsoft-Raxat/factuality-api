import uuid
import src.models as models
from sqlalchemy.orm import Session
from typing import List, Dict


def create_articles(db: Session, articles_data: List[Dict]) -> List[models.Article]:
    db_articles = []
    for article_data in articles_data:
        # Filter valid fields and set defaults for missing required fields
        valid_fields = {k: v for k, v in article_data.items() if k in models.Article.__table__.columns}
        db_article = models.Article(id=uuid.uuid4(), **valid_fields)
        
        db.add(db_article)
        db_articles.append(db_article)
        
    db.commit()  # Commit all articles at once for efficiency
    return db_articles


def get_article(db: Session, url: str) -> models.Article:
    return db.query(models.Article).filter(models.Article.url == url).first()


def get_articles_by_urls(db: Session, urls: List[str]) -> List[str]:
    return db.query(models.Article).filter(models.Article.url.in_(urls)).all()


def create_article(db: Session, article_data: Dict) -> models.Article:
    valid_fields = {k: v for k, v in article_data.items(
    ) if k in models.Article.__table__.columns}
    db_article = models.Article(**valid_fields)
    db_article.id = uuid.uuid4()
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article


def update_article(db: Session, article_id: str, article_data: Dict):
    db_article = db.query(models.Article).filter(
        models.Article.id == article_id).first()
    if db_article:
        valid_fields = {k: v for k, v in article_data.items(
        ) if k in models.Article.__table__.columns}
        for key, value in valid_fields.items():
            setattr(db_article, key, value)
        db.commit()
        db.refresh(db_article)
        return db_article
    else:
        return None
