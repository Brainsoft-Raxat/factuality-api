import uuid
import src.models as models
from sqlalchemy.orm import Session


def get_site(db: Session, url: str) -> models.Site:
    return db.query(models.Site).filter(models.Site.url == url).first()


def create_site(db: Session, url: str):
    db_site = models.Site(
        id=uuid.uuid4(),
        url=url,
    )
    db.add(db_site)
    db.commit()
    db.refresh(db_site)
    return db_site


def update_site(db: Session, site_id: str, new_scores: dict):
    db_site = db.query(models.Site).filter(
        models.Site.id == site_id).first()
    if db_site:
        db_site.scores = new_scores
        db.commit()
        db.refresh(db_site)
        return db_site
    else:
        return None
