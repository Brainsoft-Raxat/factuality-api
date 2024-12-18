from uuid import UUID

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from src.database import articles, fetch_all, fetch_one
from src.utils import logger


# Create an article
async def create_article(db: AsyncConnection, article_data: dict) -> dict:
    insert_query = (
        insert(articles)
        .values(
            site_id=article_data["site_id"],
            url=article_data["url"],
            lang=article_data["lang"],
            title=article_data.get("title", "Unknown"),
            author=article_data.get("author", "Unknown"),
            library=article_data.get("library", "Unknown"),
            content=article_data["content"],
            is_scored=article_data.get("is_scored", False),
            scores=article_data.get("scores"),
            num_of_tries=article_data.get("num_of_tries", 0),
        )
        .returning(
            articles.c.id,
            articles.c.site_id,
            articles.c.url,
            articles.c.lang,
            articles.c.title,
            articles.c.author,
            articles.c.library,
            articles.c.content,
            articles.c.is_scored,
            articles.c.scores,
            articles.c.num_of_tries,
            articles.c.created_at,
            articles.c.updated_at,
        )
    )

    created_article = await fetch_one(insert_query, db, True)

    if not created_article:
        logger.error("Failed to insert article into the database")
    else:
        logger.info(f"Article created: {created_article}")
    return created_article


# Get an article by ID
async def get_article(db: AsyncConnection, article_id: UUID) -> dict:
    select_query = select(
        articles.c.id,
        articles.c.site_id,
        articles.c.url,
        articles.c.lang,
        articles.c.title,
        articles.c.author,
        articles.c.library,
        articles.c.content,
        articles.c.is_scored,
        articles.c.scores,
        articles.c.num_of_tries,
        articles.c.created_at,
        articles.c.updated_at,
    ).where(articles.c.id == article_id)

    return await fetch_one(select_query, db, True)


# Get an article by URL
async def get_article_by_url(db: AsyncConnection, url: str) -> dict:
    select_query = select(
        articles.c.id,
        articles.c.site_id,
        articles.c.url,
        articles.c.lang,
        articles.c.title,
        articles.c.author,
        articles.c.library,
        articles.c.content,
        articles.c.is_scored,
        articles.c.scores,
        articles.c.num_of_tries,
        articles.c.created_at,
        articles.c.updated_at,
    ).where(articles.c.url == url)

    return await fetch_one(select_query, db, True)


# Update an article
async def update_article(db: AsyncConnection, article: dict) -> dict:
    update_query = (
        update(articles)
        .where(articles.c.id == article["id"])
        .values(
            site_id=article.get("site_id"),
            url=article.get("url"),
            lang=article.get("lang"),
            title=article.get("title"),
            author=article.get("author"),
            library=article.get("library"),
            content=article.get("content"),
            is_scored=article.get("is_scored"),
            scores=article.get("scores"),
            num_of_tries=article.get("num_of_tries"),
        )
        .returning(
            articles.c.id,
            articles.c.site_id,
            articles.c.url,
            articles.c.lang,
            articles.c.title,
            articles.c.author,
            articles.c.library,
            articles.c.content,
            articles.c.is_scored,
            articles.c.scores,
            articles.c.num_of_tries,
            articles.c.created_at,
            articles.c.updated_at,
        )
    )

    updated_article = await fetch_one(update_query, db, True)

    if not updated_article:
        logger.error(f"Failed to update article {article['id']} in the database")
    else:
        logger.info(f"Article updated: {updated_article}")

    return updated_article


# Delete an article
async def delete_article(db: AsyncConnection, article_id: UUID) -> bool:
    delete_query = delete(articles).where(articles.c.id == article_id)
    result = await db.execute(delete_query)
    await db.commit()

    if result.rowcount == 0:
        logger.error(f"Failed to delete article {article_id}")
        return False
    logger.info(f"Article {article_id} deleted successfully")
    return True


# List all articles
async def list_articles(db: AsyncConnection) -> list[dict]:
    select_query = select(
        articles.c.id,
        articles.c.site_id,
        articles.c.url,
        articles.c.lang,
        articles.c.title,
        articles.c.author,
        articles.c.library,
        articles.c.content,
        articles.c.is_scored,
        articles.c.scores,
        articles.c.num_of_tries,
        articles.c.created_at,
        articles.c.updated_at,
    )

    return await fetch_all(select_query, db)
