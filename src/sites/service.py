from uuid import UUID

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from src.database import fetch_all, fetch_one, sites
from src.utils import logger


# Create a site
async def create_site(db: AsyncConnection, site_data: dict) -> dict:
    insert_query = (
        insert(sites)
        .values(
            url=site_data["url"],
            scores=site_data.get("scores"),
        )
        .returning(
            sites.c.id,
            sites.c.url,
            sites.c.scores,
            sites.c.created_at,
            sites.c.updated_at,
        )
    )

    created_site = await fetch_one(insert_query, db, True)

    if not created_site:
        logger.error("Failed to insert site into the database")
    else:
        logger.info(f"Site created: {created_site}")
    return created_site


# Get a site by ID
async def get_site(db: AsyncConnection, site_id: UUID) -> dict:
    select_query = select(
        sites.c.id,
        sites.c.url,
        sites.c.scores,
        sites.c.created_at,
        sites.c.updated_at,
    ).where(sites.c.id == site_id)

    return await fetch_one(select_query, db, True)


# Get a site by base URL
async def get_site_by_base_url(db: AsyncConnection, base_url: str) -> dict:
    select_query = select(
        sites.c.id,
        sites.c.url,
        sites.c.scores,
        sites.c.created_at,
        sites.c.updated_at,
    ).where(sites.c.url == base_url)

    return await fetch_one(select_query, db, True)


# Update a site
async def update_site(db: AsyncConnection, site: dict) -> dict:
    update_query = (
        update(sites)
        .where(sites.c.id == site["id"])
        .values(
            url=site.get("url"),
            scores=site.get("scores"),
        )
        .returning(
            sites.c.id,
            sites.c.url,
            sites.c.scores,
            sites.c.created_at,
            sites.c.updated_at,
        )
    )

    updated_site = await fetch_one(update_query, db, True)

    if not updated_site:
        logger.error(f"Failed to update site {site['id']} in the database")
    else:
        logger.info(f"Site updated: {updated_site}")

    return updated_site


# Delete a site
async def delete_site(db: AsyncConnection, site_id: UUID) -> bool:
    delete_query = delete(sites).where(sites.c.id == site_id)
    result = await db.execute(delete_query)
    await db.commit()

    if result.rowcount == 0:
        logger.error(f"Failed to delete site {site_id}")
        return False
    logger.info(f"Site {site_id} deleted successfully")
    return True


# List all sites
async def list_sites(db: AsyncConnection) -> list[dict]:
    select_query = select(
        sites.c.id,
        sites.c.url,
        sites.c.scores,
        sites.c.created_at,
        sites.c.updated_at,
    )

    return await fetch_all(select_query, db)
