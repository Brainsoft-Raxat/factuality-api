from courlan import get_base_url
from sqlalchemy.ext.asyncio import AsyncConnection

import src.articles.service as articles_service
import src.sites.service as sites_service
from src.ml_models.client import AsyncClient
from src.score.parser import parse_article_async
from src.score.utils import (
    add_noise_to_scores,
    get_top_k_scores,
    load_web_source_scores,
)


async def get_model_scores(text):
    model = AsyncClient()
    return await model.get_all_scores(text)


NOISE_LEVEL = 0.1


async def score(db: AsyncConnection, url: str):
    base_url = get_base_url(url)

    res = {"article": {}, "site": {}}
    metrics = set(["factuality", "bias", "genre", "framing", "persuasion"])

    site = await sites_service.get_site_by_base_url(db, base_url)
    if not site:
        site_scores = load_web_source_scores(base_url)
        site = await sites_service.create_site(
            db, {"url": base_url, "scores": site_scores}
        )

    article_scores = {
        "factuality": [],
        "bias": [],
        "framing": [],
        "persuasion": [],
        "genre": [],
    }

    for metric in site["scores"]:
        if len(site["scores"][metric]) > 0:
            article_scores[metric] = add_noise_to_scores(
                site["scores"][metric], NOISE_LEVEL
            )
            metrics.remove(metric)

    article = await articles_service.get_article_by_url(db, url)
    if not article:
        parsed_article = await parse_article_async(url)
        if not parsed_article or not parsed_article.get("text", ""):
            raise ValueError("Failed to parse article. Web source is blocked.")

        model = AsyncClient()

        new_scores = await model.get_all_scores(parsed_article["text"], list(metrics))

        article_scores = article_scores | new_scores

        article_scores["framing"] = get_top_k_scores(article_scores["framing"], 5)
        article_scores["persuasion"] = get_top_k_scores(article_scores["persuasion"], 5)

        article = await articles_service.create_article(
            db,
            {
                "site_id": site["id"],
                "url": url,
                "lang": parsed_article["lang"],
                "title": parsed_article["title"],
                "author": parsed_article["authors"],
                "library": parsed_article["library"],
                "content": parsed_article["text"],
                "is_scored": True,
                "scores": article_scores,
            },
        )

    res["article"] = article["scores"]
    res["site"] = site["scores"]

    return res
