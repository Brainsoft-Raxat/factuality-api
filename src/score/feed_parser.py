import asyncio
import random
from typing import List

import newspaper
from courlan import is_external
from fastapi.concurrency import run_in_threadpool

from src.score.parser import parse_article
from src.utils import logger

MAX_FEED_ARTICLES = 3
MAX_CONCURRENCY = 10


async def parse_single_article(url: str):
    try:
        result = await run_in_threadpool(parse_article, url)
        if result and result.get("text"):
            return result
        return None
    except ValueError as e:
        logger.error(f"Error parsing article {url}: {e}")
        return None


async def parse_feed_async(
    base_url: str, max_articles: int = MAX_FEED_ARTICLES
) -> List[dict]:
    feed = await run_in_threadpool(newspaper.build, base_url, memoize_articles=False)

    articles_list = list(feed.articles)
    random.shuffle(articles_list)

    urls_to_parse = []
    seen_urls = set()
    for article in articles_list:
        if len(urls_to_parse) >= max_articles:
            break
        if (
            article.is_valid_url()
            and not is_external(url=article.url, reference=base_url, ignore_suffix=True)
            and article.url not in seen_urls
        ):
            seen_urls.add(article.url)
            urls_to_parse.append(article.url)

    if not urls_to_parse:
        return []

    concurrency = min(len(urls_to_parse), max_articles, MAX_CONCURRENCY)
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(url: str):
        async with semaphore:
            return await parse_single_article(url)

    tasks = [asyncio.create_task(worker(url)) for url in urls_to_parse]

    results = []
    for future in asyncio.as_completed(tasks):
        res = await future
        if res is not None:
            results.append(res)
        if len(results) >= max_articles:
            for t in tasks:
                if not t.done():
                    t.cancel()
            break

    # If we have fewer articles than desired, we can attempt recursion or another approach,
    # but ensure you have a mechanism to prevent infinite recursion.
    # if base_url and len(results) < max_articles:
    #     remaining = max_articles - len(results)
    #     additional = await parse_feed_async(base_url, remaining)
    #     results.extend(additional)

    return results


def parse_feed(base_url: str) -> List[dict]:
    return asyncio.run(parse_feed_async(base_url))
