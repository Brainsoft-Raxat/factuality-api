import json

import newspaper
import trafilatura
from fastapi.concurrency import run_in_threadpool

from src.score.utils import clean_text, trim_to_n_words
from src.utils import logger

MAX_WORDS = 100


def parse_article(url: str):
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        text = trim_to_n_words(clean_text(article.text), MAX_WORDS)

        return {
            "library": "newspaper3k",
            "title": article.title or "",
            "text": text,
            "url": url,
            "authors": ", ".join(article.authors),
            "lang": "unknow",
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
        text = trim_to_n_words(clean_text(data["text"]), MAX_WORDS)

        return {
            "library": "trafilatura",
            "title": data.get("title", ""),
            "text": text,
            "url": url,
            "authors": data.get("author", ""),
            "lang": "unknown",
        }
    except Exception as e:
        logger.error(f"Error with trafilatura for URL {url}: {e}")

    raise ValueError("Failed to parse articles. Web source is blocked.")


async def parse_article_async(url: str):
    return await run_in_threadpool(parse_article, url)
