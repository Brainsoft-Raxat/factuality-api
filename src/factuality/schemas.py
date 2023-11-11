from uuid import UUID
from typing import Optional

from pydantic import BaseModel, field_validator, UUID4

from courlan import validate_url, scrub_url

class TaskID(BaseModel):
    task_id: UUID4


class ScoreRequest(BaseModel):
    url: str

    @field_validator("url", mode="after")
    @classmethod
    def validate_url(cls, v: str) -> str:
        ok, obj = validate_url(v)
        if not ok:
            raise ValueError("Invalid URL")
        return scrub_url(v)


class ScoreResponse(BaseModel):
    article: dict[str, float]
    site: dict[str, float]


class ArticleCreate(BaseModel):
    site_id: UUID
    url: str
    title: str
    author: str
    content: str
    is_scored: bool
    scores: Optional[dict[str, float]] = None
    num_of_tries: Optional[int] = None

    @field_validator("url", mode="after")
    @classmethod
    def validate_url(cls, v: str) -> str:
        ok, obj = validate_url(v)
        if not ok:
            raise ValueError("Invalid URL")
        return scrub_url(v)
