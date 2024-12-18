from uuid import UUID
from typing import Optional, Dict, Union, List

from pydantic import BaseModel, validator, Field
from courlan import validate_url, scrub_url


class TaskID(BaseModel):
    task_id: UUID


class ScoreRequest(BaseModel):
    url: str

    @validator("url", allow_reuse=True)
    def validate_url(cls, v: str) -> str:
        ok, obj = validate_url(v)
        if not ok:
            raise ValueError("Invalid URL")
        return scrub_url(v)


class ScoreResponse(BaseModel):
    article: Dict[str, Dict[str, float]]
    site: Dict[str, Dict[str, float]]


class ArticleCreate(BaseModel):
    site_id: UUID
    url: str
    lang: Optional[str] = None  # Assuming language might be relevant
    title: str
    author: Optional[str] = None
    content: str
    is_scored: bool

    scores: Optional[Dict[str, Dict[str, float]]] = None
    num_of_tries: Optional[int] = None

    @validator("url", allow_reuse=True)
    def validate_url(cls, v: str) -> str:
        ok, obj = validate_url(v)
        if not ok:
            raise ValueError("Invalid URL")
        return scrub_url(v)


# Adjusting Score and ScoreData to fit the detailed structure.
class ScoreCategory(BaseModel):
    label: str
    score: float


class ScoreData(BaseModel):
    factuality: List[ScoreCategory] = None 
    bias: List[ScoreCategory] = None
    genre: List[ScoreCategory] = None
    persuasion: List[ScoreCategory] = None
    framing: List[ScoreCategory] = None 


class ArticleScoreData(BaseModel):
    article: ScoreData
    site: ScoreData


class ErrorDetail(BaseModel):
    message: str


class GetTaskResponse(BaseModel):
    status: str
    data: Optional[ArticleScoreData] = None
    error: Optional[ErrorDetail] = None


class SubmitTaskResponse(BaseModel):
    task_id: str
    message: str
