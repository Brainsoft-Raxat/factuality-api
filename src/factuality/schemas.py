from uuid import UUID
from typing import Optional, Dict

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
    article: Dict[str, Dict[str, float]]  # Adjusted to nested dict for detailed scores
    site: Dict[str, Dict[str, float]]     # Adjusted to nested dict for detailed scores

class ArticleCreate(BaseModel):
    site_id: UUID
    url: str
    lang: Optional[str] = None  # Assuming language might be relevant
    title: str
    author: Optional[str] = None  # Made optional, as not all articles might have a clear author
    content: str
    is_scored: bool
    scores: Optional[Dict[str, Dict[str, float]]] = None  # Adjusted for detailed scoring
    num_of_tries: Optional[int] = None

    @validator("url", allow_reuse=True)
    def validate_url(cls, v: str) -> str:
        ok, obj = validate_url(v)
        if not ok:
            raise ValueError("Invalid URL")
        return scrub_url(v)

# Adjusting Score and ScoreData to fit the detailed structure.
class ScoreCategory(BaseModel):
    LOW: float = 0.0
    MIXED: float = 0.0
    HIGH: float = 0.0

class FreedomScoreCategory(BaseModel):
    MOSTLY_FREE: float = 0.0
    EXCELLENT: float = 0.0
    LIMITED_FREEDOM: float = 0.0
    TOTAL_OPPRESSION: float = 0.0
    MODERATE_FREEDOM: float = 0.0

class BiasScoreCategory(BaseModel):
    LEAST_BIASED: float = 0.0
    FAR_RIGHT: float = 0.0
    RIGHT: float = 0.0
    RIGHT_CENTER: float = 0.0
    LEFT: float = 0.0
    LEFT_CENTER: float = 0.0
    FAR_LEFT: float = 0.0

class ScoreData(BaseModel):
    factuality: ScoreCategory
    freedom: FreedomScoreCategory
    bias: BiasScoreCategory

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
