from typing import Optional
from uuid import UUID

from pydantic import AnyHttpUrl

from src.schemas import CustomModel


class SubmitTaskRequest(CustomModel):
    url: AnyHttpUrl


class SubmitTaskResponse(CustomModel):
    task_id: UUID
    message: str


class ErrorDetail(CustomModel):
    message: str


class GetTaskResponse(CustomModel):
    status: str
    data: Optional[dict] = None
    error: Optional[ErrorDetail] = None
