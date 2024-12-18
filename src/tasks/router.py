from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, status
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncConnection

import src.tasks.service as service
from src.database import get_db_connection
from src.tasks.constants import TaskStatus
from src.tasks.exceptions import TaskNotFound, TaskSubmissionError
from src.tasks.schemas import GetTaskResponse, SubmitTaskRequest, SubmitTaskResponse

router = APIRouter()


@router.post(
    "/task/submit",
    status_code=status.HTTP_202_ACCEPTED,
    responses={202: {"description": "Submit task", "model": SubmitTaskResponse}},
)
async def submit_task(
    request: SubmitTaskRequest,
    worker: BackgroundTasks,
    db: AsyncConnection = Depends(get_db_connection),
):
    request_dict = jsonable_encoder(request)
    request_dict["status"] = str(TaskStatus.PENDING)
    task = await service.create_task(db, request_dict)

    if not task:
        raise TaskSubmissionError()

    worker.add_task(service.process_task, task["id"])

    return SubmitTaskResponse(task_id=task["id"], message="Task submitted successfully")


@router.get(
    "/task/{task_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Task Status", "model": GetTaskResponse},
        404: {"description": "Task not found", "model": dict},
    },
)
async def get_task(
    task_id: UUID,
    db: AsyncConnection = Depends(get_db_connection),  # Dependency injection
):
    task = await service.get_task(db, task_id)  # Use valid db connection

    if not task:
        raise TaskNotFound()

    return GetTaskResponse(
        status=task["status"], data=task["response"], error=task["error"]
    )
