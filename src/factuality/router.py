import courlan
from fastapi import APIRouter, status, Depends, BackgroundTasks, HTTPException

from .schemas import ScoreRequest, TaskID, GetTaskResponse, SubmitTaskResponse, ErrorDetail
from . import service
from src.dependencies import get_db
from src import models
import src.db.task as db_task

from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/task/submit", status_code=status.HTTP_202_ACCEPTED, responses={
    202: {"description": "Submit task", "model": SubmitTaskResponse}
})
async def submit_task(
        request: ScoreRequest,
        worker: BackgroundTasks,
        db: Session = Depends(get_db)
):
    task = db_task.create_task(db, request.model_dump())

    worker.add_task(service.process_task, db, task.id)

    return SubmitTaskResponse(message="Task submitted", task_id=str(task.id))


@router.get("/task/{task_id}", status_code=status.HTTP_200_OK, responses={
    200: {"description": "Task Status", "model": GetTaskResponse},
    404: {"description": "Task not found", "model": dict},
})
def get_task(task_id: TaskID = Depends(), db: Session = Depends(get_db)):
    task = db_task.get_task(db, str(task_id.task_id))

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status == models.TaskStatus.COMPLETED:
        return GetTaskResponse(status=task.status.name, data=task.response)
    elif task.status == models.TaskStatus.FAILED:
        return GetTaskResponse(status=task.status.name, error=task.error)

    return GetTaskResponse(status=task.status.name)


@router.get("/feed")
def parse_feed(request: ScoreRequest):
    feed = service.parse_feed(request.url)
    if len(feed) == 0:
        feed = service.parse_feed(courlan.get_base_url(request.url))

    return feed
