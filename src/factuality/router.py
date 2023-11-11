import courlan
from fastapi import APIRouter, status, Depends, BackgroundTasks, HTTPException

from .schemas import ScoreRequest
from . import service
from src.dependencies import get_db
from src import models

from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/task/submit", status_code=status.HTTP_202_ACCEPTED)
async def submit_task(
        request: ScoreRequest,
        worker: BackgroundTasks,
        db: Session = Depends(get_db)
):
    db_task = service.create_task(db, request.model_dump())

    worker.add_task(service.process_task, db, db_task.id)

    return {"message": "Task submitted", "task_id": db_task.id}


@router.get("/task/{task_id}", status_code=status.HTTP_200_OK)
def get_task(task_id: str, db: Session = Depends(get_db)):
    db_task = service.get_task(db, task_id)

    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if db_task.status == models.TaskStatus.COMPLETED:
        return {"status": db_task.status, "data": db_task.response}
    elif db_task.status == models.TaskStatus.FAILED:
        return {"status": db_task.status, "error": db_task.error}

    return {"status": db_task.status}

@router.get("/feed")
def parse_feed(request: ScoreRequest):
    feed = service.parse_feed(request.url)
    if len(feed) == 0:
        feed = service.parse_feed(courlan.get_base_url(request.url))

    return feed
