import uuid
import src.models as models
from sqlalchemy.orm import Session


def create_task(db: Session, request: dict) -> models.Task:
    task = models.Task(
        id=uuid.uuid4(),
        status=models.TaskStatus.PENDING,
        request=request,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def update_task(db: Session, task: models.Task) -> models.Task:
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_task(db: Session, task_id: uuid.UUID) -> models.Task:
    return db.query(models.Task).filter(models.Task.id == task_id).first()
