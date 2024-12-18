from uuid import UUID

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

import src.score.service as score_service
import src.tasks.service as tasks_service
from src.database import engine, fetch_one, tasks
from src.tasks.constants import TaskStatus
from src.tasks.exceptions import TaskNotFound
from src.utils import logger


async def create_task(db: AsyncConnection, request: dict) -> dict:
    insert_query = (
        insert(tasks)
        .values(
            status="PENDING",
            request=request,
        )
        .returning(
            tasks.c.id,
            tasks.c.created_at,
            tasks.c.updated_at,
            tasks.c.status,
            tasks.c.request,
        )
    )

    created_task = await fetch_one(insert_query, db, True)

    if not created_task:
        logger.error("Failed to insert task into the database")
    else:
        logger.info(f"Task created: {created_task}")
    return created_task


async def get_task(db: AsyncConnection, task_id: UUID) -> dict:
    select_query = select(
        tasks.c.id,
        tasks.c.created_at,
        tasks.c.updated_at,
        tasks.c.status,
        tasks.c.request,
        tasks.c.response,
        tasks.c.error,
        tasks.c.retry_count,
    ).where(tasks.c.id == task_id)

    return await fetch_one(select_query, db, True)


async def update_task(db: AsyncConnection, task: dict) -> dict:
    update_query = (
        update(tasks)
        .where(tasks.c.id == task["id"])
        .values(
            status=task.get("status"),
            response=task.get("response"),
            error=task.get("error"),
            retry_count=task.get("retry_count"),
        )
        .returning(
            tasks.c.id,
            tasks.c.created_at,
            tasks.c.updated_at,
            tasks.c.status,
            tasks.c.request,
            tasks.c.response,
            tasks.c.error,
            tasks.c.retry_count,
        )
    )

    updated_task = await fetch_one(update_query, db, True)

    if not updated_task:
        logger.error(f"Failed to update task {task['id']} in the database")
    else:
        logger.info(f"Task updated: {updated_task}")

    return updated_task


async def process_task(task_id: UUID):
    logger.info(f"Starting task: {task_id}")
    try:
        async with engine.connect() as db:
            task = await tasks_service.get_task(db, task_id)
            if not task:
                logger.error(f"Task {task_id} not found!")
                raise TaskNotFound()

            task["status"] = TaskStatus.IN_PROGRESS.value
            task = await tasks_service.update_task(
                db,
                task,
            )

            logger.info(f"Processing task: {task_id}")
            scores = await score_service.score(db, task["request"]["url"])

            task["status"] = TaskStatus.COMPLETED.value
            task["response"] = scores
            task = await tasks_service.update_task(
                db,
                task,
            )

            logger.info(f"Finished task: {task_id}")
    except Exception as e:
        logger.exception(f"Task {task_id} failed: {e}")
        async with engine.connect() as db:
            task["status"] = TaskStatus.FAILED.value
            task["error"] = {"message": f"{e}"}
            task = await tasks_service.update_task(db, task)
