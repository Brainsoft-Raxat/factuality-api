import uuid
from enum import Enum
from typing import Any, AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    CursorResult,
    DateTime,
    ForeignKey,
    Insert,
    Integer,
    MetaData,
    Select,
    String,
    Table,
    Text,
    Update,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    create_async_engine,
)
from sqlalchemy.sql import func

from src.config import settings
from src.constants import DB_NAMING_CONVENTION
from src.utils import logger

DATABASE_URL = str(settings.DATABASE_ASYNC_URL)

logger.info("Connecting to database")

engine = create_async_engine(
    DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    pool_recycle=settings.DATABASE_POOL_TTL,
    pool_pre_ping=settings.DATABASE_POOL_PRE_PING,
)


metadata = MetaData(naming_convention=DB_NAMING_CONVENTION)


async def get_db_connection() -> AsyncGenerator[AsyncConnection, None]:
    connection = await engine.connect()
    try:
        yield connection
    finally:
        await connection.close()


async def fetch_one(
    select_query: Select | Insert | Update,
    connection: AsyncConnection | None = None,
    commit_after: bool = False,
) -> dict[str, Any] | None:
    if not connection:
        async with engine.connect() as connection:
            cursor = await _execute_query(select_query, connection, commit_after)
            return cursor.first()._asdict() if cursor.rowcount > 0 else None

    cursor = await _execute_query(select_query, connection, commit_after)
    return cursor.first()._asdict() if cursor.rowcount > 0 else None


async def fetch_all(
    select_query: Select | Insert | Update,
    connection: AsyncConnection | None = None,
    commit_after: bool = False,
) -> list[dict[str, Any]]:
    if not connection:
        async with engine.connect() as connection:
            cursor = await _execute_query(select_query, connection, commit_after)
            return [r._asdict() for r in cursor.all()]

    cursor = await _execute_query(select_query, connection, commit_after)
    return [r._asdict() for r in cursor.all()]


async def execute(
    query: Insert | Update,
    connection: AsyncConnection = None,
    commit_after: bool = False,
) -> None:
    if not connection:
        async with engine.connect() as connection:
            await _execute_query(query, connection, commit_after)
            return

    await _execute_query(query, connection, commit_after)


async def _execute_query(
    query: Select | Insert | Update,
    connection: AsyncConnection,
    commit_after: bool = False,
) -> CursorResult:
    result = await connection.execute(query)
    if commit_after:
        await connection.commit()

    return result


class TaskStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


tasks = Table(
    "tasks",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column(
        "updated_at",
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    ),
    Column(
        "status",
        PgEnum(TaskStatus, name="task_status_enum", create_type=True),
        default=TaskStatus.PENDING,
    ),
    Column("request", JSONB, nullable=True),
    Column("response", JSONB, nullable=True),
    Column("error", JSONB, nullable=True),
    Column("retry_count", Integer, server_default=text("0")),
)

sites = Table(
    "sites",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("url", String, unique=True, nullable=False),
    Column("scores", JSONB),
    Column(
        "created_at", DateTime(timezone=True), server_default=func.now(), nullable=False
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    ),
)

articles = Table(
    "articles",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column(
        "site_id",
        UUID(as_uuid=True),
        ForeignKey("sites.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("url", String, nullable=False),
    Column("lang", String, nullable=False),
    Column("title", String, server_default="Unknown", nullable=False),
    Column("author", String, server_default="Unknown", nullable=False),
    Column("library", String, server_default="Unknown"),
    Column("content", Text, nullable=False),
    Column("is_scored", Boolean, server_default="False", nullable=False),
    Column("scores", JSONB),
    Column("num_of_tries", Integer, server_default=text("0")),
    Column(
        "created_at", DateTime(timezone=True), server_default=func.now(), nullable=False
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    ),
)
