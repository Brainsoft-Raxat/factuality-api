from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    Integer,
    DateTime,
    ForeignKey,
    text
)
from enum import Enum
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM as PgEnum
from sqlalchemy.orm import relationship

from .database import Base


class TaskStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Task(Base):
    __tablename__ = 'tasks'

    id = Column(UUID(as_uuid=True), primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(), onupdate=func.now())
    
    status = Column(
        PgEnum(TaskStatus, name="task_status_enum", create_type=True),
        default=TaskStatus.PENDING
    )
    
    request = Column(JSONB, nullable=True)
    response = Column(JSONB, nullable=True)
    error = Column(JSONB, nullable=True)
    retry_count = Column(Integer, server_default=text('0'))


class Site(Base):
    __tablename__ = 'sites'

    id = Column(UUID(as_uuid=True), primary_key=True)
    url = Column(String, unique=True, nullable=False)
    scores = Column(JSONB)
    created_at = Column(DateTime(timezone=True),
                        server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(), onupdate=func.now())


class Article(Base):
    __tablename__ = 'articles'

    id = Column(UUID(as_uuid=True), primary_key=True)
    site_id = Column(UUID(as_uuid=True), ForeignKey(
        'sites.id', ondelete="CASCADE"), nullable=False)
    url = Column(String, nullable=False)
    lang = Column(String, nullable=False)
    title = Column(String, server_default="Unknown", nullable=False)
    author = Column(String, server_default="Unknown", nullable=False)
    library = Column(String, server_default="Unknown")
    content = Column(Text, nullable=False)
    is_scored = Column(Boolean, server_default="False", nullable=False)
    scores = Column(JSONB)
    num_of_tries = Column(Integer, server_default=text('0'))
    created_at = Column(DateTime(timezone=True),
                        server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(), onupdate=func.now())

    site = relationship("Site", backref="articles")
