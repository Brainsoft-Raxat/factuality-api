from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

from src.config import settings

DATABASE_URL = str(settings.DATABASE_URL)

logging.info(f"Connecting to database at {DATABASE_URL}")
print(f"Connecting to database at {DATABASE_URL}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
