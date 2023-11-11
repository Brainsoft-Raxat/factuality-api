from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.factuality.router import router as factuality_router
from .database import engine
from . import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(factuality_router, prefix="/factuality",
                   tags=["Factuality"])
