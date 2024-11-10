# config.py
from pydantic_settings import BaseSettings
from functools import lru_cache


class Config(BaseSettings):
    HF_API_KEY: str
    FACTUALITY_MODEL_URL: str
    BIAS_MODEL_URL: str
    GENRE_MODEL_URL: str
    PERSUASION_MODEL_URL: str
    FRAMING_MODEL_URL: str
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # This allows extra fields in the environment


@lru_cache()
def get_settings():
    return Config()


settings = get_settings()