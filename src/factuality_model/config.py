from pydantic_settings import BaseSettings


class Config(BaseSettings):
    BASE_MODEL_URL: str
    BASE_MODEL_TOKEN: str
    FACTUALITY_MODEL_PATH: str
    FREEDOM_MODEL_PATH: str
    BIAS_MODEL_PATH: str
    GENRE_MODEL_PATH: str


settings = Config()
