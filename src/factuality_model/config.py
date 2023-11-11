from pydantic_settings import BaseSettings


class Config(BaseSettings):
    FACTUALITY_MODEL_URL: str
    FACTUALITY_MODEL_TOKEN: str


settings = Config()
