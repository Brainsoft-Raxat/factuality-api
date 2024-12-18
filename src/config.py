from typing import Any

from dotenv import load_dotenv
from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.constants import Environment

load_dotenv()


class CustomBaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class Config(BaseSettings):
    DATABASE_URL: PostgresDsn
    DATABASE_ASYNC_URL: PostgresDsn
    DATABASE_POOL_SIZE: int = 16
    DATABASE_POOL_TTL: int = 60 * 20  # 20 minutes
    DATABASE_POOL_PRE_PING: bool = True

    ENVIRONMENT: Environment = Environment.PRODUCTION

    APP_VERSION: str = "0.1"


settings = Config()

app_configs: dict[str, Any] = {"title": "Factuality API"}
# if settings.ENVIRONMENT.is_deployed:
#     app_configs["root_path"] = f"/v{settings.APP_VERSION}"

if not settings.ENVIRONMENT.is_debug:
    app_configs["openapi_url"] = None
