from pydantic import PostgresDsn
from pydantic_settings import BaseSettings

# load .env file
from dotenv import load_dotenv
load_dotenv()

class Config(BaseSettings):
    DATABASE_URL: PostgresDsn


settings = Config()
