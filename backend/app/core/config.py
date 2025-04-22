from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # APP SETTINGs
    APP_NAME: str = "admin"
    APP_PORT: str = "8000"
    WORKER: int = 1
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "prod"
    DATA_PATH: str = "data"

    # LOG SETTINGs
    LOG_LEVEL: str = "INFO"

    # # REDIS SETTINGs
    # REDIS_HOST: str = "127.0.0.1"
    # REDIS_PORT: int = 6379
    # REDIS_DATABASE: int = 10
    # REDIS_ACCESS_KEY: str = ""
    # REDIS_USE_SSL: bool = False

    # DB SETTINGS (기본값을 .env에서 읽어오도록 설정)
    DB_ENGINE: str = Field(default="mysql", env="DB_ENGINE")
    DB_USER: str = Field(default="", env="DB_USER")
    DB_PASSWORD: str = Field(default="", env="DB_PASSWORD")
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: int = Field(default=3306, env="DB_PORT")
    DB_NAME: str = Field(default="", env="DB_NAME")
    SQLALCHEMY_DATABASE_URL: Optional[str] = None
    DB_POOL_SIZE: int = 50
    DB_POOL_MAX_OVERFLOW: int = 30
    DB_POOL_RECYCLE: int = 14400
    MILVUS_URI: str = Field(default="http://localhost:19530", env="MILVUS_URI")
    # VECTOR_DB_PORT: int = Field(default=19530, env="VECTOR_DB_PORT")
    USER_ID: str = Field(default="skcc", env="USER_ID")
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")

    @validator("SQLALCHEMY_DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        preset = {
            "mysql": "mysql+mysqlconnector://{username}:{password}@{host}:{port}/{dbname}",
        }
        return preset[values.get("DB_ENGINE")].format(
            username=values.get("DB_USER"),
            password=values.get("DB_PASSWORD"),
            host=values.get("DB_HOST"),
            port=values.get("DB_PORT"),
            dbname=values.get("DB_NAME"),
        )


@lru_cache()
def get_setting():
    return Settings()
