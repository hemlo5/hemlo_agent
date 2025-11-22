from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "dev"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    redis_url: str = "redis://redis:6379/0"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000

    groq_api_key: str | None = None
    groq_llm_model_primary: str = "llama-3.1-70b"
    grok_api_key: str | None = None

    stripe_secret_key: str | None = None

    firebase_project_id: str | None = None

    secret_key: str = "dev-secret-key"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
