import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "NewsSentinel"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    STAGE: int = 1
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    SECRET_KEY: str = "newssentinel-dev-secret-key-stage1"

    # Directory Paths
    BASE_DIR: Path = BASE_DIR
    APP_DIR: Path = BASE_DIR / "app"
    TEMPLATES_DIR: Path = BASE_DIR / "app" / "templates"
    STATIC_DIR: Path = BASE_DIR / "app" / "static"

    # Stage 2 ML Pipeline Stubs & Config
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"

    # Phase 5 Workstream 3: Uncertainty Detection Threshold
    UNCERTAINTY_THRESHOLD: float = 0.65

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
