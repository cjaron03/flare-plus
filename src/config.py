"""configuration management for flare+ system."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def load_config() -> Dict[str, Any]:
    """load configuration from yaml file."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# configuration object
CONFIG = load_config()


class DatabaseConfig:
    """database connection configuration."""

    HOST = os.getenv("DB_HOST", "localhost")
    PORT = int(os.getenv("DB_PORT", "5432"))
    NAME = os.getenv("DB_NAME", "flare_prediction")
    USER = os.getenv("DB_USER", "postgres")
    PASSWORD = os.getenv("DB_PASSWORD", "")

    @classmethod
    def get_connection_string(cls) -> str:
        """get sqlalchemy connection string."""
        return f"postgresql://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.NAME}"


class DataConfig:
    """data ingestion configuration."""

    CACHE_HOURS = int(os.getenv("DATA_CACHE_HOURS", CONFIG["data_ingestion"]["cache_hours"]))
    BACKFILL_START = os.getenv("BACKFILL_START_DATE", CONFIG["data_ingestion"]["backfill_start_date"])
    UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL_MINUTES", CONFIG["data_ingestion"]["update_interval_minutes"]))

    # endpoints from config
    ENDPOINTS = CONFIG["data_ingestion"]["endpoints"]
