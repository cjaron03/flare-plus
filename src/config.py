"""configuration management for flare+ system."""

import logging
import os
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote_plus

import yaml
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# project root directory
PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """
    load configuration from yaml file.

    returns default config if file not found (allows graceful degradation).
    raises ValueError if yaml is invalid.
    """
    config_path = PROJECT_ROOT / "config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                logger.warning("config.yaml exists but is empty, using defaults")
                return _get_default_config()
            return config
    except FileNotFoundError:
        logger.warning(
            f"configuration file not found at {config_path}. "
            "using default configuration. some features may not work as expected. "
            "please create config.yaml in the project root for full functionality."
        )
        return _get_default_config()
    except yaml.YAMLError as e:
        raise ValueError(f"invalid yaml in configuration file {config_path}: {e}")


def _get_default_config() -> Dict[str, Any]:
    """return minimal default configuration."""
    return {
        "data_ingestion": {
            "cache_hours": 24,
            "backfill_start_date": "2024-01-01",
            "update_interval_minutes": 60,
            "endpoints": {},
        },
        "database": {},
        "model": {
            "target_windows": [24, 48],
        },
        "feature_engineering": {
            "rolling_windows": [6, 12, 24],
            "flare_classes": ["B", "C", "M", "X"],
        },
        "survival": {},
    }


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
        user = quote_plus(cls.USER)
        password = quote_plus(cls.PASSWORD)
        return f"postgresql://{user}:{password}@{cls.HOST}:{cls.PORT}/{cls.NAME}"


class DataConfig:
    """data ingestion configuration."""

    CACHE_HOURS = int(os.getenv("DATA_CACHE_HOURS", CONFIG["data_ingestion"]["cache_hours"]))
    BACKFILL_START = os.getenv("BACKFILL_START_DATE", CONFIG["data_ingestion"]["backfill_start_date"])
    UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL_MINUTES", CONFIG["data_ingestion"]["update_interval_minutes"]))

    # endpoints from config
    ENDPOINTS = CONFIG["data_ingestion"]["endpoints"]
