"""configuration management for flare+ system."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple
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


class AdminConfig:
    """admin access configuration for dashboard authentication."""

    ACCESS_TOKEN = os.getenv("ADMIN_ACCESS_TOKEN")
    RUNTIME_TOKEN = os.getenv("ADMIN_RUNTIME_TOKEN")
    STATUS_MESSAGE = os.getenv("ADMIN_STATUS_MESSAGE", "Admin access enabled.")
    DEV_USERNAME = os.getenv("ADMIN_UI_USERNAME", "plncake")
    DEV_PASSWORD = os.getenv("ADMIN_UI_PASSWORD", "12345")
    LOGIN_ENABLED = os.getenv("ADMIN_UI_LOGIN_ENABLED", "true").lower() not in {"0", "false", "no"}
    MAX_LOGIN_ATTEMPTS = int(os.getenv("ADMIN_UI_MAX_ATTEMPTS", "5"))
    LOGIN_WINDOW_SECONDS = int(os.getenv("ADMIN_UI_ATTEMPT_WINDOW", "60"))
    LOGIN_LOCKOUT_SECONDS = int(os.getenv("ADMIN_UI_LOCKOUT_SECONDS", "30"))
    _SESSION_MESSAGE = "Admin access enabled (UI session)"
    _session_granted = False
    _failed_attempts = []
    _locked_until = 0.0

    @classmethod
    def has_access(cls) -> bool:
        """return true if admin access is currently granted for this session."""
        if cls._session_granted:
            return True
        return cls._env_tokens_valid()

    @classmethod
    def _env_tokens_valid(cls) -> bool:
        """check if environment tokens are present and match."""
        if not cls.ACCESS_TOKEN:
            return False
        if not cls.RUNTIME_TOKEN:
            return False
        return cls.ACCESS_TOKEN == cls.RUNTIME_TOKEN

    @classmethod
    def disabled_reason(cls) -> str:
        """provide reason admin tools are disabled."""
        if cls.has_access():
            return ""
        if not cls.ACCESS_TOKEN:
            return "Admin tools locked. Use the Login tab with valid credentials to unlock admin features."
        if not cls.RUNTIME_TOKEN:
            return "Admin tools locked. Launch the dashboard with ADMIN_RUNTIME_TOKEN matching ADMIN_ACCESS_TOKEN."
        if cls.RUNTIME_TOKEN != cls.ACCESS_TOKEN:
            return "Invalid admin runtime token. Restart the dashboard with a matching ADMIN_RUNTIME_TOKEN."
        return ""

    @classmethod
    def status_indicator(cls) -> str:
        """return human-friendly status indicator text."""
        if cls._session_granted:
            return cls._SESSION_MESSAGE
        return cls.STATUS_MESSAGE if cls.has_access() else "Admin access disabled."

    @classmethod
    def validate_credentials(cls, username: str, password: str) -> Tuple[bool, str]:
        """validate UI login credentials and grant session access on success."""
        allowed, message = cls._login_allowed()
        if not allowed:
            return False, message

        if username == cls.DEV_USERNAME and password == cls.DEV_PASSWORD:
            cls.grant_session_access()
            cls._failed_attempts = []
            cls._locked_until = 0.0
            return True, "Login successful."

        cls._record_failed_attempt()
        return False, "Invalid credentials. Please try again."

    @classmethod
    def grant_session_access(cls):
        """enable admin access for current session."""
        cls._session_granted = True

    @classmethod
    def revoke_session_access(cls):
        """disable ui-granted admin access."""
        cls._session_granted = False
        cls._failed_attempts = []

    @classmethod
    def _login_allowed(cls) -> Tuple[bool, str]:
        if not cls.LOGIN_ENABLED:
            return False, "UI login disabled. Set ADMIN_UI_LOGIN_ENABLED=true to enable."

        now = time.time()
        if now < cls._locked_until:
            remaining = int(cls._locked_until - now)
            return False, f"Too many failed attempts. Try again in {remaining}s."

        return True, ""

    @classmethod
    def _record_failed_attempt(cls):
        now = time.time()
        cls._failed_attempts = [ts for ts in cls._failed_attempts if now - ts < cls.LOGIN_WINDOW_SECONDS]
        cls._failed_attempts.append(now)

        if len(cls._failed_attempts) >= max(1, cls.MAX_LOGIN_ATTEMPTS):
            cls._locked_until = now + cls.LOGIN_LOCKOUT_SECONDS
            cls._failed_attempts = []
