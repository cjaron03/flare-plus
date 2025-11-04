#!/usr/bin/env python
"""validate environment configuration and dependencies."""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging  # noqa: E402
import os  # noqa: E402
from typing import List, Tuple  # noqa: E402

from sqlalchemy import text  # noqa: E402

from src.config import load_config, DatabaseConfig, PROJECT_ROOT  # noqa: E402
from src.data.database import get_database  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_env_file() -> Tuple[bool, List[str]]:
    """
    check if .env file exists and has required variables.

    returns:
        tuple of (success, error_messages)
    """
    print("Checking .env file...")
    errors = []

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        errors.append(".env file not found in project root")
        return False, errors

    # check required environment variables
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        errors.append(f"missing environment variables: {', '.join(missing_vars)}")
        return False, errors

    print("[OK] .env file exists with all required variables")
    return True, []


def check_config_yaml() -> Tuple[bool, List[str]]:
    """
    check if config.yaml is valid.

    returns:
        tuple of (success, error_messages)
    """
    print("\nChecking config.yaml...")
    errors = []

    try:
        config = load_config()

        # check required sections
        required_sections = ["data_ingestion", "model_training"]
        for section in required_sections:
            if section not in config:
                errors.append(f"missing config section: {section}")

        if errors:
            return False, errors

        print("[OK] config.yaml is valid")
        return True, []

    except Exception as e:
        errors.append(f"failed to load config.yaml: {e}")
        return False, errors


def check_database_connection() -> Tuple[bool, List[str]]:
    """
    test database connection.

    returns:
        tuple of (success, error_messages)
    """
    print("\nChecking database connection...")
    errors = []

    try:
        db = get_database()

        # try a simple query
        with db.get_session() as session:
            result = session.execute(text("SELECT 1"))
            result.fetchone()

        print(f"[OK] database connection successful ({DatabaseConfig.get_connection_string()})")
        return True, []

    except Exception as e:
        errors.append(f"database connection failed: {e}")
        return False, errors


def check_required_directories() -> Tuple[bool, List[str]]:
    """
    check if required directories exist.

    returns:
        tuple of (success, error_messages)
    """
    print("\nChecking required directories...")
    errors = []

    required_dirs = [
        ("models", "models/"),
        ("data cache", "data/cache/"),
        ("source code", "src/"),
        ("scripts", "scripts/"),
    ]

    for name, path in required_dirs:
        full_path = PROJECT_ROOT / path
        if not full_path.exists():
            errors.append(f"{name} directory not found: {full_path}")
        else:
            print(f"[OK] {name} directory exists: {full_path}")

    return len(errors) == 0, errors


def check_disk_space() -> Tuple[bool, List[str]]:
    """
    check disk space in critical directories.

    returns:
        tuple of (success, error_messages)
    """
    print("\nChecking disk space...")
    errors = []
    warnings = []

    try:
        import shutil

        # check space in project root
        stats = shutil.disk_usage(PROJECT_ROOT)
        free_gb = stats.free / (1024**3)
        total_gb = stats.total / (1024**3)
        used_pct = (stats.used / stats.total) * 100

        print(f"Disk space: {free_gb:.2f} GB free / {total_gb:.2f} GB total ({used_pct:.1f}% used)")

        # warn if less than 1 GB free
        if free_gb < 1.0:
            errors.append(f"low disk space: only {free_gb:.2f} GB free")
        elif free_gb < 5.0:
            warnings.append(f"disk space getting low: {free_gb:.2f} GB free")

        # check models directory size if it exists
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            total_size = sum(f.stat().st_size for f in models_dir.rglob("*") if f.is_file())
            size_mb = total_size / (1024**2)
            print(f"Models directory size: {size_mb:.2f} MB")

        # check data cache directory size if it exists
        cache_dir = PROJECT_ROOT / "data" / "cache"
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            size_mb = total_size / (1024**2)
            print(f"Cache directory size: {size_mb:.2f} MB")

        if warnings:
            for warning in warnings:
                print(f"[WARNING] {warning}")

        return len(errors) == 0, errors

    except Exception as e:
        errors.append(f"disk space check failed: {e}")
        return False, errors


def main():
    """main entry point."""
    print("=" * 70)
    print("FLARE+ CONFIGURATION VALIDATOR")
    print("=" * 70)
    print()

    all_checks = [
        ("Environment File", check_env_file),
        ("Config YAML", check_config_yaml),
        ("Database Connection", check_database_connection),
        ("Required Directories", check_required_directories),
        ("Disk Space", check_disk_space),
    ]

    results = []
    all_errors = []

    for check_name, check_func in all_checks:
        try:
            success, errors = check_func()
            results.append((check_name, success))
            if errors:
                all_errors.extend(errors)
        except Exception as e:
            logger.error(f"{check_name} check failed with exception: {e}", exc_info=True)
            results.append((check_name, False))
            all_errors.append(f"{check_name}: {e}")

    # print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for check_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {check_name}")

    if all_errors:
        print("\nErrors found:")
        for error in all_errors:
            print(f"  - {error}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\n{passed}/{total} checks passed")

    if passed == total:
        print("\n[OK] All configuration checks passed")
        sys.exit(0)
    else:
        print("\n[FAIL] Configuration validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
