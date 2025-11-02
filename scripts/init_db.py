#!/usr/bin/env python
"""database initialization script."""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging  # noqa: E402

from src.data.database import init_database  # noqa: E402
from src.data.ingestion import setup_logging  # noqa: E402


def main():
    """initialize database with tables."""
    setup_logging(level=logging.INFO)

    print("initializing flare+ database...")
    print("this will create all required tables in the database")

    # ask for confirmation to drop existing
    drop = input("\ndrop existing tables? (y/N): ").lower().strip() == "y"

    try:
        init_database(drop_existing=drop)
        print("\ndatabase initialized successfully!")
        print("tables created:")
        print("  - flare_goes_xray_flux")
        print("  - flare_solar_regions")
        print("  - flare_solar_magnetogram")
        print("  - flare_events")
        print("  - flare_ingestion_log")

    except Exception as e:
        print(f"\nerror initializing database: {e}")
        print("\nmake sure:")
        print("  1. postgresql is running")
        print("  2. database 'flare_prediction' exists")
        print("  3. credentials in .env are correct")
        sys.exit(1)


if __name__ == "__main__":
    main()
