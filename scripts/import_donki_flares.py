#!/usr/bin/env python
"""import historical flare events from nasa donki api."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DataConfig  # noqa: E402
from src.data.database import get_database  # noqa: E402
from src.data.donki_fetcher import DonkiFetcher  # noqa: E402
from src.data.schema import FlareEvent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_datetime(date_str: str) -> datetime:
    """parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def import_donki_flares(start_date: datetime, end_date: datetime, api_key: str = "DEMO_KEY") -> None:
    """
    import historical flares from donki.

    args:
        start_date: start date for import
        end_date: end date for import
        api_key: nasa api key (default: DEMO_KEY)
    """
    fetcher = DonkiFetcher(api_key=api_key)

    print(f"\n{'=' * 70}")
    print("DONKI HISTORICAL FLARE IMPORT")
    print("=" * 70)
    print(f"Start date: {start_date.date()}")
    print(f"End date: {end_date.date()}")
    print(f"API key: {api_key[:10]}..." if len(api_key) > 10 else f"API key: {api_key}")
    print("=" * 70 + "\n")

    print(f"Fetching flares from {start_date.date()} to {end_date.date()}...")
    flares = fetcher.fetch_date_range(start_date, end_date)

    print(f"Retrieved {len(flares)} flares from DONKI\n")

    if not flares:
        print("No flares found in the specified date range.")
        return

    # convert to flareevent format
    db = get_database()
    inserted = 0
    skipped = 0

    with db.get_session() as session:
        for flare in flares:
            category, magnitude = fetcher.parse_class(flare.get("classType"))

            if not category:
                skipped += 1
                continue

            # parse timestamps (handle 'Z' suffix and None values)
            begin_time_str = (flare.get("beginTime") or "").replace("Z", "+00:00")
            peak_time_str = (flare.get("peakTime") or "").replace("Z", "+00:00")
            end_time_str = (flare.get("endTime") or "").replace("Z", "+00:00")

            try:
                start_time = datetime.fromisoformat(begin_time_str) if begin_time_str else None
                peak_time = datetime.fromisoformat(peak_time_str) if peak_time_str else None
                end_time = datetime.fromisoformat(end_time_str) if end_time_str else None
            except (ValueError, AttributeError) as e:
                logger.warning(f"failed to parse timestamps for flare {flare.get('flrID')}: {e}")
                skipped += 1
                continue

            if not start_time or not peak_time:
                logger.warning(f"missing required timestamps for flare {flare.get('flrID')}")
                skipped += 1
                continue

            # check if already exists (exact match on start_time)
            existing = session.query(FlareEvent).filter(FlareEvent.start_time == start_time).first()

            if existing:
                skipped += 1
                continue

            # build flare class string (e.g., "M2.5")
            flare_class = f"{category}{magnitude:.1f}" if magnitude else category

            # insert new flare
            event = FlareEvent(
                start_time=start_time,
                peak_time=peak_time,
                end_time=end_time,
                flare_class=flare_class,
                class_category=category,
                class_magnitude=magnitude,
                active_region=flare.get("activeRegionNum"),
                location=flare.get("sourceLocation"),
                source="nasa_donki",
                verified=False,
            )

            session.add(event)
            inserted += 1

        session.commit()

    print("Import complete:")
    print(f"  Inserted: {inserted} new flares")
    print(f"  Skipped: {skipped} (duplicates or invalid)")

    # show breakdown by class
    print("\nBreakdown by class:")
    with db.get_session() as session:
        for cls in ["C", "M", "X"]:
            count = (
                session.query(FlareEvent)
                .filter(
                    FlareEvent.class_category == cls,
                    FlareEvent.source == "nasa_donki",
                )
                .count()
            )
            print(f"  {cls}-class: {count}")

    # show database totals comparison
    print("\nDatabase totals:")
    with db.get_session() as session:
        for cls in ["C", "M", "X"]:
            total = session.query(FlareEvent).filter(FlareEvent.class_category == cls).count()
            donki = (
                session.query(FlareEvent)
                .filter(
                    FlareEvent.class_category == cls,
                    FlareEvent.source == "nasa_donki",
                )
                .count()
            )
            print(f"  {cls}-class: {total} total ({donki} from DONKI)")

    print("\n" + "=" * 70)
    print("IMPORT COMPLETE")
    print("=" * 70 + "\n")


def main():
    """main entry point."""
    parser = argparse.ArgumentParser(description="Import historical flares from NASA DONKI")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 2 years ago.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="NASA API key (default: from config.yaml or DEMO_KEY). Get free key at https://api.nasa.gov/",
    )

    args = parser.parse_args()

    # set defaults
    end_date = datetime.now()
    if args.end_date:
        end_date = parse_datetime(args.end_date)

    if args.start_date:
        start_date = parse_datetime(args.start_date)
    else:
        # default to 2 years ago
        start_date = end_date.replace(year=end_date.year - 2)

    # get api key from args, environment variable, or default to DEMO_KEY
    api_key = args.api_key or DataConfig.NASA_API_KEY

    import_donki_flares(start_date, end_date, api_key)


if __name__ == "__main__":
    main()
