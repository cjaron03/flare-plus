#!/usr/bin/env python
"""backfill historical data using continuous ingestion.

since noaa swpc only provides last 7 days via api, this script:
1. runs ingestion repeatedly to gather as much current data as possible
2. recommends setting up continuous collection for future historical data

for true multi-year historical data, you would need to:
- contact noaa for bulk data access
- use alternative data sources (sdo/goes satellites)
- or run continuous ingestion for 6-12 months
"""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging  # noqa: E402
from sqlalchemy import func, text  # noqa: E402

from src.data.ingestion import DataIngestionPipeline  # noqa: E402
from src.data.database import init_database, get_database  # noqa: E402
from src.data.schema import FlareEvent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_data_coverage():
    """check current data coverage in database."""
    db = get_database()

    with db.get_session() as session:
        # check flux data
        flux_query = session.execute(text("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM flare_goes_xray_flux"))
        flux_result = flux_query.fetchone()

        # check flare events
        flare_stats = (
            session.query(
                FlareEvent.class_category,
                func.count(FlareEvent.id).label("count"),
                func.min(FlareEvent.start_time).label("first"),
                func.max(FlareEvent.start_time).label("last"),
            )
            .filter(FlareEvent.class_category.in_(["C", "M", "X"]))
            .group_by(FlareEvent.class_category)
            .all()
        )

    return {
        "flux_count": flux_result[0],
        "flux_start": flux_result[1],
        "flux_end": flux_result[2],
        "flares": {cat: {"count": count, "first": first, "last": last} for cat, count, first, last in flare_stats},
    }


def main():
    """main entry point."""
    print("\n" + "=" * 70)
    print("HISTORICAL DATA BACKFILL")
    print("=" * 70)
    print()
    print("This script maximizes collection of available SWPC data.")
    print("SWPC API provides last 7 days - this will ingest all available data.")
    print()
    print("=" * 70 + "\n")

    # initialize database
    logger.info("initializing database...")
    init_database(drop_existing=False)

    # check current coverage
    logger.info("checking current data coverage...")
    coverage_before = check_data_coverage()

    print("Current data coverage:")
    print(f"  Flux records: {coverage_before['flux_count']:,}")
    if coverage_before["flux_start"]:
        print(f"  Date range: {coverage_before['flux_start']} to {coverage_before['flux_end']}")

    print("\nCurrent flare counts:")
    for cat, stats in coverage_before["flares"].items():
        print(f"  {cat}-class: {stats['count']} (from {stats['first']} to {stats['last']})")

    print("\n" + "=" * 70)
    print("Running ingestion to capture latest 7 days...")
    print("=" * 70 + "\n")

    # run ingestion
    pipeline = DataIngestionPipeline()
    pipeline.run_incremental_update(use_cache=False)

    # check coverage after
    logger.info("checking updated data coverage...")
    coverage_after = check_data_coverage()

    print("\n" + "=" * 70)
    print("BACKFILL COMPLETE")
    print("=" * 70)

    added_flux = coverage_after["flux_count"] - coverage_before["flux_count"]
    print(f"\nFlux records: {coverage_after['flux_count']:,} (added {added_flux:,})")
    if coverage_after["flux_start"]:
        total_days = (coverage_after["flux_end"] - coverage_after["flux_start"]).days
        print(f"Coverage: {coverage_after['flux_start']} to {coverage_after['flux_end']} ({total_days} days)")

    print("\nFlare distribution:")
    for cat in ["C", "M", "X"]:
        if cat in coverage_after["flares"]:
            stats = coverage_after["flares"][cat]
            before_count = coverage_before["flares"].get(cat, {}).get("count", 0)
            added = stats["count"] - before_count
            print(f"  {cat}-class: {stats['count']:,} (added {added})")
        else:
            print(f"  {cat}-class: 0")

    # check if we have enough data for training
    c_count = coverage_after["flares"].get("C", {}).get("count", 0)
    m_count = coverage_after["flares"].get("M", {}).get("count", 0)
    x_count = coverage_after["flares"].get("X", {}).get("count", 0)

    print("\n" + "=" * 70)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 70)

    print("\nMinimum recommended for training:")
    print(f"  C-class: 200+ events (you have: {c_count}) {'[OK]' if c_count >= 200 else '[INSUFFICIENT]'}")
    print(f"  M-class: 50+ events (you have: {m_count}) {'[OK]' if m_count >= 50 else '[INSUFFICIENT]'}")
    print(f"  X-class: 10+ events (you have: {x_count}) {'[OK]' if x_count >= 10 else '[INSUFFICIENT]'}")

    if c_count < 200 or m_count < 50:
        print("\n" + "=" * 70)
        print("RECOMMENDATION: INSUFFICIENT DATA FOR RELIABLE TRAINING")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Run continuous ingestion for 3-6 months")
        print("  2. Contact NOAA for bulk historical data access")
        print("  3. Use only current 7-day window for demonstration")
        print("\nTo set up continuous ingestion:")
        print("  - Schedule this script to run hourly")
        print("  - Or use: scripts/run_ingestion_with_retry.py")
        print("  - Run for at least 6 months to gather sufficient data")
    else:
        print("\n[OK] Sufficient data for training!")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
