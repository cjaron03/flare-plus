"""main data ingestion orchestration."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from src.config import DataConfig
from src.data.fetchers import GOESXRayFetcher, SolarRegionFetcher, load_cache, save_cache
from src.data.persistence import DataPersister
from src.data.database import init_database

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """orchestrates data ingestion from noaa sources."""

    def __init__(self):
        self.xray_fetcher = GOESXRayFetcher()
        self.region_fetcher = SolarRegionFetcher()
        self.persister = DataPersister()

    def run_incremental_update(self, use_cache: bool = True) -> dict:
        """
        run incremental data update for recent data.

        args:
            use_cache: whether to use cached data if available

        returns:
            dict with ingestion statistics
        """
        logger.info("starting incremental data update")
        results = {"xray_flux": None, "solar_regions": None, "timestamp": datetime.utcnow()}

        # fetch and save xray flux data
        try:
            # try cache first
            cache_name = f"goes_xrs_recent_{datetime.utcnow().strftime('%Y%m%d')}"
            xray_data = None

            if use_cache:
                xray_data = load_cache(cache_name, max_age_hours=DataConfig.CACHE_HOURS)

            if xray_data is None:
                logger.info("fetching fresh xray flux data")
                xray_data = self.xray_fetcher.fetch_recent_flux(days=7)

                if xray_data is not None and len(xray_data) > 0:
                    save_cache(xray_data, cache_name)

            if xray_data is not None and len(xray_data) > 0:
                results["xray_flux"] = self.persister.save_xray_flux(xray_data)
            else:
                logger.warning("no xray flux data available")

        except Exception as e:
            logger.error(f"error fetching xray flux: {e}")
            results["xray_flux"] = {"status": "failure", "error": str(e)}

        # fetch and save solar region data
        try:
            region_cache_name = f"solar_regions_{datetime.utcnow().strftime('%Y%m%d_%H')}"
            region_data = None

            if use_cache:
                region_data = load_cache(region_cache_name, max_age_hours=1)

            if region_data is None:
                logger.info("fetching fresh solar region data")
                region_data = self.region_fetcher.fetch_current_regions()

                if region_data is not None and len(region_data) > 0:
                    save_cache(region_data, region_cache_name)

            if region_data is not None and len(region_data) > 0:
                results["solar_regions"] = self.persister.save_solar_regions(region_data)
            else:
                logger.warning("no solar region data available")

        except Exception as e:
            logger.error(f"error fetching solar regions: {e}")
            results["solar_regions"] = {"status": "failure", "error": str(e)}

        logger.info("incremental update complete")
        return results

    def run_historical_backfill(self, start_date: str, end_date: Optional[str] = None):
        """
        backfill historical data.

        args:
            start_date: start date in yyyy-mm-dd format
            end_date: end date in yyyy-mm-dd format (defaults to today)
        """
        logger.info(f"starting historical backfill from {start_date}")
        logger.warning("historical backfill not fully implemented yet")
        logger.info("manual data download may be required from noaa ncei archives")

        # placeholder for historical data ingestion
        # would need to iterate through date ranges and satellites
        # noaa historical data is organized by satellite/year/month

        return {"status": "not_implemented", "message": "historical backfill requires manual setup"}


def setup_logging(level=logging.INFO):
    """configure logging for data ingestion."""
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    """main entry point for data ingestion."""
    setup_logging()

    logger.info("initializing flare+ data ingestion")

    # initialize database
    logger.info("setting up database")
    init_database(drop_existing=False)

    # run ingestion pipeline
    pipeline = DataIngestionPipeline()
    results = pipeline.run_incremental_update(use_cache=True)

    # print results
    logger.info("ingestion results:")
    for source, result in results.items():
        if isinstance(result, dict):
            logger.info(f"  {source}: {result}")

    logger.info("data ingestion complete")


if __name__ == "__main__":
    main()
