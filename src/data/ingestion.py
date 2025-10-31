"""main data ingestion orchestration."""

import logging
from datetime import datetime, timedelta
from typing import Optional

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        return iterable


from src.config import DataConfig
from src.data.fetchers import GOESXRayFetcher, SolarRegionFetcher, MagnetogramFetcher, load_cache, save_cache
from src.data.persistence import DataPersister
from src.data.database import init_database
from src.data.flare_detector import FlareDetector

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """orchestrates data ingestion from noaa sources."""

    def __init__(self):
        self.xray_fetcher = GOESXRayFetcher()
        self.region_fetcher = SolarRegionFetcher()
        self.magnetogram_fetcher = MagnetogramFetcher()
        self.flare_detector = FlareDetector()
        self.persister = DataPersister()

    def run_incremental_update(self, use_cache: bool = True) -> dict:
        """
        run incremental data update for recent data.

        args:
            use_cache: whether to use cached data if available

        returns:
            dict with ingestion statistics
        """
        results = {
            "xray_flux": None,
            "solar_regions": None,
            "magnetogram": None,
            "flare_events": None,
            "timestamp": datetime.utcnow(),
        }

        # fetch and save xray flux data
        try:
            # try cache first
            cache_name = f"goes_xrs_recent_{datetime.utcnow().strftime('%Y%m%d')}"
            xray_data = None

            if use_cache:
                xray_data = load_cache(cache_name, max_age_hours=DataConfig.CACHE_HOURS)

            if xray_data is None:
                if HAS_TQDM:
                    print("fetching x-ray flux data...", end="\r")
                logger.debug("fetching fresh xray flux data")
                xray_data = self.xray_fetcher.fetch_recent_flux(days=7)

                if xray_data is not None and len(xray_data) > 0:
                    save_cache(xray_data, cache_name)

            if xray_data is not None and len(xray_data) > 0:
                results["xray_flux"] = self.persister.save_xray_flux(xray_data, show_progress=True)

                # detect flare events from flux data
                try:
                    if HAS_TQDM:
                        print("detecting flare events...", end="\r")
                    logger.debug("detecting flare events from x-ray flux data")
                    flares_df = self.flare_detector.detect_flares_from_flux(xray_data, min_class="C")

                    if flares_df is not None and len(flares_df) > 0:
                        results["flare_events"] = self.persister.save_flare_events(flares_df)
                        inserted = results["flare_events"].get("records_inserted", 0)
                        duplicates = results["flare_events"].get("records_updated", 0)
                        if HAS_TQDM:
                            if inserted > 0:
                                print(f"detected {len(flares_df)} flares, saved {inserted} new                    ")
                            else:
                                print(
                                    f"detected {len(flares_df)} flares, all duplicates (already in DB)                    "
                                )
                    else:
                        if HAS_TQDM:
                            print("no flare events detected in current window           ")
                        results["flare_events"] = {"status": "success", "records_inserted": 0, "records_updated": 0}

                except Exception as e:
                    logger.error(f"error detecting flare events: {e}")
                    results["flare_events"] = {"status": "failure", "error": str(e)}
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
                if HAS_TQDM:
                    print("fetching solar region data...", end="\r")
                logger.debug("fetching fresh solar region data")
                region_data = self.region_fetcher.fetch_current_regions()

                if region_data is not None and len(region_data) > 0:
                    save_cache(region_data, region_cache_name)

            if region_data is not None and len(region_data) > 0:
                results["solar_regions"] = self.persister.save_solar_regions(region_data, show_progress=True)

                # extract magnetogram data from regions
                try:
                    if HAS_TQDM:
                        print("extracting magnetogram data...", end="\r")
                    logger.debug("extracting magnetogram data from solar regions")
                    magnetogram_data = self.magnetogram_fetcher.fetch_magnetogram_from_regions(region_data)

                    if magnetogram_data is not None and len(magnetogram_data) > 0:
                        magnetogram_cache_name = f"magnetogram_{datetime.utcnow().strftime('%Y%m%d_%H')}"
                        if use_cache:
                            save_cache(magnetogram_data, magnetogram_cache_name)

                        results["magnetogram"] = self.persister.save_magnetogram(magnetogram_data, show_progress=True)
                    else:
                        logger.warning("no magnetogram data extracted")
                except Exception as e:
                    logger.error(f"error extracting magnetogram data: {e}")
                    results["magnetogram"] = {"status": "failure", "error": str(e)}
            else:
                logger.warning("no solar region data available")

        except Exception as e:
            logger.error(f"error fetching solar regions: {e}")
            results["solar_regions"] = {"status": "failure", "error": str(e)}

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


def setup_logging(level=logging.WARNING):
    """configure logging for data ingestion (default to WARNING for cleaner output)."""
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    """main entry point for data ingestion."""
    setup_logging()

    print("\nflare-plus data ingestion")
    print("=" * 60)

    # initialize database
    logger.debug("setting up database")
    init_database(drop_existing=False)

    # run ingestion pipeline
    pipeline = DataIngestionPipeline()
    results = pipeline.run_incremental_update(use_cache=True)

    # print clean summary
    print("\n" + "=" * 60)
    print("ingestion summary:")
    print("-" * 60)

    if results.get("xray_flux"):
        flux_stats = results["xray_flux"]
        if flux_stats.get("status") == "success":
            print(f"  [OK] x-ray flux:     {flux_stats.get('records_inserted', 0):,} records saved")
        else:
            print(f"  [FAIL] x-ray flux:     failed - {flux_stats.get('error_message', 'unknown error')}")

    if results.get("flare_events"):
        flare_stats = results["flare_events"]
        if flare_stats.get("status") == "success":
            inserted = flare_stats.get("records_inserted", 0)
            duplicates = flare_stats.get("records_updated", 0)
            if inserted > 0:
                print(f"  [OK] flare events:   {inserted} new flares saved")
                if duplicates > 0:
                    print(f"  [INFO] flare events:   {duplicates} duplicates skipped")
            elif duplicates > 0:
                print(f"  [INFO] flare events:   {duplicates} detected (all duplicates, already in DB)")
            else:
                print(f"  [INFO] flare events:   none detected in current window")
        else:
            print(f"  [FAIL] flare events:   failed - {flare_stats.get('error', 'unknown error')}")

    if results.get("solar_regions"):
        region_stats = results["solar_regions"]
        if region_stats.get("status") == "success":
            print(f"  [OK] solar regions:  {region_stats.get('records_inserted', 0):,} records saved")
        else:
            print(f"  [FAIL] solar regions:  failed - {region_stats.get('error_message', 'unknown error')}")

    if results.get("magnetogram"):
        mag_stats = results["magnetogram"]
        if mag_stats.get("status") == "success":
            print(f"  [OK] magnetograms:   {mag_stats.get('records_inserted', 0):,} records saved")
        else:
            print(f"  [FAIL] magnetograms:   failed - {mag_stats.get('error_message', 'unknown error')}")

    print("=" * 60)
    print("ingestion complete!\n")


if __name__ == "__main__":
    main()
