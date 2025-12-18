"""main data ingestion orchestration."""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        return iterable


import pandas as pd

from src.config import DataConfig
from src.data.fetchers import GOESXRayFetcher, SolarRegionFetcher, MagnetogramFetcher, load_cache, save_cache
from src.data.persistence import DataPersister
from src.data.database import init_database
from src.data.flare_detector import FlareDetector
from src.data.donki_fetcher import DonkiFetcher

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """orchestrates data ingestion from noaa sources."""

    def __init__(self):
        self.xray_fetcher = GOESXRayFetcher()
        self.region_fetcher = SolarRegionFetcher()
        self.magnetogram_fetcher = MagnetogramFetcher()
        self.flare_detector = FlareDetector()
        self.persister = DataPersister()
        self.donki_fetcher = self._init_donki_fetcher()

    def _init_donki_fetcher(self) -> Optional[DonkiFetcher]:
        """initialize donki fetcher if nasa api key is configured."""
        api_key = DataConfig.NASA_API_KEY
        if not api_key or api_key == "DEMO_KEY":
            logger.info("NASA_API_KEY not configured, DONKI integration disabled")
            return None
        logger.info("DONKI integration enabled")
        return DonkiFetcher(api_key=api_key)

    def _fetch_donki_flares(self, days: int = 7) -> dict:
        """
        fetch recent flare events from nasa donki.

        args:
            days: number of days to look back (default: 7)

        returns:
            dict with ingestion statistics
        """
        if self.donki_fetcher is None:
            return {"status": "skipped", "reason": "NASA_API_KEY not configured"}

        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            logger.info(f"fetching DONKI flares for past {days} days")
            flares = self.donki_fetcher.fetch_flares(start_date, end_date)

            if not flares:
                logger.info("no DONKI flares found in date range")
                return {"status": "success", "records_inserted": 0, "records_updated": 0}

            # convert to FlareEvent DataFrame format
            flares_df = self._convert_donki_to_flare_events(flares)

            if flares_df is None or len(flares_df) == 0:
                return {"status": "success", "records_inserted": 0, "records_updated": 0}

            # save using existing persister (handles duplicates)
            result = self.persister.save_flare_events(flares_df)
            return result

        except Exception as e:
            logger.error(f"error fetching DONKI flares: {e}")
            return {"status": "failure", "error": str(e)}

    def _convert_donki_to_flare_events(self, donki_flares: List[Dict]) -> Optional[pd.DataFrame]:
        """
        convert donki api response to FlareEvent dataframe format.

        args:
            donki_flares: list of raw donki flare records

        returns:
            dataframe with columns matching save_flare_events() expectations
        """
        rows = []
        for flare in donki_flares:
            class_type = flare.get("classType")
            if not class_type:
                continue

            # parse class (e.g., "M2.5" -> ("M", 2.5))
            category, magnitude = self._parse_flare_class(class_type)
            if not category:
                continue

            # parse timestamps
            start_time = self._parse_donki_timestamp(flare.get("beginTime"))
            peak_time = self._parse_donki_timestamp(flare.get("peakTime"))
            end_time = self._parse_donki_timestamp(flare.get("endTime"))

            if not start_time or not peak_time:
                continue

            rows.append(
                {
                    "start_time": start_time,
                    "peak_time": peak_time,
                    "end_time": end_time,
                    "flare_class": class_type,
                    "class_category": category,
                    "class_magnitude": magnitude,
                    "active_region": flare.get("activeRegionNum"),
                    "source": "nasa_donki",
                    "verified": True,  # official NASA catalog
                }
            )

        if not rows:
            return None

        return pd.DataFrame(rows)

    def _parse_flare_class(self, class_str: str) -> tuple:
        """parse flare class string like 'M2.5' into ('M', 2.5)."""
        if not class_str:
            return (None, None)
        try:
            category = class_str[0].upper()
            magnitude = float(class_str[1:])
            return (category, magnitude)
        except (ValueError, IndexError):
            return (None, None)

    def _parse_donki_timestamp(self, ts_str: str) -> Optional[datetime]:
        """parse donki timestamp string to datetime."""
        if not ts_str:
            return None
        try:
            # handle Z suffix and timezone offsets
            ts_clean = ts_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts_clean)
            # convert to naive UTC
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except (ValueError, AttributeError):
            return None

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
            "donki_flares": None,
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
                logger.info("fetching x-ray flux data...")
                xray_data = self.xray_fetcher.fetch_recent_flux(days=7)

                if xray_data is not None and len(xray_data) > 0:
                    save_cache(xray_data, cache_name)

            if xray_data is not None and len(xray_data) > 0:
                results["xray_flux"] = self.persister.save_xray_flux(xray_data, show_progress=True)

                # detect flare events from flux data
                try:
                    logger.info("detecting flare events from x-ray flux data")
                    flares_df = self.flare_detector.detect_flares_from_flux(xray_data, min_class="C")

                    if flares_df is not None and len(flares_df) > 0:
                        results["flare_events"] = self.persister.save_flare_events(flares_df)
                        inserted = results["flare_events"].get("records_inserted", 0)
                        duplicates = results["flare_events"].get("records_updated", 0)
                        if inserted > 0:
                            logger.info(f"detected {len(flares_df)} flares, saved {inserted} new")
                        else:
                            logger.info(f"detected {len(flares_df)} flares, all duplicates (already in DB)")
                    else:
                        logger.info("no flare events detected in current window")
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
                logger.info("fetching solar region data...")
                region_data = self.region_fetcher.fetch_current_regions()

                if region_data is not None and len(region_data) > 0:
                    save_cache(region_data, region_cache_name)

            if region_data is not None and len(region_data) > 0:
                results["solar_regions"] = self.persister.save_solar_regions(region_data, show_progress=True)

                # extract magnetogram data from regions
                try:
                    magnetogram_cache_name = f"magnetogram_{datetime.utcnow().strftime('%Y%m%d_%H')}"
                    magnetogram_data = None

                    # try cache first
                    if use_cache:
                        magnetogram_data = load_cache(magnetogram_cache_name, max_age_hours=1)

                    if magnetogram_data is None:
                        logger.info("extracting magnetogram data from solar regions")
                        magnetogram_data = self.magnetogram_fetcher.fetch_magnetogram_from_regions(region_data)

                        if magnetogram_data is not None and len(magnetogram_data) > 0:
                            save_cache(magnetogram_data, magnetogram_cache_name)

                    if magnetogram_data is not None and len(magnetogram_data) > 0:
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

        # fetch verified flare events from NASA DONKI
        try:
            results["donki_flares"] = self._fetch_donki_flares(days=7)

            donki_stats = results["donki_flares"]
            if donki_stats.get("status") == "success":
                inserted = donki_stats.get("records_inserted", 0)
                duplicates = donki_stats.get("records_updated", 0)
                if inserted > 0:
                    logger.info(f"saved {inserted} new DONKI verified flares")
                elif duplicates > 0:
                    logger.info(f"DONKI flares: {duplicates} verified (all duplicates)")
                else:
                    logger.info("DONKI flares: none in date range")
            elif donki_stats.get("status") == "skipped":
                logger.debug(f"DONKI fetch skipped: {donki_stats.get('reason', 'unknown')}")

        except Exception as e:
            logger.error(f"error in DONKI ingestion: {e}")
            results["donki_flares"] = {"status": "failure", "error": str(e)}

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

    logger.info("flare-plus data ingestion")
    logger.info("=" * 60)

    # initialize database
    logger.debug("setting up database")
    init_database(drop_existing=False)

    # run ingestion pipeline
    pipeline = DataIngestionPipeline()
    results = pipeline.run_incremental_update(use_cache=True)

    # log clean summary
    logger.info("ingestion summary:")
    logger.info("-" * 60)

    if results.get("xray_flux"):
        flux_stats = results["xray_flux"]
        if flux_stats.get("status") == "success":
            logger.info(f"x-ray flux:     {flux_stats.get('records_inserted', 0):,} records saved")
        else:
            logger.error(f"x-ray flux:     failed - {flux_stats.get('error_message', 'unknown error')}")

    if results.get("flare_events"):
        flare_stats = results["flare_events"]
        if flare_stats.get("status") == "success":
            inserted = flare_stats.get("records_inserted", 0)
            duplicates = flare_stats.get("records_updated", 0)
            if inserted > 0:
                logger.info(f"flare events:   {inserted} new flares saved")
                if duplicates > 0:
                    logger.info(f"flare events:   {duplicates} duplicates skipped")
            elif duplicates > 0:
                logger.info(f"flare events:   {duplicates} detected (all duplicates, already in DB)")
            else:
                logger.info("flare events:   none detected in current window")
        else:
            logger.error(f"flare events:   failed - {flare_stats.get('error', 'unknown error')}")

    if results.get("solar_regions"):
        region_stats = results["solar_regions"]
        if region_stats.get("status") == "success":
            logger.info(f"solar regions:  {region_stats.get('records_inserted', 0):,} records saved")
        else:
            logger.error(f"solar regions:  failed - {region_stats.get('error_message', 'unknown error')}")

    if results.get("magnetogram"):
        mag_stats = results["magnetogram"]
        if mag_stats.get("status") == "success":
            logger.info(f"magnetograms:   {mag_stats.get('records_inserted', 0):,} records saved")
        else:
            logger.error(f"magnetograms:   failed - {mag_stats.get('error_message', 'unknown error')}")

    if results.get("donki_flares"):
        donki_stats = results["donki_flares"]
        if donki_stats.get("status") == "success":
            inserted = donki_stats.get("records_inserted", 0)
            duplicates = donki_stats.get("records_updated", 0)
            if inserted > 0:
                logger.info(f"donki flares:   {inserted} new verified flares saved")
            elif duplicates > 0:
                logger.info(f"donki flares:   {duplicates} verified (all duplicates)")
            else:
                logger.info("donki flares:   none in date range")
        elif donki_stats.get("status") == "skipped":
            logger.info(f"donki flares:   skipped - {donki_stats.get('reason', 'disabled')}")
        else:
            logger.error(f"donki flares:   failed - {donki_stats.get('error', 'unknown error')}")

    logger.info("=" * 60)
    logger.info("ingestion complete")


if __name__ == "__main__":
    main()
