#!/usr/bin/env python
"""download historical goes x-ray data from noaa ncei archives (2020-2024)."""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from typing import Optional  # noqa: E402

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.database import init_database  # noqa: E402
from src.data.flare_detector import FlareDetector  # noqa: E402
from src.data.persistence import DataPersister  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class HistoricalGOESDownloader:
    """downloads historical goes x-ray data from noaa ncei archives."""

    def __init__(self):
        self.session = requests.Session()
        self.persister = DataPersister()
        self.flare_detector = FlareDetector()

        # NOAA NCEI GOES XRS primary data endpoint (JSON format)
        # This provides daily averaged data, but we need higher resolution
        # We'll use the SWPC archive which has 1-minute data in CSV format
        self.base_url = "https://services.swpc.noaa.gov/json/goes/primary"

    def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        batch_days: int = 7,
    ) -> dict:
        """
        download goes xrs data for a date range.

        downloads in batches to avoid overwhelming the API.

        args:
            start_date: start date
            end_date: end date
            batch_days: days per batch

        returns:
            dict with download statistics
        """
        logger.info(f"downloading historical goes data from {start_date.date()} to {end_date.date()}")

        total_flux_records = 0
        total_flares_detected = 0
        total_days = (end_date - start_date).days

        # process in batches
        current_date = start_date

        with tqdm(total=total_days, desc="Downloading historical data", unit="day") as pbar:
            while current_date < end_date:
                batch_end = min(current_date + timedelta(days=batch_days), end_date)

                try:
                    # download batch
                    flux_data = self._download_batch(current_date, batch_end)

                    if flux_data is not None and len(flux_data) > 0:
                        # save flux data
                        result = self.persister.save_xray_flux(flux_data, show_progress=False)
                        if result.get("status") == "success":
                            total_flux_records += result.get("records_inserted", 0)

                        # detect and save flares
                        flares_df = self.flare_detector.detect_flares_from_flux(flux_data, min_class="C")
                        if flares_df is not None and len(flares_df) > 0:
                            flare_result = self.persister.save_flare_events(flares_df)
                            if flare_result.get("status") == "success":
                                total_flares_detected += flare_result.get("records_inserted", 0)

                    # small delay to be nice to NOAA servers
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"error downloading batch {current_date.date()} to {batch_end.date()}: {e}")

                days_processed = (batch_end - current_date).days
                pbar.update(days_processed)
                current_date = batch_end

        logger.info(f"download complete: {total_flux_records} flux records, {total_flares_detected} flares detected")

        return {
            "flux_records": total_flux_records,
            "flares_detected": total_flares_detected,
            "start_date": start_date,
            "end_date": end_date,
        }

    def _download_batch(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        download a batch of data using swpc archive.

        swpc provides recent data via json api (last 7 days).
        for historical data, we need to construct the data differently.

        note: noaa swpc json api only provides last 7 days.
        for true historical data (2020-2024), we would need to:
        1. use ncei netcdf files (requires netcdf4 library)
        2. or scrape swpc daily reports
        3. or use alternative data sources

        for now, this will synthesize data from the current 7-day endpoint
        to demonstrate the pipeline. for production, implement ncei download.
        """
        try:
            # for demonstration: use the 7-day endpoint
            # in production: download from ncei archives
            url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

            logger.debug(f"fetching batch: {start_date.date()} to {end_date.date()}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                return None

            # parse into dataframe
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["time_tag"])

            # filter to date range (though this only has last 7 days)
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]

            # separate short and long wavelength bands
            df_short = df[df["energy"] == "0.05-0.4nm"].copy()
            df_long = df[df["energy"] == "0.1-0.8nm"].copy()

            # merge
            result = pd.merge(
                df_short[["timestamp", "flux", "satellite"]].rename(columns={"flux": "flux_short"}),
                df_long[["timestamp", "flux"]].rename(columns={"flux": "flux_long"}),
                on="timestamp",
                how="outer",
            )

            # convert to numeric
            result["flux_short"] = pd.to_numeric(result.get("flux_short"), errors="coerce")
            result["flux_long"] = pd.to_numeric(result.get("flux_long"), errors="coerce")

            # filter out negative/invalid values
            result = result[(result["flux_short"] > 0) & (result["flux_long"] > 0)].copy()

            result = result.sort_values("timestamp").reset_index(drop=True)

            logger.debug(f"fetched {len(result)} records for batch")
            return result

        except Exception as e:
            logger.error(f"failed to download batch: {e}")
            return None


def main():
    """main entry point."""
    parser = argparse.ArgumentParser(description="Download historical GOES X-ray data")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--batch-days",
        type=int,
        default=7,
        help="Number of days per batch",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database before download",
    )

    args = parser.parse_args()

    # parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    print("\n" + "=" * 70)
    print("HISTORICAL GOES DATA DOWNLOAD")
    print("=" * 70)
    print(f"Start date: {start_date.date()}")
    print(f"End date: {end_date.date()}")
    print(f"Total days: {(end_date - start_date).days}")
    print("=" * 70 + "\n")

    # warning about current limitation
    print("WARNING: This script currently uses SWPC 7-day API endpoint")
    print("         which only provides last 7 days of data.")
    print("         For true 2020-2024 historical data, implement NCEI download.")
    print("         See: https://www.ncei.noaa.gov/data/goes-space-environment-monitor/\n")

    # initialize database if requested
    if args.init_db:
        logger.info("initializing database...")
        init_database(drop_existing=False)

    # download data
    downloader = HistoricalGOESDownloader()
    stats = downloader.download_date_range(
        start_date=start_date,
        end_date=end_date,
        batch_days=args.batch_days,
    )

    # print summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Flux records saved: {stats['flux_records']:,}")
    print(f"Flares detected: {stats['flares_detected']:,}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
