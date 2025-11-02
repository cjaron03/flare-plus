#!/usr/bin/env python
"""download historical goes x-ray data from noaa ncei archives (2020-2024).

ncei provides goes xrs data in csv format at:
https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/

data is organized by: satellite / year / month / day
"""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse  # noqa: E402
import io  # noqa: E402
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


class NCEIHistoricalDownloader:
    """downloads historical goes data from ncei archives."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "flare-prediction-research/1.0"})
        self.persister = DataPersister()
        self.flare_detector = FlareDetector()

        # NCEI base URL - science quality xrs data
        # format: satellite/level/product/year/month/day/
        self.base_url = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs"

        # satellite mapping for different time periods
        # 2020-2024: primarily GOES-16 (GOES-R series)
        # 2020: GOES-16, GOES-17
        # 2021-2024: GOES-16, GOES-17, GOES-18 (launched 2022)
        self.satellites = {
            (datetime(2020, 1, 1), datetime(2022, 6, 30)): ["goes16", "goes17"],
            (datetime(2022, 7, 1), datetime(2024, 12, 31)): ["goes16", "goes18"],
        }

    def get_satellite_for_date(self, date: datetime) -> str:
        """determine which satellite to use for a given date."""
        for (start, end), sats in self.satellites.items():
            if start <= date <= end:
                # prefer primary satellite (first in list)
                return sats[0]
        # default to goes16
        return "goes16"

    def download_day(self, date: datetime, satellite: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        download goes xrs data for a single day.

        args:
            date: date to download
            satellite: satellite name (e.g., 'goes16'), if none will auto-select

        returns:
            dataframe with flux data or none if failed
        """
        if satellite is None:
            satellite = self.get_satellite_for_date(date)

        # construct url: base/satellite/year/month/sci_xrsf-l2-flx1s_gXX_dYYYYMMDD_v#-#-#.nc
        # actually, ncei provides csv files too
        # url pattern: /goes16/2020/01/sci_xrsf-l2-avg1m_g16_d20200101_v2-2-0.csv
        year = date.strftime("%Y")
        month = date.strftime("%m")
        date_str = date.strftime("%Y%m%d")

        # try different filename patterns and versions
        # ncei uses semantic versioning, try recent versions first
        versions = ["v2-2-0", "v2-1-0", "v2-0-0"]

        for version in versions:
            # 1-minute averaged data
            filename = f"sci_xrsf-l2-avg1m_{satellite[0:3]}{satellite[-2:]}_{date_str}_{version}.csv"
            url = f"{self.base_url}/{satellite}/{year}/{month}/{filename}"

            try:
                logger.debug(f"trying {url}")
                response = self.session.get(url, timeout=30)

                if response.status_code == 200:
                    return self._parse_ncei_csv(response.text, satellite, date)
                elif response.status_code == 404:
                    continue  # try next version
                else:
                    logger.warning(f"unexpected status {response.status_code} for {url}")

            except Exception as e:
                logger.debug(f"failed to download {url}: {e}")
                continue

        logger.warning(f"no data found for {date.date()} from {satellite}")
        return None

    def _parse_ncei_csv(self, csv_text: str, satellite: str, date: datetime) -> Optional[pd.DataFrame]:
        """
        parse ncei csv format.

        ncei csv format has:
        - header lines starting with #
        - data columns: time_tag, xrsa_flux, xrsb_flux, xrsa_flag, xrsb_flag
        """
        try:
            # skip comment lines, read csv
            lines = [line for line in csv_text.split("\n") if not line.startswith("#") and line.strip()]

            if len(lines) < 2:  # need header + at least one data row
                return None

            # read into dataframe
            df = pd.read_csv(io.StringIO("\n".join(lines)))

            # ncei column names (check first few rows to confirm format)
            # typical: time_tag, xrsa_flux (0.05-0.4nm), xrsb_flux (0.1-0.8nm)
            if "time_tag" not in df.columns:
                logger.warning(f"unexpected csv format for {date.date()}, columns: {df.columns.tolist()}")
                return None

            # rename columns to match our schema
            df = df.rename(
                columns={
                    "time_tag": "timestamp",
                    "xrsa_flux": "flux_short",  # 0.05-0.4 nm (short wavelength, higher energy)
                    "xrsb_flux": "flux_long",  # 0.1-0.8 nm (long wavelength, lower energy)
                }
            )

            # parse timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # convert flux to numeric, handling flags/invalid values
            df["flux_short"] = pd.to_numeric(df["flux_short"], errors="coerce")
            df["flux_long"] = pd.to_numeric(df["flux_long"], errors="coerce")

            # add satellite
            df["satellite"] = satellite

            # filter out invalid/negative values
            df = df[
                (df["flux_short"] > 0) & (df["flux_long"] > 0) & (df["flux_short"].notna()) & (df["flux_long"].notna())
            ].copy()

            # select only needed columns
            df = df[["timestamp", "flux_short", "flux_long", "satellite"]]

            logger.debug(f"parsed {len(df)} valid records for {date.date()}")
            return df

        except Exception as e:
            logger.error(f"failed to parse csv for {date.date()}: {e}")
            return None

    def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 30,
    ) -> dict:
        """
        download data for a date range.

        args:
            start_date: start date
            end_date: end date
            batch_size: days to accumulate before saving

        returns:
            dict with statistics
        """
        logger.info(f"downloading ncei data from {start_date.date()} to {end_date.date()}")

        total_days = (end_date - start_date).days
        total_flux_records = 0
        total_flares = 0
        failed_days = 0

        current_date = start_date
        batch_data = []

        with tqdm(total=total_days, desc="Downloading", unit="day") as pbar:
            while current_date < end_date:
                # download day
                day_data = self.download_day(current_date)

                if day_data is not None and len(day_data) > 0:
                    batch_data.append(day_data)
                else:
                    failed_days += 1

                # save batch when full
                if len(batch_data) >= batch_size or current_date + timedelta(days=1) >= end_date:
                    if batch_data:
                        combined = pd.concat(batch_data, ignore_index=True)

                        # save flux data
                        flux_result = self.persister.save_xray_flux(combined, show_progress=False)
                        if flux_result.get("status") == "success":
                            total_flux_records += flux_result.get("records_inserted", 0)

                        # detect and save flares
                        flares_df = self.flare_detector.detect_flares_from_flux(combined, min_class="C")
                        if flares_df is not None and len(flares_df) > 0:
                            flare_result = self.persister.save_flare_events(flares_df)
                            if flare_result.get("status") == "success":
                                total_flares += flare_result.get("records_inserted", 0)

                        flare_count = len(flares_df) if flares_df is not None else 0
                        logger.info(f"batch saved: {len(combined)} flux records, {flare_count} flares")
                        batch_data = []

                # rate limiting
                time.sleep(0.1)

                current_date += timedelta(days=1)
                pbar.update(1)

        logger.info(
            f"download complete: {total_flux_records:,} flux records, "
            f"{total_flares:,} flares, {failed_days} failed days"
        )

        return {
            "flux_records": total_flux_records,
            "flares_detected": total_flares,
            "failed_days": failed_days,
            "start_date": start_date,
            "end_date": end_date,
        }


def main():
    """main entry point."""
    parser = argparse.ArgumentParser(description="Download historical GOES data from NCEI (2020-2024)")
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
        "--batch-size",
        type=int,
        default=30,
        help="Days to accumulate before saving to database",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database before download",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: download only 7 days",
    )

    args = parser.parse_args()

    # parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    if args.test:
        # test mode: just 7 days
        end_date = start_date + timedelta(days=7)
        logger.info("TEST MODE: downloading only 7 days")

    print("\n" + "=" * 70)
    print("NCEI HISTORICAL GOES DATA DOWNLOAD")
    print("=" * 70)
    print(f"Start date: {start_date.date()}")
    print(f"End date: {end_date.date()}")
    print(f"Total days: {(end_date - start_date).days}")
    print(f"Estimated time: ~{(end_date - start_date).days * 0.5 / 60:.1f} minutes")
    print("=" * 70 + "\n")

    # initialize database if requested
    if args.init_db:
        logger.info("initializing database...")
        init_database(drop_existing=False)

    # download data
    downloader = NCEIHistoricalDownloader()
    stats = downloader.download_date_range(
        start_date=start_date,
        end_date=end_date,
        batch_size=args.batch_size,
    )

    # print summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Flux records: {stats['flux_records']:,}")
    print(f"Flares detected: {stats['flares_detected']:,}")
    print(f"Failed days: {stats['failed_days']}")
    total_days = (end_date - start_date).days
    success_rate = (total_days - stats["failed_days"]) / total_days * 100
    print(f"Success rate: {success_rate:.1f}%")
    print("=" * 70 + "\n")

    # check flare distribution
    print("Checking flare distribution in database...")
    from src.data.database import get_database
    from src.data.schema import FlareEvent
    from sqlalchemy import func

    db = get_database()
    with db.get_session() as session:
        flare_stats = (
            session.query(FlareEvent.class_category, func.count(FlareEvent.id).label("count"))
            .filter(FlareEvent.class_category.in_(["C", "M", "X"]))
            .group_by(FlareEvent.class_category)
            .all()
        )

        print("\nFlare distribution:")
        for category, count in flare_stats:
            print(f"  {category}-class: {count:,}")


if __name__ == "__main__":
    main()
