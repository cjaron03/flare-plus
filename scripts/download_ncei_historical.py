#!/usr/bin/env python
"""download historical GOES XRS 1-minute data from NCEI archive and persist to DB.

This ingests legacy monthly CSV bundles from:
https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/

The archive currently exposes historical GOES 14/15 monthly bundles and is a
practical way to improve long-range flux continuity without requiring netCDF
dependencies.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func  # noqa: E402

from src.config import DataConfig  # noqa: E402
from src.data.database import get_database, init_database  # noqa: E402
from src.data.flare_detector import FlareDetector  # noqa: E402
from src.data.persistence import DataPersister  # noqa: E402
from src.data.schema import FlareEvent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg"
HREF_PATTERN = re.compile(r'href="([^"]+)"')
XRS_FILE_PATTERN = re.compile(r"^g\d{2}_xrs_1m_\d{8}_\d{8}\.csv$")


def month_iter(start_date: datetime, end_date: datetime) -> Iterable[Tuple[int, int]]:
    """Yield (year, month) tuples from start_date inclusive to end_date exclusive."""
    cursor = datetime(start_date.year, start_date.month, 1)
    limit = datetime(end_date.year, end_date.month, 1)
    while cursor <= limit:
        if cursor >= end_date:
            break
        yield cursor.year, cursor.month
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1)


class NCEIHistoricalDownloader:
    """Downloads and persists monthly historical GOES XRS data from NCEI avg archive."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "flare-plus-historical-xrs/1.0"})
        self.persister = DataPersister()
        self.flare_detector = FlareDetector()
        self._available_years: Optional[List[int]] = None

    def _fetch_text(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=45)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.text
        except Exception as exc:
            logger.warning("request failed for %s: %s", url, exc)
            return None

    @staticmethod
    def _parse_links(html: str) -> List[str]:
        return HREF_PATTERN.findall(html)

    def get_available_years(self) -> List[int]:
        """Query the archive root and cache available YYYY directories."""
        if self._available_years is not None:
            return self._available_years

        html = self._fetch_text(f"{self.base_url}/")
        if not html:
            self._available_years = []
            return self._available_years

        years: List[int] = []
        for link in self._parse_links(html):
            cleaned = link.strip("/")
            if cleaned.isdigit() and len(cleaned) == 4:
                years.append(int(cleaned))

        self._available_years = sorted(set(years))
        return self._available_years

    def _list_month_satellites(self, year: int, month: int) -> List[str]:
        html = self._fetch_text(f"{self.base_url}/{year}/{month:02d}/")
        if not html:
            return []

        satellites: List[str] = []
        for link in self._parse_links(html):
            cleaned = link.strip("/")
            if cleaned.startswith("goes") and cleaned[4:].isdigit():
                satellites.append(cleaned)
        return sorted(set(satellites), key=self._satellite_sort_key, reverse=True)

    @staticmethod
    def _satellite_sort_key(satellite: str) -> int:
        tail = satellite[4:]
        if tail.isdigit():
            return int(tail)
        return -1

    def _find_xrs_csv(self, year: int, month: int, satellite: str) -> Optional[Tuple[str, str]]:
        csv_dir = f"{self.base_url}/{year}/{month:02d}/{satellite}/csv/"
        html = self._fetch_text(csv_dir)
        if not html:
            return None

        files = [link for link in self._parse_links(html) if XRS_FILE_PATTERN.match(link)]
        if not files:
            return None

        filename = sorted(files)[-1]
        return f"{csv_dir}{filename}", filename

    @staticmethod
    def _normalize_satellite_name(satellite: str) -> str:
        sat = satellite.upper()
        if sat.startswith("GOES") and "-" not in sat:
            return sat.replace("GOES", "GOES-")
        return sat

    def _parse_xrs_payload(self, payload: str, satellite: str) -> Optional[pd.DataFrame]:
        lines = payload.splitlines()
        data_idx = None
        for idx, line in enumerate(lines):
            if line.strip().lower() == "data:":
                data_idx = idx
                break

        if data_idx is None or data_idx + 1 >= len(lines):
            return None

        data_lines = [line for line in lines[data_idx + 1 :] if line.strip()]
        if len(data_lines) < 2:
            return None

        try:
            df = pd.read_csv(io.StringIO("\n".join(data_lines)), skipinitialspace=True)
        except Exception as exc:
            logger.warning("failed to parse xrs csv payload: %s", exc)
            return None

        required_cols = {"time_tag", "A_AVG", "B_AVG"}
        if not required_cols.issubset(set(df.columns)):
            logger.warning("xrs csv missing columns. found=%s", df.columns.tolist())
            return None

        parsed = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["time_tag"], errors="coerce"),
                "flux_short": pd.to_numeric(df["A_AVG"], errors="coerce"),
                "flux_long": pd.to_numeric(df["B_AVG"], errors="coerce"),
            }
        )
        parsed["satellite"] = self._normalize_satellite_name(satellite)
        parsed = parsed.dropna(subset=["timestamp", "flux_short", "flux_long"])
        parsed = parsed[(parsed["flux_short"] > 0) & (parsed["flux_long"] > 0)].copy()
        parsed = parsed.sort_values("timestamp").reset_index(drop=True)
        return parsed

    def download_month(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """Download one month by choosing the satellite source with most valid XRS rows."""
        satellites = self._list_month_satellites(year, month)
        if not satellites:
            logger.info("no satellite directories for %04d-%02d", year, month)
            return None

        best_df = None
        best_satellite = None

        for satellite in satellites:
            csv_info = self._find_xrs_csv(year, month, satellite)
            if not csv_info:
                continue

            csv_url, filename = csv_info
            payload = self._fetch_text(csv_url)
            if not payload:
                continue

            parsed = self._parse_xrs_payload(payload, satellite=satellite)
            if parsed is None or parsed.empty:
                continue

            logger.info(
                "loaded %d rows from %s (%04d-%02d)",
                len(parsed),
                filename,
                year,
                month,
            )
            if best_df is None or len(parsed) > len(best_df):
                best_df = parsed
                best_satellite = satellite

        if best_df is not None:
            logger.info(
                "selected satellite %s for %04d-%02d (%d rows)",
                best_satellite,
                year,
                month,
                len(best_df),
            )
        return best_df

    def _persist_batch(self, frames: List[pd.DataFrame]) -> Tuple[int, int]:
        if not frames:
            return 0, 0

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        flux_result = self.persister.save_xray_flux(
            combined,
            source_name="ncei_historical_xrs",
            show_progress=False,
        )
        inserted_flux = int(flux_result.get("records_inserted", 0))

        flares_detected = 0
        flares_df = self.flare_detector.detect_flares_from_flux(combined, min_class="C")
        if flares_df is not None and len(flares_df) > 0:
            flare_result = self.persister.save_flare_events(
                flares_df,
                source_name="ncei_historical_detected",
            )
            flares_detected = int(flare_result.get("records_inserted", 0))

        return inserted_flux, flares_detected

    def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        batch_size_months: int = 3,
    ) -> Dict[str, object]:
        """Download monthly bundles, persist flux, and auto-detect flare events."""
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

        years_available = set(self.get_available_years())
        months = list(month_iter(start_date, end_date))

        total_flux_records = 0
        total_flares = 0
        months_succeeded = 0
        months_failed = 0
        months_unavailable = 0
        buffered_frames: List[pd.DataFrame] = []

        with tqdm(total=len(months), desc="Downloading months", unit="month") as pbar:
            for year, month in months:
                if year not in years_available:
                    months_unavailable += 1
                    pbar.update(1)
                    continue

                month_df = self.download_month(year, month)
                if month_df is None or month_df.empty:
                    months_failed += 1
                    pbar.update(1)
                    continue

                month_df = month_df[(month_df["timestamp"] >= start_date) & (month_df["timestamp"] < end_date)].copy()
                if month_df.empty:
                    pbar.update(1)
                    continue

                buffered_frames.append(month_df)
                months_succeeded += 1

                if len(buffered_frames) >= max(1, batch_size_months):
                    inserted_flux, inserted_flares = self._persist_batch(buffered_frames)
                    total_flux_records += inserted_flux
                    total_flares += inserted_flares
                    buffered_frames = []

                time.sleep(0.1)
                pbar.update(1)

        if buffered_frames:
            inserted_flux, inserted_flares = self._persist_batch(buffered_frames)
            total_flux_records += inserted_flux
            total_flares += inserted_flares

        return {
            "flux_records": total_flux_records,
            "flares_detected": total_flares,
            "months_total": len(months),
            "months_succeeded": months_succeeded,
            "months_failed": months_failed,
            "months_unavailable": months_unavailable,
            "start_date": start_date,
            "end_date": end_date,
            "available_years": sorted(years_available),
        }


def print_db_flare_distribution() -> None:
    db = get_database()
    with db.get_session() as session:
        flare_stats = (
            session.query(FlareEvent.class_category, func.count(FlareEvent.id).label("count"))
            .filter(FlareEvent.class_category.in_(["C", "M", "X"]))
            .group_by(FlareEvent.class_category)
            .all()
        )

    print("\nFlare distribution in DB:")
    for category, count in flare_stats:
        print(f"  {category}-class: {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical GOES XRS 1-minute data from NCEI avg archive")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2017-01-01",
        help="start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="months per persistence batch (default: 3)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=(DataConfig.ENDPOINTS.get("goes_archive") or DEFAULT_BASE_URL),
        help=f"NCEI archive base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="initialize database before download",
    )

    args = parser.parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    print("\n" + "=" * 72)
    print("NCEI HISTORICAL XRS DOWNLOAD")
    print("=" * 72)
    print(f"Start date: {start_date.date()}")
    print(f"End date:   {end_date.date()}")
    print(f"Base URL:   {args.base_url}")
    print("=" * 72 + "\n")

    if args.init_db:
        logger.info("initializing database...")
        init_database(drop_existing=False)

    downloader = NCEIHistoricalDownloader(base_url=args.base_url)
    stats = downloader.download_date_range(
        start_date=start_date,
        end_date=end_date,
        batch_size_months=max(1, args.batch_size),
    )

    print("\n" + "=" * 72)
    print("DOWNLOAD COMPLETE")
    print("=" * 72)
    print(f"Flux records inserted: {stats['flux_records']:,}")
    print(f"Flares detected:       {stats['flares_detected']:,}")
    print(f"Months attempted:      {stats['months_total']}")
    print(f"Months succeeded:      {stats['months_succeeded']}")
    print(f"Months failed:         {stats['months_failed']}")
    print(f"Months unavailable:    {stats['months_unavailable']}")
    print(f"Available years:       {stats['available_years']}")
    print("=" * 72)

    print_db_flare_distribution()
    print("\n" + "=" * 72 + "\n")


if __name__ == "__main__":
    main()
