"""data fetchers for noaa/swpc endpoints."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import DataConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

# cache directory
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class NOAAFetcher:
    """base class for noaa data fetching with retry logic and caching."""

    def __init__(self, timeout: int = 30):
        """
        initialize fetcher with retry logic.

        args:
            timeout: request timeout in seconds
        """
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """create requests session with retry logic."""
        session = requests.Session()

        # retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def fetch_json(self, url: str) -> Optional[List[Dict]]:
        """
        fetch json data from url with error handling.

        args:
            url: endpoint url

        returns:
            parsed json data or none if failed
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"failed to fetch data from {url}: {e}")
            return None


class GOESXRayFetcher(NOAAFetcher):
    """fetcher for goes x-ray flux data."""

    def fetch_recent_flux(self, days: int = 7) -> Optional[pd.DataFrame]:
        """
        fetch recent goes xrs flux data.

        args:
            days: number of days to fetch (7 or 6-hour endpoint)

        returns:
            dataframe with columns: timestamp, flux_short, flux_long, satellite
        """
        # choose endpoint based on days
        if days <= 0.25:  # 6 hours
            url = DataConfig.ENDPOINTS["goes_xrs_6hour"]
        else:
            url = DataConfig.ENDPOINTS["goes_xrs_7day"]

        logger.info(f"fetching goes xrs data from {url}")
        data = self.fetch_json(url)

        if not data:
            return None

        try:
            df = pd.DataFrame(data)

            # parse timestamp
            df["timestamp"] = pd.to_datetime(df["time_tag"])

            # rename flux columns (noaa uses different naming)
            # typical fields: energy, flux, time_tag, satellite
            if "energy" in df.columns:
                # separate short and long wavelength bands
                df_short = df[df["energy"] == "0.05-0.4nm"].copy()
                df_long = df[df["energy"] == "0.1-0.8nm"].copy()

                # merge on timestamp
                result = pd.merge(
                    df_short[["timestamp", "flux", "satellite"]].rename(columns={"flux": "flux_short"}),
                    df_long[["timestamp", "flux"]].rename(columns={"flux": "flux_long"}),
                    on="timestamp",
                    how="outer",
                )
            else:
                # handle alternative format
                result = df.rename(columns={"A_FLUX": "flux_short", "B_FLUX": "flux_long"})

            # convert flux to float, handling negative/invalid values
            result["flux_short"] = pd.to_numeric(result.get("flux_short"), errors="coerce")
            result["flux_long"] = pd.to_numeric(result.get("flux_long"), errors="coerce")

            # sort by timestamp
            result = result.sort_values("timestamp").reset_index(drop=True)

            logger.info(f"fetched {len(result)} xrs flux records")
            return result

        except Exception as e:
            logger.error(f"failed to parse goes xrs data: {e}")
            return None

    def fetch_historical_flux(self, start_date: str, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        fetch historical goes data from noaa archives.

        note: this is a placeholder - actual implementation requires accessing
        the noaa ncei archive which has a different structure per satellite/year.

        args:
            start_date: start date in yyyy-mm-dd format
            end_date: end date in yyyy-mm-dd format (defaults to today)

        returns:
            dataframe with historical flux data
        """
        logger.warning("historical data fetching not yet implemented")
        logger.info("historical goes data requires accessing ncei archives per satellite/year")
        logger.info(
            "recommend manual download from: https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/"
        )

        # placeholder - would need to iterate through dates and satellites
        # goes-13, goes-14, goes-15, goes-16, goes-17, goes-18
        # data is organized by satellite/year/month/day

        return None


class SolarRegionFetcher(NOAAFetcher):
    """fetcher for active solar region data."""

    def fetch_current_regions(self) -> Optional[pd.DataFrame]:
        """
        fetch current active solar regions.

        returns:
            dataframe with solar region data
        """
        url = DataConfig.ENDPOINTS["solar_regions"]
        logger.info(f"fetching solar regions from {url}")

        data = self.fetch_json(url)

        if not data:
            return None

        try:
            df = pd.DataFrame(data)

            # noaa solar regions api format:
            # region, latitude, longitude, location, area, spot_class, extent, number_spots,
            # mag_class, observed_date, etc.

            # parse observed_date to timestamp
            if "observed_date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["observed_date"])
            else:
                df["timestamp"] = datetime.utcnow()

            # copy mag_class to magnetic_type before renaming
            if "mag_class" in df.columns:
                df["magnetic_type"] = df["mag_class"]

            # apply column mappings
            df = df.rename(
                columns={
                    "region": "region_number",
                    "spot_class": "mcintosh_class",
                    "mag_class": "mount_wilson_class",
                    "number_spots": "num_sunspots",
                }
            )

            # filter out rows without region_number (required field)
            initial_count = len(df)
            df = df[df["region_number"].notna()]
            filtered_count = len(df)
            if filtered_count < initial_count:
                logger.warning(f"filtered out {initial_count - filtered_count} records without region_number")

            # ensure region_number is integer (handle any float values)
            if len(df) > 0 and "region_number" in df.columns:
                df["region_number"] = df["region_number"].astype("Int64")  # nullable integer

            logger.info(f"fetched {len(df)} active solar regions")
            return df

        except Exception as e:
            logger.error(f"failed to parse solar region data: {e}")
            return None


class MagnetogramFetcher(NOAAFetcher):
    """fetcher for solar magnetogram data."""

    def fetch_magnetogram_from_regions(self, regions_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        extract magnetogram data from solar regions dataframe.

        note: noaa swpc doesn't provide direct magnetogram json endpoints.
        this extracts magnetic field information from the solar regions data.
        for full magnetogram data, integrate with nasa sdo/hmi jsoc api.

        args:
            regions_df: dataframe with solar region data (from SolarRegionFetcher)

        returns:
            dataframe with magnetogram data
        """
        if regions_df is None or len(regions_df) == 0:
            logger.warning("no region data available for magnetogram extraction")
            return None

        try:
            magnetogram_data = []

            for _, row in regions_df.iterrows():
                # skip rows without region_number (required field)
                region_number = row.get("region_number")
                if pd.isna(region_number) or region_number is None:
                    continue

                # extract magnetic field information from region data
                magnetic_type = row.get("magnetic_type", "")
                magnetic_complexity = self._parse_magnetic_complexity(magnetic_type)

                magnetogram_data.append(
                    {
                        "timestamp": row.get("timestamp", datetime.utcnow()),
                        "region_number": int(region_number),
                        "magnetic_field_polarity": self._parse_polarity(magnetic_type),
                        "magnetic_complexity": magnetic_complexity,
                        "latitude": row.get("latitude"),
                        "longitude": row.get("longitude"),
                        "magnetic_field_strength": None,  # not available from noaa swpc endpoint
                        "source": "noaa_swpc",
                        "data_quality": "good" if magnetic_type else "fair",
                    }
                )

            if not magnetogram_data:
                logger.warning("no valid magnetogram data extracted (all regions missing region_number)")
                return None

            df = pd.DataFrame(magnetogram_data)
            logger.info(f"extracted magnetogram data for {len(df)} regions")
            return df

        except Exception as e:
            logger.error(f"failed to extract magnetogram data: {e}")
            return None

    def _parse_magnetic_complexity(self, magnetic_type: str) -> str:
        """
        parse magnetic complexity from magnetic type string.

        args:
            magnetic_type: string like "Beta-Gamma", "Alpha", etc.

        returns:
            normalized complexity string
        """
        if not magnetic_type:
            return "unknown"

        magnetic_type_lower = magnetic_type.lower()

        # map common magnetic classifications
        if "delta" in magnetic_type_lower:
            return "beta-gamma-delta"
        elif "gamma" in magnetic_type_lower:
            return "beta-gamma"
        elif "beta" in magnetic_type_lower:
            return "beta"
        elif "alpha" in magnetic_type_lower:
            return "alpha"
        else:
            return magnetic_type

    def _parse_polarity(self, magnetic_type: str) -> str:
        """
        parse magnetic polarity from magnetic type.

        args:
            magnetic_type: magnetic type string

        returns:
            polarity: positive, negative, or mixed
        """
        if not magnetic_type:
            return "unknown"

        magnetic_type_lower = magnetic_type.lower()

        # simple heuristic: if multiple types mentioned, likely mixed
        if "gamma" in magnetic_type_lower or "delta" in magnetic_type_lower:
            return "mixed"
        elif "beta" in magnetic_type_lower:
            return "mixed"
        else:
            return "unknown"

    def fetch_sdo_hmi_magnetogram(
        self, start_date: datetime, end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        fetch sdo/hmi magnetogram data from nasa jsoc.

        note: this is a placeholder for future sdo/hmi integration.
        nasa jsoc api requires authentication and has complex query syntax.

        args:
            start_date: start datetime
            end_date: end datetime (defaults to now)

        returns:
            dataframe with magnetogram data (not implemented yet)
        """
        logger.warning("sdo/hmi magnetogram fetching not yet implemented")
        logger.info("requires nasa jsoc api integration")
        logger.info("see: https://jsoc.stanford.edu/ajax/lookdata.html")
        return None


def save_cache(data: pd.DataFrame, cache_name: str):
    """
    save dataframe to cache.

    args:
        data: dataframe to cache
        cache_name: name for cache file
    """
    cache_path = CACHE_DIR / f"{cache_name}.parquet"
    try:
        data.to_parquet(cache_path, index=False)
        logger.info(f"cached data to {cache_path}")
    except Exception as e:
        logger.error(f"failed to save cache: {e}")


def load_cache(cache_name: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    """
    load dataframe from cache if fresh enough.

    args:
        cache_name: name of cache file
        max_age_hours: maximum age of cache in hours

    returns:
        cached dataframe or none if expired/missing
    """
    cache_path = CACHE_DIR / f"{cache_name}.parquet"

    if not cache_path.exists():
        return None

    # check cache age
    cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
    cache_age_hours = cache_age / 3600

    if cache_age_hours > max_age_hours:
        logger.info(f"cache {cache_name} expired ({cache_age_hours:.1f}h old)")
        return None

    try:
        data = pd.read_parquet(cache_path)
        logger.info(f"loaded cache {cache_name} ({cache_age_hours:.1f}h old, {len(data)} records)")
        return data
    except Exception as e:
        logger.error(f"failed to load cache: {e}")
        return None
