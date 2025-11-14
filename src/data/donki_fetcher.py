"""nasa donki api fetcher for historical flare events."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DonkiFetcher:
    """fetch historical flare data from nasa donki api."""

    BASE_URL = "https://api.nasa.gov/DONKI/FLR"

    def __init__(self, api_key: str = "DEMO_KEY", timeout: int = 30):
        """
        initialize donki fetcher.

        args:
            api_key: nasa api key (default: DEMO_KEY for testing)
            timeout: request timeout in seconds
        """
        self.api_key = api_key
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

    def fetch_flares(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        fetch flare events between dates.

        donki limits: 30 days per request max.

        args:
            start_date: start date for query
            end_date: end date for query

        returns:
            list of flare event dictionaries
        """
        params = {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "api_key": self.api_key,
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"failed to fetch donki flares from {start_date.date()} to {end_date.date()}: {e}")
            return []

    def fetch_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        fetch flares across large date range.

        handles 30-day limit by chunking requests.

        args:
            start_date: start date for query
            end_date: end date for query

        returns:
            list of all flare event dictionaries
        """
        all_flares = []
        current = start_date

        while current < end_date:
            chunk_end = min(current + timedelta(days=30), end_date)

            logger.info(f"fetching {current.date()} to {chunk_end.date()}...")
            flares = self.fetch_flares(current, chunk_end)
            all_flares.extend(flares)

            current = chunk_end + timedelta(days=1)

        return all_flares

    def parse_class(self, class_str: str) -> Tuple[Optional[str], Optional[float]]:
        """
        parse class string like 'M2.5' into ('M', 2.5).

        args:
            class_str: flare class string (e.g., 'M2.5', 'X1.2', 'C3.4')

        returns:
            tuple of (category, magnitude) or (None, None) if invalid
        """
        if not class_str:
            return (None, None)

        try:
            category = class_str[0].upper()  # 'M', 'X', 'C', etc.
            intensity = float(class_str[1:])  # 2.5
            return (category, intensity)
        except (ValueError, IndexError):
            logger.warning(f"failed to parse flare class: {class_str}")
            return (None, None)
