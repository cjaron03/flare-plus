"""NASA DONKI client for FLR (solar flare) events.

Fetches FLR events in chunks (<=30 days) to respect API limits.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DonkiClient:
    """Minimal client for NASA DONKI FLR endpoint."""

    BASE_URL = "https://api.nasa.gov/DONKI/FLR"

    def __init__(self, api_key: str = "DEMO_KEY", timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
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

    def fetch_flares(self, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """Fetch FLR events for a date window [start_date, end_date]."""
        params = {"startDate": start_date, "endDate": end_date, "api_key": self.api_key}
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                logger.error("Unexpected DONKI response shape: not a list")
                return None
            return data
        except requests.RequestException as e:
            logger.error(f"DONKI request failed: {e}")
            return None

    def iter_flares_chunked(self, start: datetime, end: datetime, days_per_chunk: int = 30) -> Iterable[List[Dict]]:
        """Yield lists of events per chunk within [start, end]."""
        if start > end:
            return
        current = start
        delta = timedelta(days=days_per_chunk)
        while current <= end:
            chunk_end = min(current + delta, end)
            start_str = current.strftime("%Y-%m-%d")
            end_str = chunk_end.strftime("%Y-%m-%d")
            logger.info(f"Fetching DONKI FLR: {start_str} -> {end_str}")
            events = self.fetch_flares(start_str, end_str) or []
            yield events
            # advance by 1 day after chunk_end to avoid overlap
            current = chunk_end + timedelta(days=1)
