"""High-level runner for DONKI solar flare ingestion."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from flare_plus.ingestion.donki_client import DonkiClient
from flare_plus.ingestion.normalize_donki import normalize_donki_flares
from flare_plus.ingestion.store_flares import upsert_solar_flares

logger = logging.getLogger(__name__)


def run_ingestion(
    start_date: datetime,
    end_date: Optional[datetime] = None,
    api_key: str = "DEMO_KEY",
    days_per_chunk: int = 30,
) -> dict:
    """Fetch DONKI FLR events, normalize, and upsert into DB."""
    end_date = end_date or datetime.utcnow()
    client = DonkiClient(api_key=api_key)

    aggregated = []
    for events in client.iter_flares_chunked(start_date, end_date, days_per_chunk=days_per_chunk):
        if not events:
            continue
        aggregated.extend(events)

    if not aggregated:
        logger.info("No DONKI FLR events found for the requested window.")
        return {"records_fetched": 0, "records_inserted": 0, "records_updated": 0}

    df = normalize_donki_flares(aggregated)
    df = df[df["external_id"].notna()]
    df = df.drop_duplicates(subset=["external_id", "source"], keep="last")
    df = df.sort_values("peak_time").reset_index(drop=True)

    stats = upsert_solar_flares(df)
    logger.info("DONKI ingestion finished: %s", stats)
    return stats
