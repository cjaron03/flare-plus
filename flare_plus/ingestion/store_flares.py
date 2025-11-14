"""Storage helpers for DONKI solar flare events.

Provides idempotent upsert into `solar_flare_events` table.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.data.database import get_database
from src.data.schema import SolarFlareEvent


def upsert_solar_flares(df: pd.DataFrame, session: Optional[Session] = None) -> Dict[str, int | str | None]:
    """Upsert normalized flare events into the database.

    Works across dialects. Uses explicit lookup+update for non-Postgres engines
    to remain sqlite-friendly in tests. Accepts an optional SQLAlchemy session
    (handy for unit tests) and otherwise manages its own connection.
    """
    stats = {
        "records_fetched": int(len(df)),
        "records_inserted": 0,
        "records_updated": 0,
        "status": "success",
        "error_message": None,
    }

    def _process(sess: Session) -> None:
        for _, row in df.iterrows():
            ext_id = row["external_id"]
            src = row["source"]
            existing = (
                sess.query(SolarFlareEvent)
                .filter(SolarFlareEvent.external_id == ext_id, SolarFlareEvent.source == src)
                .first()
            )

            if existing:
                existing.begin_time = row.get("begin_time")
                existing.peak_time = row.get("peak_time")
                existing.end_time = row.get("end_time")
                existing.class_type = row.get("class_type")
                existing.source_location = row.get("source_location")
                existing.active_region_num = row.get("active_region_num")
                existing.instruments = row.get("instruments")
                existing.linked_events = row.get("linked_events")
                existing.raw_payload = row.get("raw_payload")
                existing.updated_at = datetime.utcnow()
                stats["records_updated"] += 1
            else:
                sess.add(
                    SolarFlareEvent(
                        external_id=ext_id,
                        source=src,
                        begin_time=row.get("begin_time"),
                        peak_time=row.get("peak_time"),
                        end_time=row.get("end_time"),
                        class_type=row.get("class_type"),
                        source_location=row.get("source_location"),
                        active_region_num=row.get("active_region_num"),
                        instruments=row.get("instruments"),
                        linked_events=row.get("linked_events"),
                        raw_payload=row.get("raw_payload"),
                    )
                )
                stats["records_inserted"] += 1

    try:
        if session is not None:
            _process(session)
        else:
            db = get_database()
            with db.get_session() as scoped_session:
                _process(scoped_session)
    except Exception as e:  # pragma: no cover
        stats["status"] = "failure"
        stats["error_message"] = str(e)

    return stats
