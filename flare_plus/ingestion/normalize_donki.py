"""Normalization utilities for DONKI FLR events.

Converts raw JSON payloads to a dataframe matching SolarFlareEvent schema.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List

import pandas as pd


def _to_utc(dt_str: Any) -> datetime | None:
    if not dt_str:
        return None
    try:
        # pandas handles Z and timezone offsets; convert to UTC and drop tzinfo
        ts = pd.to_datetime(dt_str, utc=True)
        if isinstance(ts, pd.Timestamp):
            return ts.tz_convert("UTC").tz_localize(None).to_pydatetime()
        return None
    except Exception:
        return None


def normalize_donki_flares(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Normalize DONKI FLR records to the SolarFlareEvent schema fields.

    Output columns:
    - external_id, source, begin_time, peak_time, end_time, class_type,
      source_location, active_region_num, instruments, linked_events,
      raw_payload
    """
    rows: List[Dict[str, Any]] = []
    for r in records:
        # DONKI FLR often includes: flrID, beginTime, peakTime, endTime,
        # classType, sourceLocation, activeRegionNum, instruments, linkedEvents
        external_id = r.get("flrID") or r.get("id") or r.get("activityID")
        begin_time = _to_utc(r.get("beginTime"))
        peak_time = _to_utc(r.get("peakTime"))
        end_time = _to_utc(r.get("endTime"))
        class_type = r.get("classType")
        source_location = r.get("sourceLocation")
        active_region_num = r.get("activeRegionNum")

        # ensure integer or None for active_region_num
        if pd.isna(active_region_num):
            active_region_num = None
        else:
            try:
                active_region_num = int(active_region_num) if active_region_num is not None else None
            except (TypeError, ValueError):
                active_region_num = None

        instruments = json.dumps(r.get("instruments") or [])
        linked_events = json.dumps(r.get("linkedEvents") or [])
        raw_payload = json.dumps(r)

        rows.append(
            {
                "external_id": external_id,
                "source": "nasa_donki",
                "begin_time": begin_time,
                "peak_time": peak_time,
                "end_time": end_time,
                "class_type": class_type,
                "source_location": source_location,
                "active_region_num": active_region_num,
                "instruments": instruments,
                "linked_events": linked_events,
                "raw_payload": raw_payload,
            }
        )

    df = pd.DataFrame(rows)
    # enforce column order for clarity
    columns = [
        "external_id",
        "source",
        "begin_time",
        "peak_time",
        "end_time",
        "class_type",
        "source_location",
        "active_region_num",
        "instruments",
        "linked_events",
        "raw_payload",
    ]
    return df[columns] if len(df) else pd.DataFrame(columns=columns)
