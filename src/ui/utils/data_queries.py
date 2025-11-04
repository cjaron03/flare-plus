"""database query utilities for ui dashboard."""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from src.data.database import get_database
from src.data.schema import FlareEvent, GOESXRayFlux, SolarRegion

logger = logging.getLogger(__name__)


def query_historical_flares(
    start_date: datetime,
    end_date: datetime,
    min_class: str = "B",
    region_number: Optional[int] = None,
) -> pd.DataFrame:
    """
    query historical flare events from database.

    args:
        start_date: start of date range
        end_date: end of date range
        min_class: minimum flare class to include (B, C, M, X)
        region_number: optional region number to filter by

    returns:
        dataframe with flare events (empty if none found)
    """
    try:
        db = get_database()
        with db.get_session() as session:
            query = session.query(FlareEvent).filter(
                FlareEvent.start_time >= start_date,
                FlareEvent.start_time <= end_date,
            )

            # filter by minimum class
            class_order = {"B": 0, "C": 1, "M": 2, "X": 3}
            if min_class in class_order:
                min_order = class_order[min_class]
                # include classes with order >= min_order
                valid_classes = [c for c, order in class_order.items() if order >= min_order]
                query = query.filter(FlareEvent.class_category.in_(valid_classes))

            # filter by region if specified
            if region_number is not None:
                query = query.filter(FlareEvent.active_region == region_number)

            records = query.order_by(FlareEvent.start_time).all()

            if len(records) == 0:
                return pd.DataFrame()

            # convert to dataframe
            data = []
            for r in records:
                data.append(
                    {
                        "start_time": r.start_time,
                        "peak_time": r.peak_time,
                        "end_time": r.end_time,
                        "flare_class": r.flare_class,
                        "class_category": r.class_category,
                        "class_magnitude": r.class_magnitude,
                        "active_region": r.active_region,
                        "location": r.location,
                        "source": r.source,
                    }
                )

            return pd.DataFrame(data)

    except Exception as e:
        logger.error(f"error querying historical flares: {e}")
        return pd.DataFrame()


def get_latest_data_timestamps() -> Dict[str, Any]:
    """
    get latest timestamps for each data type.

    returns:
        dict with latest timestamps and record counts
    """
    try:
        db = get_database()
        result = {
            "flux_latest": None,
            "flux_count": 0,
            "regions_latest": None,
            "regions_count": 0,
            "flares_latest": None,
            "flares_count": 0,
        }

        with db.get_session() as session:
            from sqlalchemy import func

            # flux data
            flux_max = session.query(func.max(GOESXRayFlux.timestamp)).scalar()
            flux_count = session.query(func.count(GOESXRayFlux.id)).scalar()
            result["flux_latest"] = flux_max
            result["flux_count"] = flux_count or 0

            # solar regions
            regions_max = session.query(func.max(SolarRegion.timestamp)).scalar()
            regions_count = session.query(func.count(SolarRegion.id)).scalar()
            result["regions_latest"] = regions_max
            result["regions_count"] = regions_count or 0

            # flare events
            flares_max = session.query(func.max(FlareEvent.start_time)).scalar()
            flares_count = session.query(func.count(FlareEvent.id)).scalar()
            result["flares_latest"] = flares_max
            result["flares_count"] = flares_count or 0

        return result

    except Exception as e:
        logger.error(f"error getting latest timestamps: {e}")
        return {
            "flux_latest": None,
            "flux_count": 0,
            "regions_latest": None,
            "regions_count": 0,
            "flares_latest": None,
            "flares_count": 0,
        }


def calculate_data_freshness(latest_timestamp: Optional[datetime]) -> Dict[str, Any]:
    """
    calculate data freshness metrics.

    args:
        latest_timestamp: latest data timestamp

    returns:
        dict with freshness info (hours_ago, status, color)
    """
    if latest_timestamp is None:
        return {
            "hours_ago": None,
            "status": "unknown",
            "color": "gray",
        }

    now = datetime.utcnow()
    if latest_timestamp.tzinfo is not None:
        # assume utc if timezone aware
        now = now.replace(tzinfo=latest_timestamp.tzinfo)

    hours_ago = (now - latest_timestamp).total_seconds() / 3600

    if hours_ago < 24:
        status = "fresh"
        color = "green"
    elif hours_ago < 48:
        status = "stale"
        color = "yellow"
    else:
        status = "outdated"
        color = "red"

    return {
        "hours_ago": hours_ago,
        "status": status,
        "color": color,
    }
