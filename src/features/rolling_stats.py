"""rolling statistics and recency-weighted flare counts."""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_timestamp(ts: datetime) -> datetime:
    """normalize timestamp to timezone-naive UTC for pandas compatibility."""
    if ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts


def compute_rolling_statistics(
    data: pd.DataFrame,
    timestamp: datetime,
    value_column: str,
    windows_hours: List[int],
    aggregation_functions: List[str] = None,
) -> Dict[str, Any]:
    """
    compute rolling statistics over multiple time windows.

    args:
        data: dataframe with timestamp and value columns
        timestamp: current timestamp to compute statistics from
        value_column: name of column to aggregate
        windows_hours: list of window sizes in hours (e.g., [6, 12, 24])
        aggregation_functions: list of functions to apply (default: ['mean', 'std', 'min', 'max'])

    returns:
        dict with rolling statistics for each window
    """
    if aggregation_functions is None:
        aggregation_functions = ["mean", "std", "min", "max"]

    if data is None or len(data) == 0 or value_column not in data.columns:
        return {}

    # filter data up to current timestamp
    valid_data = data[data["timestamp"] <= timestamp].copy()
    if len(valid_data) == 0:
        return {}

    valid_data = valid_data.sort_values("timestamp")

    features: Dict[str, Any] = {}

    timestamp = _normalize_timestamp(timestamp)

    for window_hours in windows_hours:
        # filter data within window
        window_start = timestamp - timedelta(hours=window_hours)
        window_data = valid_data[valid_data["timestamp"] >= window_start].copy()

        if len(window_data) == 0:
            # no data in window, set all features to None
            for func in aggregation_functions:
                features[f"{value_column}_{window_hours}h_{func}"] = None
            continue

        values = window_data[value_column].dropna()

        if len(values) == 0:
            # no valid values in window
            for func in aggregation_functions:
                features[f"{value_column}_{window_hours}h_{func}"] = None
            continue

        # compute statistics
        for func in aggregation_functions:
            if func == "mean":
                features[f"{value_column}_{window_hours}h_mean"] = values.mean()
            elif func == "std":
                features[f"{value_column}_{window_hours}h_std"] = values.std()
            elif func == "min":
                features[f"{value_column}_{window_hours}h_min"] = values.min()
            elif func == "max":
                features[f"{value_column}_{window_hours}h_max"] = values.max()
            elif func == "median":
                features[f"{value_column}_{window_hours}h_median"] = values.median()
            elif func == "count":
                features[f"{value_column}_{window_hours}h_count"] = len(values)
            elif func == "sum":
                features[f"{value_column}_{window_hours}h_sum"] = values.sum()

    return features


def compute_recency_weighted_flare_counts(
    flare_data: pd.DataFrame,
    timestamp: datetime,
    flare_classes: List[str],
    windows_hours: List[int],
    decay_factor: float = 0.9,
) -> Dict[str, Any]:
    """
    compute recency-weighted flare counts by class over multiple time windows.

    recent flares are weighted more heavily than older flares using exponential decay.

    args:
        flare_data: dataframe with columns: peak_time, class_category
        timestamp: current timestamp
        flare_classes: list of flare classes to count (e.g., ['B', 'C', 'M', 'X'])
        windows_hours: list of window sizes in hours (e.g., [6, 12, 24])
        decay_factor: exponential decay factor per hour (default: 0.9)

    returns:
        dict with recency-weighted flare counts for each class and window
    """
    if flare_data is None or len(flare_data) == 0:
        return {}

    timestamp = _normalize_timestamp(timestamp)
    features = {}

    for window_hours in windows_hours:
        # filter flares within window
        window_start = timestamp - timedelta(hours=window_hours)
        window_flares = flare_data[
            (flare_data["peak_time"] >= window_start) & (flare_data["peak_time"] <= timestamp)
        ].copy()

        if len(window_flares) == 0:
            # no flares in window, set all counts to 0
            for flare_class in flare_classes:
                features[f"flare_{flare_class}_{window_hours}h_count"] = 0
                features[f"flare_{flare_class}_{window_hours}h_weighted_count"] = 0.0
            continue

        # compute weighted counts for each class
        for flare_class in flare_classes:
            class_flares = window_flares[window_flares["class_category"] == flare_class].copy()

            # simple count
            count = len(class_flares)
            features[f"flare_{flare_class}_{window_hours}h_count"] = count

            # recency-weighted count
            if count > 0:
                weighted_sum = 0.0
                for _, flare in class_flares.iterrows():
                    # compute hours since flare
                    hours_ago = (timestamp - flare["peak_time"]).total_seconds() / 3600
                    # exponential decay weight
                    weight = decay_factor**hours_ago
                    weighted_sum += weight

                features[f"flare_{flare_class}_{window_hours}h_weighted_count"] = weighted_sum
            else:
                features[f"flare_{flare_class}_{window_hours}h_weighted_count"] = 0.0

        # also compute total weighted count across all classes
        total_weighted = 0.0
        for _, flare in window_flares.iterrows():
            hours_ago = (timestamp - flare["peak_time"]).total_seconds() / 3600
            weight = decay_factor**hours_ago
            total_weighted += weight

        features[f"flare_total_{window_hours}h_weighted_count"] = total_weighted

    return features
