"""flux trend features from goes x-ray flux data."""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_flux_trends(
    flux_data: pd.DataFrame,
    timestamp: datetime,
    lookback_hours: int = 24,
    window_hours: int = 6,
) -> Dict[str, Any]:
    """
    compute flux trend features from goes x-ray flux data.

    args:
        flux_data: dataframe with columns: timestamp, flux_short, flux_long
        timestamp: current timestamp to compute trends from
        lookback_hours: number of hours to look back
        window_hours: window size for trend computation

    returns:
        dict with flux trend features
    """
    if flux_data is None or len(flux_data) == 0:
        return {
            "flux_short_mean": None,
            "flux_short_max": None,
            "flux_short_min": None,
            "flux_short_std": None,
            "flux_long_mean": None,
            "flux_long_max": None,
            "flux_long_min": None,
            "flux_long_std": None,
            "flux_trend_short": None,
            "flux_trend_long": None,
            "flux_ratio_short_long": None,
        }

    # filter data within lookback window
    cutoff_time = timestamp - timedelta(hours=lookback_hours)
    recent_data = flux_data[flux_data["timestamp"] >= cutoff_time].copy()

    if len(recent_data) == 0:
        return {
            "flux_short_mean": None,
            "flux_short_max": None,
            "flux_short_min": None,
            "flux_short_std": None,
            "flux_long_mean": None,
            "flux_long_max": None,
            "flux_long_min": None,
            "flux_long_std": None,
            "flux_trend_short": None,
            "flux_trend_long": None,
            "flux_ratio_short_long": None,
        }

    # sort by timestamp
    recent_data = recent_data.sort_values("timestamp")

    # compute basic statistics for short wavelength
    flux_short = recent_data["flux_short"].dropna()
    flux_short_mean = flux_short.mean() if len(flux_short) > 0 else None
    flux_short_max = flux_short.max() if len(flux_short) > 0 else None
    flux_short_min = flux_short.min() if len(flux_short) > 0 else None
    flux_short_std = flux_short.std() if len(flux_short) > 0 else None

    # compute basic statistics for long wavelength
    flux_long = recent_data["flux_long"].dropna()
    flux_long_mean = flux_long.mean() if len(flux_long) > 0 else None
    flux_long_max = flux_long.max() if len(flux_long) > 0 else None
    flux_long_min = flux_long.min() if len(flux_long) > 0 else None
    flux_long_std = flux_long.std() if len(flux_long) > 0 else None

    # compute trend (slope) over recent window
    window_cutoff = timestamp - timedelta(hours=window_hours)
    window_data = recent_data[recent_data["timestamp"] >= window_cutoff].copy()

    flux_trend_short = None
    flux_trend_long = None

    if len(window_data) >= 2:
        # compute linear trend (slope) for short wavelength
        flux_short_window = window_data["flux_short"].dropna()
        if len(flux_short_window) >= 2:
            x = np.arange(len(flux_short_window))
            y = flux_short_window.values
            coeffs = np.polyfit(x, y, 1)
            flux_trend_short = coeffs[0]  # slope

        # compute linear trend (slope) for long wavelength
        flux_long_window = window_data["flux_long"].dropna()
        if len(flux_long_window) >= 2:
            x = np.arange(len(flux_long_window))
            y = flux_long_window.values
            coeffs = np.polyfit(x, y, 1)
            flux_trend_long = coeffs[0]  # slope

    # compute ratio of short to long wavelength (indicates spectral hardness)
    flux_ratio_short_long = None
    if flux_short_mean and flux_long_mean and flux_long_mean > 0:
        flux_ratio_short_long = flux_short_mean / flux_long_mean

    return {
        "flux_short_mean": flux_short_mean,
        "flux_short_max": flux_short_max,
        "flux_short_min": flux_short_min,
        "flux_short_std": flux_short_std,
        "flux_long_mean": flux_long_mean,
        "flux_long_max": flux_long_max,
        "flux_long_min": flux_long_min,
        "flux_long_std": flux_long_std,
        "flux_trend_short": flux_trend_short,
        "flux_trend_long": flux_trend_long,
        "flux_ratio_short_long": flux_ratio_short_long,
    }


def compute_flux_rate_of_change(
    flux_data: pd.DataFrame,
    timestamp: datetime,
    lookback_hours: int = 12,
) -> Dict[str, Any]:
    """
    compute rate of change features for flux data.

    args:
        flux_data: dataframe with columns: timestamp, flux_short, flux_long
        timestamp: current timestamp
        lookback_hours: number of hours to look back

    returns:
        dict with rate of change features
    """
    if flux_data is None or len(flux_data) == 0:
        return {
            "flux_short_rate_of_change": None,
            "flux_long_rate_of_change": None,
            "flux_short_acceleration": None,
            "flux_long_acceleration": None,
        }

    # filter data within lookback window
    cutoff_time = timestamp - timedelta(hours=lookback_hours)
    recent_data = flux_data[flux_data["timestamp"] >= cutoff_time].copy()

    if len(recent_data) < 2:
        return {
            "flux_short_rate_of_change": None,
            "flux_long_rate_of_change": None,
            "flux_short_acceleration": None,
            "flux_long_acceleration": None,
        }

    # sort by timestamp
    recent_data = recent_data.sort_values("timestamp")

    # compute rate of change (first derivative)
    flux_short_rate = None
    flux_long_rate = None

    flux_short_valid = recent_data["flux_short"].dropna()
    if len(flux_short_valid) >= 2:
        # compute change per hour
        time_diff_hours = (recent_data["timestamp"].iloc[-1] - recent_data["timestamp"].iloc[0]).total_seconds() / 3600
        if time_diff_hours > 0:
            flux_diff = flux_short_valid.iloc[-1] - flux_short_valid.iloc[0]
            flux_short_rate = flux_diff / time_diff_hours

    flux_long_valid = recent_data["flux_long"].dropna()
    if len(flux_long_valid) >= 2:
        time_diff_hours = (recent_data["timestamp"].iloc[-1] - recent_data["timestamp"].iloc[0]).total_seconds() / 3600
        if time_diff_hours > 0:
            flux_diff = flux_long_valid.iloc[-1] - flux_long_valid.iloc[0]
            flux_long_rate = flux_diff / time_diff_hours

    # compute acceleration (second derivative) - change in rate of change
    flux_short_accel = None
    flux_long_accel = None

    if len(recent_data) >= 3:
        # split into two halves and compute rate for each
        mid_point = len(recent_data) // 2
        first_half = recent_data.iloc[:mid_point]
        second_half = recent_data.iloc[mid_point:]

        # compute rate for first half
        flux_short_first = first_half["flux_short"].dropna()
        if len(flux_short_first) >= 2:
            time_diff_1 = (first_half["timestamp"].iloc[-1] - first_half["timestamp"].iloc[0]).total_seconds() / 3600
            if time_diff_1 > 0:
                flux_diff_1 = flux_short_first.iloc[-1] - flux_short_first.iloc[0]
                rate_1 = flux_diff_1 / time_diff_1

                # compute rate for second half
                flux_short_second = second_half["flux_short"].dropna()
                if len(flux_short_second) >= 2:
                    time_diff_2 = (
                        second_half["timestamp"].iloc[-1] - second_half["timestamp"].iloc[0]
                    ).total_seconds() / 3600
                    if time_diff_2 > 0:
                        flux_diff_2 = flux_short_second.iloc[-1] - flux_short_second.iloc[0]
                        rate_2 = flux_diff_2 / time_diff_2

                        # acceleration = change in rate / time
                        total_time = time_diff_1 + time_diff_2
                        if total_time > 0:
                            flux_short_accel = (rate_2 - rate_1) / total_time

        # same for long wavelength
        flux_long_first = first_half["flux_long"].dropna()
        if len(flux_long_first) >= 2:
            time_diff_1 = (first_half["timestamp"].iloc[-1] - first_half["timestamp"].iloc[0]).total_seconds() / 3600
            if time_diff_1 > 0:
                flux_diff_1 = flux_long_first.iloc[-1] - flux_long_first.iloc[0]
                rate_1 = flux_diff_1 / time_diff_1

                flux_long_second = second_half["flux_long"].dropna()
                if len(flux_long_second) >= 2:
                    time_diff_2 = (
                        second_half["timestamp"].iloc[-1] - second_half["timestamp"].iloc[0]
                    ).total_seconds() / 3600
                    if time_diff_2 > 0:
                        flux_diff_2 = flux_long_second.iloc[-1] - flux_long_second.iloc[0]
                        rate_2 = flux_diff_2 / time_diff_2

                        total_time = time_diff_1 + time_diff_2
                        if total_time > 0:
                            flux_long_accel = (rate_2 - rate_1) / total_time

    return {
        "flux_short_rate_of_change": flux_short_rate,
        "flux_long_rate_of_change": flux_long_rate,
        "flux_short_acceleration": flux_short_accel,
        "flux_long_acceleration": flux_long_accel,
    }
