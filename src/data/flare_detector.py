"""flare event detection from goes x-ray flux data."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from src.data.database import get_database
from src.data.schema import GOESXRayFlux

logger = logging.getLogger(__name__)

# flare classification thresholds (W/m²)
# based on noaa swpc classification
FLARE_THRESHOLDS = {
    "X": 1e-4,  # 100 microW/m²
    "M": 1e-5,  # 10 microW/m²
    "C": 1e-6,  # 1 microW/m²
    "B": 1e-7,  # 0.1 microW/m²
}


class FlareDetector:
    """detects solar flare events from goes x-ray flux data."""

    def __init__(self):
        self.db = get_database()

    def detect_flares_from_flux(
        self,
        flux_df: pd.DataFrame,
        min_class: str = "C",
    ) -> pd.DataFrame:
        """
        detect flare events from x-ray flux data.

        args:
            flux_df: dataframe with timestamp, flux_long columns
            min_class: minimum flare class to detect (B, C, M, X)

        returns:
            dataframe with detected flare events
        """
        if len(flux_df) == 0:
            return pd.DataFrame()

        # sort by timestamp
        flux_df = flux_df.sort_values("timestamp").reset_index(drop=True)

        # use flux_long for detection (1.0-8.0 angstrom band)
        if "flux_long" not in flux_df.columns:
            logger.warning("flux_long column not found")
            return pd.DataFrame()

        # remove NaN values
        flux_clean = flux_df[flux_df["flux_long"].notna()].copy()

        if len(flux_clean) == 0:
            return pd.DataFrame()

        # get threshold for minimum class
        min_threshold = FLARE_THRESHOLDS.get(min_class, FLARE_THRESHOLDS["C"])

        # find peaks above threshold
        flux_values = flux_clean["flux_long"].values
        timestamps = flux_clean["timestamp"].values

        # use scipy to find peaks
        # prominence = minimum height difference between peak and surrounding
        # width = minimum width of peak in samples
        peaks, properties = find_peaks(
            flux_values,
            height=min_threshold,
            prominence=min_threshold * 0.1,  # 10% of threshold as prominence
            width=1,  # at least 1 sample wide
            distance=10,  # at least 10 samples (about 5 minutes) between peaks
        )

        if len(peaks) == 0:
            return pd.DataFrame()

        flares = []

        for peak_idx in peaks:
            peak_time = timestamps[peak_idx]
            peak_flux = flux_values[peak_idx]

            # determine flare class
            flare_class, class_category, class_magnitude = self._classify_flare(peak_flux)

            # find flare start (when flux rises above background)
            start_time = self._find_flare_start(flux_clean, peak_idx, peak_flux)

            # find flare end (when flux returns to background)
            end_time = self._find_flare_end(flux_clean, peak_idx, peak_flux)

            # get associated region if available (from nearby solar region observations)
            active_region = self._get_nearby_region(peak_time)

            flares.append(
                {
                    "start_time": start_time,
                    "peak_time": peak_time,
                    "end_time": end_time,
                    "flare_class": flare_class,
                    "class_category": class_category,
                    "class_magnitude": class_magnitude,
                    "active_region": active_region,
                    "source": "auto_detected",
                    "verified": False,
                }
            )

        flares_df = pd.DataFrame(flares)

        # remove duplicates (same flare detected multiple times)
        flares_df = self._remove_duplicate_flares(flares_df)

        logger.info(f"detected {len(flares_df)} flare events from flux data")
        return flares_df

    def _classify_flare(self, flux_value: float) -> tuple:
        """
        classify flare based on peak flux value.

        args:
            flux_value: peak flux in W/m²

        returns:
            tuple of (flare_class, class_category, class_magnitude)
            e.g., ("X2.1", "X", 2.1)
        """
        if flux_value >= FLARE_THRESHOLDS["X"]:
            category = "X"
            magnitude = flux_value / FLARE_THRESHOLDS["X"]
        elif flux_value >= FLARE_THRESHOLDS["M"]:
            category = "M"
            magnitude = flux_value / FLARE_THRESHOLDS["M"]
        elif flux_value >= FLARE_THRESHOLDS["C"]:
            category = "C"
            magnitude = flux_value / FLARE_THRESHOLDS["C"]
        else:
            category = "B"
            magnitude = flux_value / FLARE_THRESHOLDS["B"]

        # round magnitude to 1 decimal
        magnitude = round(magnitude, 1)

        flare_class = f"{category}{magnitude}"

        return flare_class, category, magnitude

    def _find_flare_start(
        self,
        flux_df: pd.DataFrame,
        peak_idx: int,
        peak_flux: float,
    ) -> datetime:
        """
        find flare start time (when flux starts rising).

        args:
            flux_df: flux dataframe
            peak_idx: index of peak
            peak_flux: peak flux value

        returns:
            start time
        """
        # look back from peak
        lookback = min(60, peak_idx)  # max 60 samples back (about 5 hours)
        start_idx = max(0, peak_idx - lookback)

        flux_values = flux_df["flux_long"].values[start_idx : peak_idx + 1]
        timestamps = flux_df["timestamp"].values[start_idx : peak_idx + 1]

        # find when flux was at 10% of peak (start of rise)
        threshold = peak_flux * 0.1

        for i in range(len(flux_values) - 1, -1, -1):
            if flux_values[i] < threshold:
                ts = timestamps[i] if i < len(timestamps) else timestamps[0]
                # convert pandas Timestamp to datetime if needed
                if hasattr(ts, "to_pydatetime"):
                    return ts.to_pydatetime()
                return ts

        # fallback: use first timestamp in range
        ts = timestamps[0]
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
        return ts

    def _find_flare_end(
        self,
        flux_df: pd.DataFrame,
        peak_idx: int,
        peak_flux: float,
    ) -> datetime:
        """
        find flare end time (when flux returns to background).

        args:
            flux_df: flux dataframe
            peak_idx: index of peak
            peak_flux: peak flux value

        returns:
            end time
        """
        # look forward from peak
        lookahead = min(60, len(flux_df) - peak_idx - 1)
        end_idx = min(len(flux_df), peak_idx + lookahead)

        flux_values = flux_df["flux_long"].values[peak_idx:end_idx]
        timestamps = flux_df["timestamp"].values[peak_idx:end_idx]

        # find when flux drops to 10% of peak
        threshold = peak_flux * 0.1

        for i in range(len(flux_values)):
            if flux_values[i] < threshold:
                ts = timestamps[i] if i < len(timestamps) else timestamps[-1]
                # convert pandas Timestamp to datetime if needed
                if hasattr(ts, "to_pydatetime"):
                    return ts.to_pydatetime()
                return ts

        # fallback: use last timestamp in range
        ts = timestamps[-1] if len(timestamps) > 0 else flux_df["timestamp"].iloc[-1]
        if hasattr(ts, "to_pydatetime"):
            return ts.to_pydatetime()
        return ts

    def _get_nearby_region(self, timestamp: datetime) -> Optional[int]:
        """
        get active region number from nearby solar region observations.

        args:
            timestamp: flare timestamp

        returns:
            region number or None
        """
        try:
            from src.data.schema import SolarRegion

            # look for regions within 6 hours
            window = timedelta(hours=6)

            # ensure timestamp is datetime, not pandas Timestamp
            if hasattr(timestamp, "to_pydatetime"):
                timestamp = timestamp.to_pydatetime()
            elif not isinstance(timestamp, datetime):
                timestamp = pd.to_datetime(timestamp).to_pydatetime()

            with self.db.get_session() as session:
                region = (
                    session.query(SolarRegion)
                    .filter(
                        SolarRegion.timestamp >= timestamp - window,
                        SolarRegion.timestamp <= timestamp + window,
                    )
                    .order_by(SolarRegion.timestamp.desc())
                    .first()
                )

                if region:
                    return region.region_number

        except Exception as e:
            logger.debug(f"error finding nearby region: {e}")

        return None

    def _remove_duplicate_flares(self, flares_df: pd.DataFrame) -> pd.DataFrame:
        """
        remove duplicate flares (same peak time within 1 hour).

        args:
            flares_df: dataframe with flare events

        returns:
            dataframe with duplicates removed
        """
        if len(flares_df) == 0:
            return flares_df

        # sort by peak time
        flares_df = flares_df.sort_values("peak_time").reset_index(drop=True)

        # remove flares with peak times within 1 hour of each other (keep higher magnitude)
        unique_flares = []
        last_peak = None

        for _, flare in flares_df.iterrows():
            if last_peak is None:
                unique_flares.append(flare)
                last_peak = flare["peak_time"]
            else:
                time_diff = (flare["peak_time"] - last_peak).total_seconds() / 3600.0
                if time_diff > 1.0:  # more than 1 hour apart
                    unique_flares.append(flare)
                    last_peak = flare["peak_time"]
                else:
                    # same flare, keep the one with higher magnitude
                    if flare["class_magnitude"] > unique_flares[-1]["class_magnitude"]:
                        unique_flares[-1] = flare

        return pd.DataFrame(unique_flares)

    def detect_flares_from_database(
        self,
        start_date: datetime,
        end_date: datetime,
        min_class: str = "C",
    ) -> pd.DataFrame:
        """
        detect flares from flux data already in database.

        args:
            start_date: start of time range
            end_date: end of time range
            min_class: minimum flare class to detect

        returns:
            dataframe with detected flare events
        """
        try:
            with self.db.get_session() as session:
                query = (
                    session.query(GOESXRayFlux)
                    .filter(
                        GOESXRayFlux.timestamp >= start_date,
                        GOESXRayFlux.timestamp <= end_date,
                    )
                    .order_by(GOESXRayFlux.timestamp)
                )

                records = query.all()

                flux_df = pd.DataFrame(
                    [
                        {
                            "timestamp": r.timestamp,
                            "flux_long": r.flux_long,
                            "flux_short": r.flux_short,
                        }
                        for r in records
                    ]
                )

                if len(flux_df) == 0:
                    return pd.DataFrame()

                return self.detect_flares_from_flux(flux_df, min_class)

        except Exception as e:
            logger.error(f"error detecting flares from database: {e}")
            return pd.DataFrame()
