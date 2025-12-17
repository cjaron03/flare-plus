# fmt: off
"""time-to-event labeling for survival analysis - time until next X-class flare."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.config import CONFIG
from src.data.database import get_database
from src.data.schema import FlareEvent

logger = logging.getLogger(__name__)

# survival analysis config
SURVIVAL_CONFIG = CONFIG.get("survival", {})
TARGET_FLARE_CLASS = SURVIVAL_CONFIG.get("target_flare_class", "X")  # focus on X-class flares
MAX_TIME_HOURS = SURVIVAL_CONFIG.get("max_time_hours", 168)  # 7 days max observation window
TIME_BUCKETS = SURVIVAL_CONFIG.get("time_buckets", [0, 6, 12, 24, 48, 72, 96, 120, 168])  # hours


def _normalize_timestamp(ts: datetime) -> datetime:
    """ensure timestamps are timezone-naive for pandas compatibility."""
    if ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts


class SurvivalLabeler:
    """creates time-to-event labels for survival analysis."""

    def __init__(self, target_flare_class: str = TARGET_FLARE_CLASS, max_time_hours: int = MAX_TIME_HOURS):
        """
        initialize survival labeler.

        args:
            target_flare_class: flare class to predict (X, M, C, etc.)
            max_time_hours: maximum observation time (censoring window)
        """
        self.db = get_database()
        self.target_flare_class = target_flare_class
        self.max_time_hours = max_time_hours
        self.time_buckets = TIME_BUCKETS

    def _load_target_flares(
        self,
        timestamps: List[datetime],
        region_number: Optional[int] = None,
    ) -> pd.DataFrame:
        """bulk load flares within the observation window for given timestamps."""
        if not timestamps:
            return pd.DataFrame()

        normalized_timestamps = [_normalize_timestamp(ts) for ts in timestamps]
        min_timestamp = min(normalized_timestamps)
        max_timestamp = max(normalized_timestamps)
        range_end = max_timestamp + timedelta(hours=self.max_time_hours)

        try:
            with self.db.get_session() as session:
                query = (
                    session.query(FlareEvent)
                    .filter(
                        FlareEvent.start_time > min_timestamp,
                        FlareEvent.start_time <= range_end,
                        FlareEvent.class_category == self.target_flare_class,
                    )
                    .order_by(FlareEvent.start_time)
                )
                if region_number is not None:
                    query = query.filter(FlareEvent.active_region == region_number)

                records = query.all()

                # extract attributes while session is still open to avoid DetachedInstanceError
                flares_df = pd.DataFrame(
                    [
                        {
                            "start_time": _normalize_timestamp(r.start_time),
                            "active_region": r.active_region,
                        }
                        for r in records
                    ]
                )
        except Exception as e:
            logger.error(f"error bulk loading target flares: {e}")
            return pd.DataFrame()

        if len(flares_df) == 0:
            return pd.DataFrame()

        flares_df = flares_df.sort_values("start_time").reset_index(drop=True)
        return flares_df

    def find_next_flare_time(
        self,
        timestamp: datetime,
        region_number: Optional[int] = None,
        preloaded_flares: Optional[pd.DataFrame] = None,
    ) -> Optional[datetime]:
        """
        find the time of the next target flare class after the given timestamp.

        args:
            timestamp: reference timestamp
            region_number: optional region number to filter by

        returns:
            datetime of next flare, or None if none found within max_time_hours
        """
        try:
            if preloaded_flares is not None and len(preloaded_flares) > 0:
                flares_df = preloaded_flares
                mask = flares_df["start_time"] > timestamp
                if region_number is not None and "active_region" in flares_df.columns:
                    mask &= flares_df["active_region"] == region_number

                candidates = flares_df.loc[mask]
                if len(candidates) == 0:
                    return None

                next_time = candidates["start_time"].iloc[0]
                time_delta = next_time - timestamp
                if time_delta.total_seconds() / 3600.0 > self.max_time_hours:
                    return None
                return next_time

            with self.db.get_session() as session:  # type: ignore[assignment]
                query = (
                    session.query(FlareEvent)
                    .filter(
                        FlareEvent.start_time > timestamp,
                        FlareEvent.class_category == self.target_flare_class,
                    )
                    .order_by(FlareEvent.start_time)
                    .limit(1)
                )

                if region_number is not None:
                    query = query.filter(FlareEvent.active_region == region_number)

                flare = query.first()

                if flare is None:
                    return None

                # check if within max observation window
                time_delta = flare.start_time - timestamp
                if time_delta.total_seconds() / 3600.0 > self.max_time_hours:
                    return None

                return flare.start_time

        except Exception as e:
            logger.error(f"error finding next flare for {timestamp}: {e}")
            return None

    def create_survival_label(
        self,
        timestamp: datetime,
        region_number: Optional[int] = None,
        preloaded_flares: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        create survival label for a single timestamp.

        returns dict with:
            - duration: time to event in hours (or max_time if censored)
            - event: 1 if event occurred, 0 if censored
            - event_time: datetime of event (None if censored)

        args:
            timestamp: observation timestamp
            region_number: optional region number

        returns:
            dict with survival label
        """
        next_flare_time = self.find_next_flare_time(
            timestamp,
            region_number,
            preloaded_flares=preloaded_flares,
        )

        if next_flare_time is None:
            # censored observation - no event within max_time_hours
            return {
                "timestamp": timestamp,
                "duration": float(self.max_time_hours),
                "event": 0,
                "event_time": None,
                "region_number": region_number,
            }

        # event occurred
        duration_hours = (next_flare_time - timestamp).total_seconds() / 3600.0

        return {
            "timestamp": timestamp,
            "duration": float(duration_hours),
            "event": 1,
            "event_time": next_flare_time,
            "region_number": region_number,
        }

    def create_survival_labels(
        self,
        timestamps: List[datetime],
        region_number: Optional[int] = None,
        show_progress: bool = True,
        preloaded_flares: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        create survival labels for multiple timestamps.

        args:
            timestamps: list of observation timestamps
            region_number: optional region number
            show_progress: whether to show progress bar

        returns:
            dataframe with survival labels (duration, event, event_time)
        """
        if preloaded_flares is None:
            preloaded_flares = self._load_target_flares(timestamps, region_number)

        labels_list = []

        iterable = timestamps
        if show_progress and HAS_TQDM and len(timestamps) > 10:
            iterable = tqdm(timestamps, desc="creating labels", unit="timestamp")

        for timestamp in iterable:
            try:
                label_dict = self.create_survival_label(
                    timestamp,
                    region_number,
                    preloaded_flares=preloaded_flares,
                )
                labels_list.append(label_dict)
            except Exception as e:
                logger.error(f"error creating survival label for {timestamp}: {e}")
                continue

        if len(labels_list) == 0:
            return pd.DataFrame()

        labels_df = pd.DataFrame(labels_list)

        return labels_df

    def compute_probability_distribution(
        self,
        survival_function: np.ndarray,
        time_points: np.ndarray,
        time_buckets: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        compute probability distribution over time buckets.

        probability of event in bucket [t_i, t_{i+1}] = S(t_i) - S(t_{i+1})

        args:
            survival_function: survival probabilities at time_points
            time_points: time points in hours (must match survival_function)
            time_buckets: list of bucket boundaries in hours

        returns:
            dict mapping bucket labels to probabilities
        """
        if time_buckets is None:
            time_buckets = self.time_buckets

        # ensure time_points are sorted
        sort_idx = np.argsort(time_points)
        time_points_sorted = time_points[sort_idx]
        survival_sorted = survival_function[sort_idx]

        prob_dist = {}

        # debug: log survival function range if all probabilities are zero
        if len(survival_sorted) > 0:
            logger.debug(f"survival function range: [{survival_sorted.min():.4f}, {survival_sorted.max():.4f}]")
            logger.debug(f"time_points range: [{time_points_sorted.min():.2f}h, {time_points_sorted.max():.2f}h]")

        for i in range(len(time_buckets) - 1):
            bucket_start = time_buckets[i]
            bucket_end = time_buckets[i + 1]
            # format bucket label
            bucket_label = f"{int(bucket_start)}h-{int(bucket_end)}h"

            # find survival probabilities at bucket boundaries
            # interpolate if needed (extrapolate with constant value if outside range)
            if len(survival_sorted) == 0:
                # no survival data, assume no events
                prob = 0.0
            else:
                # get last survival value for extrapolation (survival shouldn't jump to 0)
                last_survival = survival_sorted[-1] if len(survival_sorted) > 0 else 1.0

                s_start = np.interp(
                    bucket_start,
                    time_points_sorted,
                    survival_sorted,
                    left=survival_sorted[0] if len(survival_sorted) > 0 else 1.0,
                    right=last_survival
                )
                s_end = np.interp(
                    bucket_end,
                    time_points_sorted,
                    survival_sorted,
                    left=survival_sorted[0] if len(survival_sorted) > 0 else 1.0,
                    right=last_survival
                )

                # probability of event in this bucket = survival at start - survival at end
                # survival function S(t) = P(T > t), so P(event in [t1, t2]) = S(t1) - S(t2)
                prob = float(s_start - s_end)

                # handle edge cases: if survival is constant or increases
                # (shouldn't happen but numerical errors)
                if prob < 0:
                    logger.debug(
                        f"negative probability for {bucket_label}: {prob:.6f} "
                        f"(s_start={s_start:.4f}, s_end={s_end:.4f}), clamping to 0"
                    )
                    prob = 0.0

            prob_dist[bucket_label] = max(0.0, min(1.0, prob))

        return prob_dist

    def create_labels_from_features(
        self,
        features_df: pd.DataFrame,
        region_number: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        create survival labels for feature dataframe.

        args:
            features_df: dataframe with features (must have 'timestamp' column)
            region_number: optional region number

        returns:
            dataframe with features and survival labels
        """
        if "timestamp" not in features_df.columns:
            raise ValueError("features dataframe must have 'timestamp' column")

        timestamps = features_df["timestamp"].tolist()
        labels_df = self.create_survival_labels(timestamps, region_number)

        # merge labels with features
        result_df = features_df.merge(labels_df, on="timestamp", how="left")

        return result_df
# fmt: on
