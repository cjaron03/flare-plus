# fmt: off
"""time-varying covariates for survival analysis - features that change over time."""

import logging
from typing import Optional, Dict, List, Any
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
from src.data.schema import GOESXRayFlux, SolarRegion, FlareEvent
from src.features.complexity import compute_mcintosh_complexity, compute_mount_wilson_complexity

logger = logging.getLogger(__name__)


def _normalize_timestamp(ts: datetime) -> datetime:
    """normalize timestamp to timezone-naive UTC for pandas compatibility."""
    if ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts

# time-varying covariate config
COVARIATE_CONFIG = CONFIG.get("time_varying_covariates", {})
LOOKBACK_WINDOWS = COVARIATE_CONFIG.get("lookback_windows", [1, 3, 6, 12, 24])  # hours


class TimeVaryingCovariateEngineer:
    """engineers time-varying covariates for survival analysis."""

    def __init__(self):
        self.db = get_database()
        self.lookback_windows = LOOKBACK_WINDOWS

    def bulk_load_data(
        self,
        min_timestamp: datetime,
        max_timestamp: datetime,
        max_lookback_hours: int = 24,
    ) -> Dict[str, pd.DataFrame]:
        """
        bulk load all data needed for a time range (used by batch processing).

        args:
            min_timestamp: earliest timestamp needed
            max_timestamp: latest timestamp needed
            max_lookback_hours: maximum lookback window

        returns:
            dict with flux, regions, magnetograms, flares dataframes
        """
        min_timestamp = _normalize_timestamp(min_timestamp)
        max_timestamp = _normalize_timestamp(max_timestamp)
        cutoff_time = min_timestamp - timedelta(hours=max_lookback_hours)
        data = {}

        try:
            with self.db.get_session() as session:
                # load flux data
                flux_records = (
                    session.query(GOESXRayFlux)
                    .filter(GOESXRayFlux.timestamp >= cutoff_time, GOESXRayFlux.timestamp <= max_timestamp)
                    .order_by(GOESXRayFlux.timestamp)
                    .all()
                )
                data["flux"] = pd.DataFrame(
                    [{"timestamp": _normalize_timestamp(r.timestamp), "flux_long": r.flux_long, "flux_short": r.flux_short} for r in flux_records]
                )

                # load solar regions
                regions_records = (
                    session.query(SolarRegion)
                    .filter(SolarRegion.timestamp >= cutoff_time, SolarRegion.timestamp <= max_timestamp)
                    .order_by(SolarRegion.timestamp)
                    .all()
                )
                data["regions"] = pd.DataFrame(
                    [
                        {
                            "timestamp": _normalize_timestamp(r.timestamp),
                            "region_number": r.region_number,
                            "mcintosh_class": r.mcintosh_class,
                            "mount_wilson_class": r.mount_wilson_class,
                            "area": r.area,
                        }
                        for r in regions_records
                    ]
                )

                # load flares
                flares_records = (
                    session.query(FlareEvent)
                    .filter(FlareEvent.peak_time >= cutoff_time, FlareEvent.peak_time <= max_timestamp)
                    .order_by(FlareEvent.peak_time)
                    .all()
                )
                data["flares"] = pd.DataFrame(
                    [
                        {
                            "peak_time": _normalize_timestamp(r.peak_time),
                            "start_time": _normalize_timestamp(r.start_time) if r.start_time else None,
                            "class_category": r.class_category,
                            "class_magnitude": r.class_magnitude,
                            "active_region": r.active_region,
                        }
                        for r in flares_records
                    ]
                )

                return data
        except Exception as e:
            logger.error(f"error bulk loading data: {e}")
            return {"flux": pd.DataFrame(), "regions": pd.DataFrame(), "flares": pd.DataFrame()}

    def compute_recent_flux_metrics(
        self,
        timestamp: datetime,
        lookback_hours: int = 24,
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """
        compute recent flux metrics (mean, max, trend) over lookback window.

        args:
            timestamp: current timestamp
            lookback_hours: hours to look back
            preloaded_data: optional pre-loaded data dict to avoid DB queries

        returns:
            dict with flux metrics
        """
        timestamp = _normalize_timestamp(timestamp)
        cutoff_time = timestamp - timedelta(hours=lookback_hours)

        try:
            if preloaded_data is not None and "flux" in preloaded_data:
                # use preloaded data
                flux_df = preloaded_data["flux"][
                    (preloaded_data["flux"]["timestamp"] >= cutoff_time) &
                    (preloaded_data["flux"]["timestamp"] <= timestamp)
                ].copy()
            else:
                # load from database
                with self.db.get_session() as session:
                    flux_query = (
                        session.query(GOESXRayFlux)
                        .filter(
                            GOESXRayFlux.timestamp >= cutoff_time,
                            GOESXRayFlux.timestamp <= timestamp,
                        )
                        .order_by(GOESXRayFlux.timestamp)
                    )
                    flux_records = flux_query.all()
                    flux_df = pd.DataFrame(
                        [
                            {
                                "timestamp": r.timestamp,
                                "flux_long": r.flux_long if r.flux_long is not None else 0.0,
                                "flux_short": r.flux_short if r.flux_short is not None else 0.0,
                            }
                            for r in flux_records
                        ]
                    )

            if len(flux_df) == 0:
                return {
                    f"flux_long_mean_{lookback_hours}h": 0.0,
                    f"flux_long_max_{lookback_hours}h": 0.0,
                    f"flux_long_trend_{lookback_hours}h": 0.0,
                    f"flux_short_mean_{lookback_hours}h": 0.0,
                    f"flux_short_max_{lookback_hours}h": 0.0,
                    f"flux_short_trend_{lookback_hours}h": 0.0,
                }

            # compute metrics (use actual values, not default to 0)
            flux_long_mean = flux_df["flux_long"].mean()
            flux_long_max = flux_df["flux_long"].max()
            flux_short_mean = flux_df["flux_short"].mean()
            flux_short_max = flux_df["flux_short"].max()

            metrics = {
                f"flux_long_mean_{lookback_hours}h": float(flux_long_mean) if pd.notna(flux_long_mean) else 0.0,
                f"flux_long_max_{lookback_hours}h": float(flux_long_max) if pd.notna(flux_long_max) else 0.0,
                f"flux_short_mean_{lookback_hours}h": float(flux_short_mean) if pd.notna(flux_short_mean) else 0.0,
                f"flux_short_max_{lookback_hours}h": float(flux_short_max) if pd.notna(flux_short_max) else 0.0,
            }

            # compute trend (slope of linear regression)
            if len(flux_df) > 1:
                try:
                    time_numeric = (flux_df["timestamp"] - flux_df["timestamp"].min()).dt.total_seconds() / 3600.0
                    flux_long_trend = np.polyfit(time_numeric, flux_df["flux_long"], 1)[0]
                    flux_short_trend = np.polyfit(time_numeric, flux_df["flux_short"], 1)[0]
                    metrics[f"flux_long_trend_{lookback_hours}h"] = float(flux_long_trend)
                    metrics[f"flux_short_trend_{lookback_hours}h"] = float(flux_short_trend)
                except Exception as e:
                    logger.debug(f"error computing trends for {lookback_hours}h: {e}")
                    metrics[f"flux_long_trend_{lookback_hours}h"] = 0.0
                    metrics[f"flux_short_trend_{lookback_hours}h"] = 0.0
            else:
                metrics[f"flux_long_trend_{lookback_hours}h"] = 0.0
                metrics[f"flux_short_trend_{lookback_hours}h"] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"error computing flux metrics for {timestamp}: {e}")
            return {
                f"flux_long_mean_{lookback_hours}h": 0.0,
                f"flux_long_max_{lookback_hours}h": 0.0,
                f"flux_long_trend_{lookback_hours}h": 0.0,
                f"flux_short_mean_{lookback_hours}h": 0.0,
                f"flux_short_max_{lookback_hours}h": 0.0,
                f"flux_short_trend_{lookback_hours}h": 0.0,
            }

    def compute_recent_region_complexity(
        self,
        timestamp: datetime,
        lookback_hours: int = 24,
        region_number: Optional[int] = None,
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """
        compute recent region complexity metrics.

        args:
            timestamp: current timestamp
            lookback_hours: hours to look back
            region_number: optional region number
            preloaded_data: optional pre-loaded data dict to avoid DB queries

        returns:
            dict with complexity metrics
        """
        timestamp = _normalize_timestamp(timestamp)
        cutoff_time = timestamp - timedelta(hours=lookback_hours)

        try:
            if preloaded_data is not None and "regions" in preloaded_data:
                # use preloaded data
                regions_df = preloaded_data["regions"][
                    (preloaded_data["regions"]["timestamp"] >= cutoff_time) &
                    (preloaded_data["regions"]["timestamp"] <= timestamp)
                ].copy()

                if region_number is not None:
                    regions_df = regions_df[regions_df["region_number"] == region_number].copy()
            else:
                # load from database
                with self.db.get_session() as session:
                    query = (
                        session.query(SolarRegion)
                        .filter(
                            SolarRegion.timestamp >= cutoff_time,
                            SolarRegion.timestamp <= timestamp,
                        )
                    )

                    if region_number is not None:
                        query = query.filter(SolarRegion.region_number == region_number)

                    region_records = query.order_by(SolarRegion.timestamp).all()

                    if len(region_records) == 0:
                        return {
                            f"max_complexity_{lookback_hours}h": 0.0,
                            f"avg_area_{lookback_hours}h": 0.0,
                            f"max_mcintosh_{lookback_hours}h": 0.0,
                            f"max_mount_wilson_{lookback_hours}h": 0.0,
                        }

                    regions_df = pd.DataFrame(
                        [
                            {
                                "timestamp": r.timestamp,
                                "region_number": r.region_number,
                                "mcintosh_class": r.mcintosh_class,
                                "mount_wilson_class": r.mount_wilson_class,
                                "area": r.area if r.area is not None else 0,
                            }
                            for r in region_records
                        ]
                    )

            if len(regions_df) == 0:
                return {
                    f"max_complexity_{lookback_hours}h": 0.0,
                    f"avg_area_{lookback_hours}h": 0.0,
                    f"max_mcintosh_{lookback_hours}h": 0.0,
                    f"max_mount_wilson_{lookback_hours}h": 0.0,
                }

            # compute complexity scores
            mcintosh_scores = []
            mount_wilson_scores = []

            for _, row in regions_df.iterrows():
                if pd.notna(row["mcintosh_class"]):
                    metrics_dict = compute_mcintosh_complexity(row["mcintosh_class"])
                    score = metrics_dict.get("mcintosh_complexity_score", 0.0)
                    mcintosh_scores.append(score)

                if pd.notna(row["mount_wilson_class"]):
                    metrics_dict = compute_mount_wilson_complexity(row["mount_wilson_class"])
                    score = metrics_dict.get("mount_wilson_complexity_score", 0.0)
                    mount_wilson_scores.append(score)

            metrics = {
                f"max_complexity_{lookback_hours}h": float(max(mcintosh_scores + mount_wilson_scores)) if len(mcintosh_scores + mount_wilson_scores) > 0 else 0.0,
                f"avg_area_{lookback_hours}h": float(regions_df["area"].mean()) if len(regions_df) > 0 else 0.0,
                f"max_mcintosh_{lookback_hours}h": float(max(mcintosh_scores)) if len(mcintosh_scores) > 0 else 0.0,
                f"max_mount_wilson_{lookback_hours}h": float(max(mount_wilson_scores)) if len(mount_wilson_scores) > 0 else 0.0,
            }

            return metrics

        except Exception as e:
            logger.error(f"error computing region complexity for {timestamp}: {e}")
            return {
                f"max_complexity_{lookback_hours}h": 0.0,
                f"avg_area_{lookback_hours}h": 0.0,
                f"max_mcintosh_{lookback_hours}h": 0.0,
                f"max_mount_wilson_{lookback_hours}h": 0.0,
            }

    def compute_recent_flare_history(
        self,
        timestamp: datetime,
        lookback_hours: int = 24,
        region_number: Optional[int] = None,
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """
        compute recent flare history (counts, recency).

        args:
            timestamp: current timestamp
            lookback_hours: hours to look back
            region_number: optional region number
            preloaded_data: optional pre-loaded data dict to avoid DB queries

        returns:
            dict with flare history metrics
        """
        timestamp = _normalize_timestamp(timestamp)
        cutoff_time = timestamp - timedelta(hours=lookback_hours)

        try:
            if preloaded_data is not None and "flares" in preloaded_data:
                # use preloaded data
                flares_df = preloaded_data["flares"][
                    (preloaded_data["flares"]["peak_time"] >= cutoff_time) &
                    (preloaded_data["flares"]["peak_time"] < timestamp)
                ].copy()

                if region_number is not None:
                    flares_df = flares_df[flares_df["active_region"] == region_number].copy()
            else:
                # load from database
                with self.db.get_session() as session:
                    query = (
                        session.query(FlareEvent)
                        .filter(
                            FlareEvent.start_time >= cutoff_time,
                            FlareEvent.start_time < timestamp,
                        )
                    )

                    if region_number is not None:
                        query = query.filter(FlareEvent.active_region == region_number)

                    flare_records = query.order_by(FlareEvent.start_time.desc()).all()

                    if len(flare_records) == 0:
                        return {
                            f"flare_count_{lookback_hours}h": 0.0,
                            f"max_flare_class_{lookback_hours}h": 0.0,
                            f"hours_since_last_flare_{lookback_hours}h": float(lookback_hours),
                        }

                    flares_df = pd.DataFrame(
                        [
                            {
                                "start_time": r.start_time,
                                "peak_time": r.peak_time,
                                "class_category": r.class_category,
                                "class_magnitude": r.class_magnitude,
                                "active_region": r.active_region,
                            }
                            for r in flare_records
                        ]
                    )

            if len(flares_df) == 0:
                return {
                    f"flare_count_{lookback_hours}h": 0.0,
                    f"max_flare_class_{lookback_hours}h": 0.0,
                    f"hours_since_last_flare_{lookback_hours}h": float(lookback_hours),
                }

            # class hierarchy for max class
            class_hierarchy = {"B": 1, "C": 2, "M": 3, "X": 4}
            max_class_val = max([class_hierarchy.get(c, 0) for c in flares_df["class_category"].unique()])

            # hours since last flare - use peak_time if available, else start_time
            time_col = "peak_time" if "peak_time" in flares_df.columns else "start_time"
            last_flare_time = flares_df[time_col].max()
            hours_since = (timestamp - last_flare_time).total_seconds() / 3600.0

            metrics = {
                f"flare_count_{lookback_hours}h": float(len(flares_df)),
                f"max_flare_class_{lookback_hours}h": float(max_class_val),
                f"hours_since_last_flare_{lookback_hours}h": float(hours_since),
            }

            return metrics

        except Exception as e:
            logger.error(f"error computing flare history for {timestamp}: {e}")
            return {
                f"flare_count_{lookback_hours}h": 0.0,
                f"max_flare_class_{lookback_hours}h": 0.0,
                f"hours_since_last_flare_{lookback_hours}h": float(lookback_hours),
            }

    def compute_time_varying_covariates(
        self,
        timestamp: datetime,
        region_number: Optional[int] = None,
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        compute all time-varying covariates for a timestamp.

        args:
            timestamp: observation timestamp
            region_number: optional region number
            preloaded_data: optional pre-loaded data dict to avoid DB queries

        returns:
            dataframe with time-varying covariates (single row)
        """
        covariates: Dict[str, Any] = {"timestamp": timestamp}

        if region_number is not None:
            covariates["region_number"] = region_number

        # compute covariates for each lookback window
        for lookback in self.lookback_windows:
            # flux metrics
            flux_metrics = self.compute_recent_flux_metrics(timestamp, lookback, preloaded_data)
            covariates.update(flux_metrics)

            # region complexity
            complexity_metrics = self.compute_recent_region_complexity(timestamp, lookback, region_number, preloaded_data)
            covariates.update(complexity_metrics)

            # flare history
            flare_metrics = self.compute_recent_flare_history(timestamp, lookback, region_number, preloaded_data)
            covariates.update(flare_metrics)

        return pd.DataFrame([covariates])

    def compute_time_varying_covariates_batch(
        self,
        timestamps: List[datetime],
        region_number: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        compute time-varying covariates for multiple timestamps.

        args:
            timestamps: list of observation timestamps
            region_number: optional region number
            show_progress: whether to show progress bar

        returns:
            dataframe with time-varying covariates
        """
        if not timestamps:
            return pd.DataFrame()

        # calculate overall time window and bulk load data
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)
        max_lookback = max(self.lookback_windows) if self.lookback_windows else 24

        logger.info(f"bulk loading data for {len(timestamps)} timestamps")
        preloaded_data = self.bulk_load_data(min_timestamp, max_timestamp, max_lookback)

        covariates_list = []

        iterable = timestamps
        if show_progress and HAS_TQDM and len(timestamps) > 10:
            iterable = tqdm(timestamps, desc="computing covariates", unit="timestamp")

        for timestamp in iterable:
            try:
                cov_df = self.compute_time_varying_covariates(timestamp, region_number, preloaded_data)
                covariates_list.append(cov_df)
            except Exception as e:
                logger.error(f"error computing covariates for {timestamp}: {e}")
                continue

        if len(covariates_list) == 0:
            return pd.DataFrame()

        return pd.concat(covariates_list, ignore_index=True)
# fmt: on
