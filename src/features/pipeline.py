"""main feature engineering pipeline."""

import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data.database import get_database
from src.data.schema import GOESXRayFlux, SolarRegion, SolarMagnetogram, FlareEvent

from src.features.complexity import (
    compute_mcintosh_complexity,
    compute_mount_wilson_complexity,
    compute_magnetic_complexity_score,
)
from src.features.flux_trends import (
    compute_flux_trends,
    compute_flux_rate_of_change,
)
from src.features.rolling_stats import (
    compute_rolling_statistics,
    compute_recency_weighted_flare_counts,
)
from src.features.normalization import (
    normalize_features,
    standardize_features,
    handle_missing_data,
    flag_missing_data_paths,
)

logger = logging.getLogger(__name__)


def _normalize_timestamp(ts: datetime) -> datetime:
    """normalize timestamp to timezone-naive UTC for pandas compatibility."""
    if ts.tzinfo is not None:
        return ts.replace(tzinfo=None)
    return ts


# feature engineering config
FEATURE_CONFIG = CONFIG.get("feature_engineering", {})
ROLLING_WINDOWS = FEATURE_CONFIG.get("rolling_windows", [6, 12, 24])
FLARE_CLASSES = FEATURE_CONFIG.get("flare_classes", ["B", "C", "M", "X"])


class FeatureEngineer:
    """main feature engineering pipeline."""

    def __init__(self):
        self.db = get_database()
        self.rolling_windows = ROLLING_WINDOWS
        self.flare_classes = FLARE_CLASSES

    def load_data(
        self,
        timestamp: datetime,
        lookback_hours: int = 48,
    ) -> Dict[str, pd.DataFrame]:
        """
        load raw data from database for feature engineering.

        args:
            timestamp: current timestamp
            lookback_hours: hours to look back for data

        returns:
            dict with dataframes: flux, regions, magnetograms, flares
        """
        # normalize timestamp to timezone-naive UTC for pandas compatibility
        timestamp = _normalize_timestamp(timestamp)
        cutoff_time = timestamp - timedelta(hours=lookback_hours)

        data = {}

        try:
            with self.db.get_session() as session:
                # load flux data
                flux_query = (
                    session.query(GOESXRayFlux)
                    .filter(GOESXRayFlux.timestamp >= cutoff_time, GOESXRayFlux.timestamp <= timestamp)
                    .order_by(GOESXRayFlux.timestamp)
                )
                flux_records = flux_query.all()
                data["flux"] = pd.DataFrame(
                    [
                        {
                            "timestamp": _normalize_timestamp(r.timestamp),
                            "flux_short": r.flux_short,
                            "flux_long": r.flux_long,
                            "satellite": r.satellite,
                        }
                        for r in flux_records
                    ]
                )
                # ensure timestamp column is timezone-naive
                if len(data["flux"]) > 0 and "timestamp" in data["flux"].columns:
                    if hasattr(data["flux"]["timestamp"].dtype, 'tz') and data["flux"]["timestamp"].dtype.tz is not None:
                        data["flux"]["timestamp"] = data["flux"]["timestamp"].dt.tz_localize(None)
                    else:
                        # normalize individual values if needed
                        data["flux"]["timestamp"] = data["flux"]["timestamp"].apply(_normalize_timestamp)

                # load solar regions
                regions_query = (
                    session.query(SolarRegion)
                    .filter(SolarRegion.timestamp >= cutoff_time, SolarRegion.timestamp <= timestamp)
                    .order_by(SolarRegion.timestamp)
                )
                regions_records = regions_query.all()
                data["regions"] = pd.DataFrame(
                    [
                        {
                            "timestamp": _normalize_timestamp(r.timestamp),
                            "region_number": r.region_number,
                            "latitude": r.latitude,
                            "longitude": r.longitude,
                            "mcintosh_class": r.mcintosh_class,
                            "mount_wilson_class": r.mount_wilson_class,
                            "area": r.area,
                            "num_sunspots": r.num_sunspots,
                            "magnetic_type": r.magnetic_type,
                        }
                        for r in regions_records
                    ]
                )

                # load magnetograms
                magnetogram_query = (
                    session.query(SolarMagnetogram)
                    .filter(SolarMagnetogram.timestamp >= cutoff_time, SolarMagnetogram.timestamp <= timestamp)
                    .order_by(SolarMagnetogram.timestamp)
                )
                magnetogram_records = magnetogram_query.all()
                data["magnetograms"] = pd.DataFrame(
                    [
                        {
                            "timestamp": _normalize_timestamp(r.timestamp),
                            "region_number": r.region_number,
                            "magnetic_complexity": r.magnetic_complexity,
                            "magnetic_field_polarity": r.magnetic_field_polarity,
                            "latitude": r.latitude,
                            "longitude": r.longitude,
                        }
                        for r in magnetogram_records
                    ]
                )

                # load flare events
                flares_query = (
                    session.query(FlareEvent)
                    .filter(FlareEvent.peak_time >= cutoff_time, FlareEvent.peak_time <= timestamp)
                    .order_by(FlareEvent.peak_time)
                )
                flares_records = flares_query.all()
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

        except Exception as e:
            logger.error(f"error loading data: {e}")
            return {
                "flux": pd.DataFrame(),
                "regions": pd.DataFrame(),
                "magnetograms": pd.DataFrame(),
                "flares": pd.DataFrame(),
            }

        return data

    def compute_features(
        self,
        timestamp: datetime,
        region_number: Optional[int] = None,
        normalize: bool = False,
        standardize: bool = False,
        handle_missing: bool = True,
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        compute all features for a given timestamp.

        args:
            timestamp: timestamp to compute features for
            region_number: optional region number to filter by
            normalize: whether to normalize features
            standardize: whether to standardize features
            handle_missing: whether to handle missing data
            preloaded_data: optional pre-loaded data dict to avoid DB queries

        returns:
            dataframe with computed features
        """
        # normalize timestamp to timezone-naive UTC for pandas compatibility
        timestamp = _normalize_timestamp(timestamp)
        
        # load data (or use preloaded)
        if preloaded_data is not None:
            # filter preloaded data to the relevant time window for this timestamp
            lookback_hours = max(self.rolling_windows) + 24
            cutoff_time = timestamp - timedelta(hours=lookback_hours)
            data = {
                "flux": (
                    preloaded_data["flux"][
                        (preloaded_data["flux"]["timestamp"] >= cutoff_time)
                        & (preloaded_data["flux"]["timestamp"] <= timestamp)
                    ].copy()
                    if len(preloaded_data["flux"]) > 0
                    else pd.DataFrame()
                ),
                "regions": (
                    preloaded_data["regions"][
                        (preloaded_data["regions"]["timestamp"] >= cutoff_time)
                        & (preloaded_data["regions"]["timestamp"] <= timestamp)
                    ].copy()
                    if len(preloaded_data["regions"]) > 0
                    else pd.DataFrame()
                ),
                "magnetograms": (
                    preloaded_data["magnetograms"][
                        (preloaded_data["magnetograms"]["timestamp"] >= cutoff_time)
                        & (preloaded_data["magnetograms"]["timestamp"] <= timestamp)
                    ].copy()
                    if len(preloaded_data["magnetograms"]) > 0
                    else pd.DataFrame()
                ),
                "flares": (
                    preloaded_data["flares"][
                        (preloaded_data["flares"]["peak_time"] >= cutoff_time)
                        & (preloaded_data["flares"]["peak_time"] <= timestamp)
                    ].copy()
                    if len(preloaded_data["flares"]) > 0
                    else pd.DataFrame()
                ),
            }
        else:
            data = self.load_data(timestamp, lookback_hours=max(self.rolling_windows) + 24)

        features = {}

        # base features
        features["timestamp"] = timestamp
        if region_number is not None:
            features["region_number"] = region_number

        # 1. complexity features from solar regions
        if len(data["regions"]) > 0:
            if region_number is not None:
                region_data = data["regions"][data["regions"]["region_number"] == region_number]
            else:
                # use most recent region
                region_data = (
                    data["regions"].sort_values("timestamp").iloc[-1:] if len(data["regions"]) > 0 else pd.DataFrame()
                )

            if len(region_data) > 0:
                region = region_data.iloc[0]

                # mcintosh complexity
                mcintosh_features = compute_mcintosh_complexity(region.get("mcintosh_class"))
                features.update(mcintosh_features)

                # mount wilson complexity
                mount_wilson_features = compute_mount_wilson_complexity(region.get("mount_wilson_class"))
                features.update(mount_wilson_features)

                # magnetic complexity
                magnetic_features = compute_magnetic_complexity_score(
                    region.get("magnetic_type"), region.get("magnetic_type")
                )
                features.update(magnetic_features)

                # region characteristics
                features["region_area"] = region.get("area")
                features["region_num_sunspots"] = region.get("num_sunspots")
                features["region_latitude"] = region.get("latitude")
                features["region_longitude"] = region.get("longitude")

        # 2. flux trend features
        if len(data["flux"]) > 0:
            flux_features = compute_flux_trends(data["flux"], timestamp, lookback_hours=24, window_hours=6)
            features.update(flux_features)

            flux_rate_features = compute_flux_rate_of_change(data["flux"], timestamp, lookback_hours=12)
            features.update(flux_rate_features)

            # rolling statistics for flux
            for window in self.rolling_windows:
                flux_rolling = compute_rolling_statistics(
                    data["flux"], timestamp, "flux_short", [window], ["mean", "max", "std"]
                )
                features.update(flux_rolling)

                flux_rolling_long = compute_rolling_statistics(
                    data["flux"], timestamp, "flux_long", [window], ["mean", "max", "std"]
                )
                features.update(flux_rolling_long)

        # 3. recency-weighted flare counts
        if len(data["flares"]) > 0:
            flare_counts = compute_recency_weighted_flare_counts(
                data["flares"], timestamp, self.flare_classes, self.rolling_windows
            )
            features.update(flare_counts)

        # convert to dataframe
        # replace None with np.nan to ensure proper numeric dtype inference
        features_clean = {k: (v if v is not None else np.nan) for k, v in features.items()}
        features_df = pd.DataFrame([features_clean])

        # ensure numeric columns are properly typed (pandas sometimes infers object dtype)
        for col in features_df.columns:
            if col != "timestamp":  # skip timestamp column
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors='ignore')
                except Exception:
                    pass  # keep original if conversion fails

        # handle missing data
        if handle_missing:
            missing_info = flag_missing_data_paths(features_df)
            if missing_info["missing_count"] > 0:
                logger.info(f"missing data detected: {missing_info['missing_fraction']*100:.1f}%")
                features_df = handle_missing_data(features_df, strategy="mean")

        # normalize
        if normalize:
            features_df = normalize_features(features_df)

        # standardize
        if standardize:
            features_df = standardize_features(features_df)

        return features_df

    def compute_features_batch(
        self,
        timestamps: List[datetime],
        region_number: Optional[int] = None,
        normalize: bool = False,
        standardize: bool = False,
        handle_missing: bool = True,
    ) -> pd.DataFrame:
        """
        compute features for multiple timestamps.

        args:
            timestamps: list of timestamps to compute features for
            region_number: optional region number to filter by
            normalize: whether to normalize features
            standardize: whether to standardize features
            handle_missing: whether to handle missing data

        returns:
            dataframe with computed features for all timestamps
        """
        if not timestamps:
            return pd.DataFrame()

        # calculate overall time window needed for bulk load
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)
        lookback_hours = max(self.rolling_windows) + 24

        # load all data once for the entire time range
        logger.info(f"bulk loading data for {len(timestamps)} timestamps")
        preloaded_data = self.load_data(
            max_timestamp, lookback_hours=lookback_hours + int((max_timestamp - min_timestamp).total_seconds() / 3600)
        )

        all_features = []

        for timestamp in timestamps:
            try:
                features = self.compute_features(
                    timestamp,
                    region_number=region_number,
                    normalize=False,  # normalize/standardize after combining
                    standardize=False,
                    handle_missing=handle_missing,
                    preloaded_data=preloaded_data,  # pass preloaded data
                )
                all_features.append(features)
            except Exception as e:
                logger.error(f"error computing features for {timestamp}: {e}")
                continue

        if len(all_features) == 0:
            return pd.DataFrame()

        # combine all features
        features_df = pd.concat(all_features, ignore_index=True)

        # normalize/standardize on combined data
        if normalize:
            features_df = normalize_features(features_df)

        if standardize:
            features_df = standardize_features(features_df)

        return features_df
