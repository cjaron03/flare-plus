# fmt: off
"""label creation for supervised learning - next-24h and next-48h flare classes."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.config import CONFIG
from src.data.database import get_database
from src.data.schema import FlareEvent

logger = logging.getLogger(__name__)

# model config
MODEL_CONFIG = CONFIG.get("model", {})
TARGET_WINDOWS = MODEL_CONFIG.get("target_windows", [24, 48])  # hours ahead
TARGET_CLASSES = MODEL_CONFIG.get("classes", ["None", "C", "M", "X"])  # ordered by severity


class FlareLabeler:
    """creates supervised labels for flare prediction."""

    def __init__(self):
        self.db = get_database()
        self.target_windows = TARGET_WINDOWS
        self.target_classes = TARGET_CLASSES
        # class hierarchy: X > M > C > None
        self.class_hierarchy = {"None": 0, "C": 1, "M": 2, "X": 3}

    def get_max_flare_class(self, flares: pd.DataFrame) -> str:
        """
        get the maximum flare class from a list of flares.

        args:
            flares: dataframe with flare events

        returns:
            highest class category (X > M > C > None)
        """
        if len(flares) == 0:
            return "None"

        # get highest class by hierarchy
        flare_classes = flares["class_category"].unique()
        max_class = "None"
        max_hierarchy = -1

        for flare_class in flare_classes:
            hierarchy = self.class_hierarchy.get(flare_class, -1)
            if hierarchy > max_hierarchy:
                max_hierarchy = hierarchy
                max_class = flare_class

        return max_class

    def create_labels_for_timestamp(
        self,
        timestamp: datetime,
        windows: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        create labels for a single timestamp.

        args:
            timestamp: timestamp to create labels for
            windows: list of prediction windows in hours (default: [24, 48])

        returns:
            dict with labels for each window
        """
        if windows is None:
            windows = self.target_windows

        labels = {"timestamp": timestamp}

        # query flare events in future windows
        for window in windows:
            window_start = timestamp
            window_end = timestamp + timedelta(hours=window)

            try:
                with self.db.get_session() as session:
                    # find flares that start within the prediction window
                    # we use start_time to catch flares that begin in the window
                    flares_query = (
                        session.query(FlareEvent)
                        .filter(
                            FlareEvent.start_time >= window_start,
                            FlareEvent.start_time < window_end,
                        )
                        .order_by(FlareEvent.start_time)
                    )
                    flares_records = flares_query.all()

                    flares_df = pd.DataFrame(
                        [
                            {
                                "start_time": r.start_time,
                                "peak_time": r.peak_time,
                                "class_category": r.class_category,
                                "class_magnitude": r.class_magnitude,
                            }
                            for r in flares_records
                        ],
                    )

                    # get maximum flare class
                    max_class = self.get_max_flare_class(flares_df)

                    # store label
                    labels[f"label_{window}h"] = max_class
                    labels[f"num_flares_{window}h"] = len(flares_df)

                    # store detailed info about flares
                    if len(flares_df) > 0:
                        labels[f"max_magnitude_{window}h"] = flares_df["class_magnitude"].max()
                        labels[f"flare_classes_{window}h"] = ",".join(flares_df["class_category"].unique())
                    else:
                        labels[f"max_magnitude_{window}h"] = 0.0
                        labels[f"flare_classes_{window}h"] = ""

            except Exception as e:
                logger.error(f"error creating labels for {timestamp} (window {window}h): {e}")
                labels[f"label_{window}h"] = "None"
                labels[f"num_flares_{window}h"] = 0

        return labels

    def create_labels(
        self,
        timestamps: List[datetime],
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        create labels for multiple timestamps.

        args:
            timestamps: list of timestamps to create labels for
            windows: list of prediction windows in hours (default: [24, 48])

        returns:
            dataframe with labels for each timestamp
        """
        if windows is None:
            windows = self.target_windows

        labels_list = []

        for timestamp in timestamps:
            try:
                label_dict = self.create_labels_for_timestamp(timestamp, windows)
                labels_list.append(label_dict)
            except Exception as e:
                logger.error(f"error creating labels for {timestamp}: {e}")
                continue

        if len(labels_list) == 0:
            return pd.DataFrame()

        labels_df = pd.DataFrame(labels_list)

        # ensure all target classes are represented
        for window in windows:
            label_col = f"label_{window}h"
            if label_col in labels_df.columns:
                # convert to categorical with all classes
                labels_df[label_col] = pd.Categorical(
                    labels_df[label_col],
                    categories=self.target_classes,
                    ordered=True,
                )

        return labels_df

    def create_labels_from_features(
        self,
        features_df: pd.DataFrame,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        create labels for feature dataframe.

        args:
            features_df: dataframe with features (must have 'timestamp' column)
            windows: list of prediction windows in hours (default: [24, 48])

        returns:
            dataframe with features and labels
        """
        if "timestamp" not in features_df.columns:
            raise ValueError("features dataframe must have 'timestamp' column")

        timestamps = features_df["timestamp"].tolist()
        labels_df = self.create_labels(timestamps, windows)

        # merge labels with features
        result_df = features_df.merge(labels_df, on="timestamp", how="left")

        return result_df


def create_labels(
    timestamps: List[datetime],
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    convenience function to create labels.

    args:
        timestamps: list of timestamps to create labels for
        windows: list of prediction windows in hours (default: [24, 48])

    returns:
        dataframe with labels for each timestamp
    """
    labeler = FlareLabeler()
    return labeler.create_labels(timestamps, windows)
# fmt: on

