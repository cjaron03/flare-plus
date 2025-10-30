"""feature normalization, standardization, and missing data handling."""

import logging
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


def normalize_features(
    features: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "minmax",
) -> pd.DataFrame:
    """
    normalize features to [0, 1] range using min-max scaling.

    args:
        features: dataframe with features to normalize
        columns: list of column names to normalize (None = all numeric columns)
        method: normalization method ('minmax', 'robust')

    returns:
        dataframe with normalized features
    """
    if features is None or len(features) == 0:
        return features

    features_normalized = features.copy()

    # select columns to normalize
    if columns is None:
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        columns = [col for col in numeric_cols if col not in ["id", "timestamp"]]

    if len(columns) == 0:
        return features_normalized

    # apply normalization
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        logger.warning(f"unknown normalization method: {method}, using minmax")
        scaler = MinMaxScaler()

    # normalize only numeric columns that exist
    valid_columns = [col for col in columns if col in features_normalized.columns]
    if len(valid_columns) == 0:
        return features_normalized

    # handle missing values temporarily
    features_normalized[valid_columns] = features_normalized[valid_columns].fillna(0)

    # apply scaling
    features_normalized[valid_columns] = scaler.fit_transform(features_normalized[valid_columns])

    return features_normalized


def standardize_features(
    features: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "standard",
) -> pd.DataFrame:
    """
    standardize features to zero mean and unit variance.

    args:
        features: dataframe with features to standardize
        columns: list of column names to standardize (None = all numeric columns)
        method: standardization method ('standard', 'robust')

    returns:
        dataframe with standardized features
    """
    if features is None or len(features) == 0:
        return features

    features_standardized = features.copy()

    # select columns to standardize
    if columns is None:
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        columns = [col for col in numeric_cols if col not in ["id", "timestamp"]]

    if len(columns) == 0:
        return features_standardized

    # apply standardization
    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        logger.warning(f"unknown standardization method: {method}, using standard")
        scaler = StandardScaler()

    # standardize only numeric columns that exist
    valid_columns = [col for col in columns if col in features_standardized.columns]
    if len(valid_columns) == 0:
        return features_standardized

    # handle missing values temporarily
    features_standardized[valid_columns] = features_standardized[valid_columns].fillna(0)

    # apply scaling
    features_standardized[valid_columns] = scaler.fit_transform(features_standardized[valid_columns])

    return features_standardized


def handle_missing_data(
    features: pd.DataFrame,
    strategy: str = "forward_fill",
    fill_value: Optional[float] = None,
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    handle missing data using various strategies.

    args:
        features: dataframe with potentially missing values
        strategy: imputation strategy:
            - 'forward_fill': forward fill missing values
            - 'backward_fill': backward fill missing values
            - 'mean': fill with column mean
            - 'median': fill with column median
            - 'zero': fill with zero
            - 'constant': fill with constant value
            - 'drop': drop columns/rows with too many missing values
        fill_value: constant value to use for 'constant' strategy
        drop_threshold: threshold for dropping columns/rows (fraction of missing values)

    returns:
        dataframe with missing data handled
    """
    if features is None or len(features) == 0:
        return features

    features_processed = features.copy()

    # drop columns with too many missing values
    missing_frac = features_processed.isnull().sum() / len(features_processed)
    columns_to_drop = missing_frac[missing_frac > drop_threshold].index.tolist()
    if columns_to_drop:
        logger.info(f"dropping columns with >{drop_threshold*100}% missing: {columns_to_drop}")
        features_processed = features_processed.drop(columns=columns_to_drop)

    # apply imputation strategy
    numeric_cols = features_processed.select_dtypes(include=[np.number]).columns.tolist()

    if strategy == "forward_fill":
        features_processed[numeric_cols] = features_processed[numeric_cols].fillna(method="ffill")
        features_processed[numeric_cols] = features_processed[numeric_cols].fillna(method="bfill")  # fill remaining
    elif strategy == "backward_fill":
        features_processed[numeric_cols] = features_processed[numeric_cols].fillna(method="bfill")
        features_processed[numeric_cols] = features_processed[numeric_cols].fillna(method="ffill")  # fill remaining
    elif strategy == "mean":
        features_processed[numeric_cols] = features_processed[numeric_cols].fillna(
            features_processed[numeric_cols].mean()
        )
    elif strategy == "median":
        features_processed[numeric_cols] = features_processed[numeric_cols].fillna(
            features_processed[numeric_cols].median()
        )
    elif strategy == "zero":
        features_processed[numeric_cols] = features_processed[numeric_cols].fillna(0)
    elif strategy == "constant":
        if fill_value is None:
            logger.warning("fill_value not provided, using zero")
            fill_value = 0
        features_processed[numeric_cols] = features_processed[numeric_cols].fillna(fill_value)
    elif strategy == "drop":
        # drop rows with any missing values
        features_processed = features_processed.dropna()
    else:
        logger.warning(f"unknown strategy: {strategy}, using forward_fill")
        features_processed[numeric_cols] = features_processed[numeric_cols].fillna(method="ffill")
        features_processed[numeric_cols] = features_processed[numeric_cols].fillna(method="bfill")

    # fill any remaining missing values in non-numeric columns with a placeholder
    non_numeric_cols = features_processed.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in non_numeric_cols:
        if features_processed[col].isnull().any():
            features_processed[col] = features_processed[col].fillna("unknown")

    return features_processed


def flag_missing_data_paths(features: pd.DataFrame) -> Dict[str, Any]:
    """
    flag missing data paths and provide statistics.

    args:
        features: dataframe to analyze

    returns:
        dict with missing data statistics and flags
    """
    if features is None or len(features) == 0:
        return {"missing_count": 0, "missing_fraction": 0.0, "columns_with_missing": []}

    missing_count = features.isnull().sum().sum()
    missing_fraction = missing_count / (len(features) * len(features.columns))
    columns_with_missing = features.columns[features.isnull().any()].tolist()

    missing_stats = {}
    for col in columns_with_missing:
        missing_stats[col] = {
            "count": features[col].isnull().sum(),
            "fraction": features[col].isnull().sum() / len(features),
        }

    return {
        "missing_count": missing_count,
        "missing_fraction": missing_fraction,
        "columns_with_missing": columns_with_missing,
        "missing_stats": missing_stats,
    }
