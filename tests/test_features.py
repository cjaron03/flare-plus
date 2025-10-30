"""tests for feature engineering module."""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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


def test_mcintosh_complexity():
    """test mcintosh complexity computation."""
    # test valid classification
    result = compute_mcintosh_complexity("Dkc")
    assert result["mcintosh_size"] == "D"
    assert result["mcintosh_shape"] == "K"
    assert result["mcintosh_penumbra"] == "C"
    assert result["mcintosh_size_encoded"] == 4
    assert result["mcintosh_complexity_score"] > 0

    # test None input
    result = compute_mcintosh_complexity(None)
    assert result["mcintosh_complexity_score"] == 0.0

    # test invalid input
    result = compute_mcintosh_complexity("")
    assert result["mcintosh_complexity_score"] == 0.0


def test_mount_wilson_complexity():
    """test mount wilson complexity computation."""
    # test beta-gamma
    result = compute_mount_wilson_complexity("beta-gamma")
    assert result["mount_wilson_has_beta"] == 1
    assert result["mount_wilson_has_gamma"] == 1
    assert result["mount_wilson_complexity_score"] > 0

    # test delta
    result = compute_mount_wilson_complexity("beta-gamma-delta")
    assert result["mount_wilson_has_delta"] == 1
    assert result["mount_wilson_complexity_score"] > 0.5

    # test None
    result = compute_mount_wilson_complexity(None)
    assert result["mount_wilson_complexity_score"] == 0.0


def test_magnetic_complexity_score():
    """test magnetic complexity score computation."""
    result = compute_magnetic_complexity_score("beta-gamma")
    assert result["magnetic_has_beta"] == 1
    assert result["magnetic_has_gamma"] == 1
    assert result["magnetic_complexity_score"] > 0

    result = compute_magnetic_complexity_score(None)
    assert result["magnetic_complexity_score"] == 0.0


def test_flux_trends():
    """test flux trend computation."""
    # create sample flux data
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    flux_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "flux_short": np.random.uniform(1e-9, 1e-7, len(timestamps)),
            "flux_long": np.random.uniform(1e-8, 1e-6, len(timestamps)),
        }
    )

    result = compute_flux_trends(flux_data, datetime.now(), lookback_hours=24, window_hours=6)
    assert "flux_short_mean" in result
    assert "flux_long_mean" in result
    assert result["flux_short_mean"] is not None

    # test empty data
    result = compute_flux_trends(pd.DataFrame(), datetime.now())
    assert result["flux_short_mean"] is None


def test_flux_rate_of_change():
    """test flux rate of change computation."""
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(12, 0, -1)]
    flux_data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "flux_short": np.linspace(1e-9, 1e-7, len(timestamps)),
            "flux_long": np.linspace(1e-8, 1e-6, len(timestamps)),
        }
    )

    result = compute_flux_rate_of_change(flux_data, datetime.now(), lookback_hours=12)
    assert "flux_short_rate_of_change" in result
    assert result["flux_short_rate_of_change"] is not None


def test_rolling_statistics():
    """test rolling statistics computation."""
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": np.random.uniform(0, 100, len(timestamps)),
        }
    )

    result = compute_rolling_statistics(data, datetime.now(), "value", [6, 12, 24])
    assert "value_6h_mean" in result
    assert "value_12h_mean" in result
    assert "value_24h_mean" in result


def test_recency_weighted_flare_counts():
    """test recency-weighted flare counts."""
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    flare_data = pd.DataFrame(
        {
            "peak_time": timestamps[:5],
            "class_category": ["C", "M", "X", "C", "M"],
        }
    )

    result = compute_recency_weighted_flare_counts(flare_data, datetime.now(), ["C", "M", "X"], [6, 12, 24])
    assert "flare_C_6h_count" in result
    assert "flare_C_6h_weighted_count" in result
    assert result["flare_C_6h_count"] >= 0


def test_normalize_features():
    """test feature normalization."""
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "id": [1, 2, 3, 4, 5],
        }
    )

    normalized = normalize_features(data)
    assert normalized["feature1"].min() >= 0
    assert normalized["feature1"].max() <= 1
    assert normalized["id"].equals(data["id"])  # id should not be normalized


def test_standardize_features():
    """test feature standardization."""
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
        }
    )

    standardized = standardize_features(data)
    assert abs(standardized["feature1"].mean()) < 0.01  # should be near zero
    assert abs(standardized["feature1"].std() - 1.0) < 0.01  # should be near 1


def test_handle_missing_data():
    """test missing data handling."""
    data = pd.DataFrame(
        {
            "feature1": [1, 2, None, 4, 5],
            "feature2": [10, None, 30, 40, 50],
            "feature3": [None, None, None, None, None],  # all missing
        }
    )

    # test mean imputation
    processed = handle_missing_data(data, strategy="mean")
    assert processed["feature1"].isnull().sum() == 0
    assert processed["feature2"].isnull().sum() == 0

    # test drop strategy
    processed = handle_missing_data(data, strategy="drop", drop_threshold=0.5)
    assert "feature3" not in processed.columns  # should be dropped


def test_flag_missing_data_paths():
    """test missing data flagging."""
    data = pd.DataFrame(
        {
            "feature1": [1, 2, None, 4, 5],
            "feature2": [10, None, 30, 40, 50],
        }
    )

    result = flag_missing_data_paths(data)
    assert result["missing_count"] > 0
    assert len(result["columns_with_missing"]) > 0
    assert "feature1" in result["missing_stats"]
