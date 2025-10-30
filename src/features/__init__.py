"""feature engineering module for flare+ prediction system."""

from src.features.pipeline import FeatureEngineer
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
)

__all__ = [
    "FeatureEngineer",
    "compute_mcintosh_complexity",
    "compute_mount_wilson_complexity",
    "compute_magnetic_complexity_score",
    "compute_flux_trends",
    "compute_flux_rate_of_change",
    "compute_rolling_statistics",
    "compute_recency_weighted_flare_counts",
    "normalize_features",
    "standardize_features",
    "handle_missing_data",
]
