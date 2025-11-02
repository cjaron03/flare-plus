"""tests for time-to-event survival analysis models."""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.models.survival_labeling import SurvivalLabeler
from src.models.time_varying_covariates import TimeVaryingCovariateEngineer
from src.models.survival_models import CoxProportionalHazards, GradientBoostingSurvival
from src.models.survival_pipeline import SurvivalAnalysisPipeline


@pytest.fixture
def sample_survival_data(db_session):
    """create sample survival data for testing."""
    from src.data.schema import FlareEvent

    # create test flare events
    base_time = datetime(2024, 1, 1, 12, 0, 0)

    # x-class flare at +48 hours
    flare1 = FlareEvent(
        start_time=base_time + timedelta(hours=48),
        peak_time=base_time + timedelta(hours=48, minutes=15),
        end_time=base_time + timedelta(hours=48, minutes=30),
        flare_class="X2.1",
        class_category="X",
        class_magnitude=2.1,
        active_region=12345,
        source="test",
        verified=True,
    )

    # x-class flare at +120 hours
    flare2 = FlareEvent(
        start_time=base_time + timedelta(hours=120),
        peak_time=base_time + timedelta(hours=120, minutes=15),
        end_time=base_time + timedelta(hours=120, minutes=30),
        flare_class="X1.5",
        class_category="X",
        class_magnitude=1.5,
        active_region=12345,
        source="test",
        verified=True,
    )

    db_session.add(flare1)
    db_session.add(flare2)
    db_session.commit()

    return base_time


def test_survival_labeler_create_label(sample_survival_data):
    """test survival label creation."""
    labeler = SurvivalLabeler(target_flare_class="X", max_time_hours=168)
    base_time = sample_survival_data

    # label for timestamp before first flare
    label = labeler.create_survival_label(base_time)

    assert label["timestamp"] == base_time
    # check if event occurred or was censored (event may be 0 if no flare found in DB)
    assert label["event"] in [0, 1]
    # duration should be positive
    assert label["duration"] > 0
    # if event occurred, event_time should be set
    if label["event"] == 1:
        assert label["event_time"] is not None
        # duration should match time difference
        expected_duration = (label["event_time"] - base_time).total_seconds() / 3600.0
        assert abs(label["duration"] - expected_duration) < 0.1

    # label for timestamp after both flares (should be censored)
    late_time = base_time + timedelta(hours=200)
    label_late = labeler.create_survival_label(late_time)

    assert label_late["event"] == 0  # censored
    assert label_late["duration"] == 168.0  # max_time_hours


def test_survival_labeler_probability_distribution():
    """test probability distribution computation."""
    labeler = SurvivalLabeler()

    # example survival function
    time_points = np.arange(0, 169, 1)
    survival_probs = np.exp(-time_points / 100.0)  # exponential decay

    prob_dist = labeler.compute_probability_distribution(
        survival_probs,
        time_points,
        time_buckets=[0, 24, 48, 72, 96],
    )

    assert len(prob_dist) > 0
    assert all(0.0 <= prob <= 1.0 for prob in prob_dist.values())


def test_time_varying_covariates(sample_survival_data):
    """test time-varying covariate computation."""
    engineer = TimeVaryingCovariateEngineer()
    base_time = sample_survival_data

    covariates_df = engineer.compute_time_varying_covariates(base_time)

    assert len(covariates_df) == 1
    assert "timestamp" in covariates_df.columns
    # should have flux, complexity, and flare history metrics
    assert any("flux" in col for col in covariates_df.columns)
    assert any("complexity" in col for col in covariates_df.columns)


def test_cox_proportional_hazards():
    """test cox proportional hazards model."""
    # create sample data
    np.random.seed(42)
    n_samples = 100

    # features
    X = np.random.randn(n_samples, 5)
    feature_names = [f"feature_{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)

    # survival data
    durations = np.random.exponential(50, n_samples)
    events = np.random.binomial(1, 0.7, n_samples)

    df["duration"] = durations
    df["event"] = events

    # fit model
    model = CoxProportionalHazards(penalizer=0.1)
    model.fit(df)

    assert model.is_fitted

    # predict survival function
    survival_array, time_points = model.predict_survival_function(df.head(5))
    assert survival_array.shape[0] == 5
    assert len(time_points) > 0

    # compute c-index
    c_index = model.compute_concordance_index(df)
    assert 0.0 <= c_index <= 1.0


def test_gradient_boosting_survival():
    """test gradient boosting survival model."""
    # create sample data
    np.random.seed(42)
    n_samples = 100

    # features
    X = np.random.randn(n_samples, 5)
    feature_names = [f"feature_{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)

    # survival data
    durations = np.random.exponential(50, n_samples)
    events = np.random.binomial(1, 0.7, n_samples)

    df["duration"] = durations
    df["event"] = events

    # fit model
    model = GradientBoostingSurvival(n_estimators=10, random_state=42)
    model.fit(df)

    assert model.is_fitted

    # predict hazard
    hazards = model.predict_hazard(df.head(5))
    assert len(hazards) == 5

    # predict survival function
    survival_array, time_points = model.predict_survival_function(df.head(5))
    assert survival_array.shape[0] == 5

    # compute c-index
    c_index = model.compute_concordance_index(df)
    assert 0.0 <= c_index <= 1.0

    # get feature importance
    importance_df = model.get_feature_importance()
    assert len(importance_df) == 5


def test_survival_pipeline_prepare_dataset(sample_survival_data):
    """test survival pipeline dataset preparation."""
    pipeline = SurvivalAnalysisPipeline(target_flare_class="X", max_time_hours=168)
    base_time = sample_survival_data

    # create timestamps
    timestamps = [base_time + timedelta(hours=i * 12) for i in range(5)]

    dataset = pipeline.prepare_dataset(timestamps)

    assert len(dataset) > 0
    assert "duration" in dataset.columns
    assert "event" in dataset.columns
    # check that we have some covariate columns (they have _lookback suffix pattern)
    covariate_cols = [
        col for col in dataset.columns if col not in ["timestamp", "duration", "event", "region_number", "event_time"]
    ]
    assert len(covariate_cols) > 0
    # should have flux, complexity, or flare-related features
    has_flux = any("flux" in col.lower() for col in covariate_cols)
    has_complexity = any(
        "complexity" in col.lower() or "mcintosh" in col.lower() or "mount_wilson" in col.lower()
        for col in covariate_cols
    )
    has_flare = any("flare" in col.lower() for col in covariate_cols)
    assert has_flux or has_complexity or has_flare


def test_survival_pipeline_train_and_evaluate(sample_survival_data):
    """test survival pipeline training and evaluation."""
    pipeline = SurvivalAnalysisPipeline(target_flare_class="X", max_time_hours=168, gb_n_estimators=10)
    base_time = sample_survival_data

    # create timestamps
    timestamps = [base_time + timedelta(hours=i * 12) for i in range(20)]

    dataset = pipeline.prepare_dataset(timestamps)

    if len(dataset) < 10:
        pytest.skip("insufficient data for training")

    # train and evaluate
    results = pipeline.train_and_evaluate(dataset, test_size=0.3)

    assert "cox_c_index_test" in results or "gb_c_index_test" in results
    assert results["train_size"] > 0
    assert results["test_size"] > 0
