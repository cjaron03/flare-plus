"""tests for short-term classification models."""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.models.labeling import FlareLabeler
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.models.pipeline import ClassificationPipeline


def _balanced_labels(n_samples: int):
    """generate at least three samples per flare class for stratified cv."""
    base = ["None", "C", "M", "X"]
    repeats = (n_samples + len(base) - 1) // len(base)
    return (base * repeats)[:n_samples]


@pytest.fixture
def sample_flare_events(db_session):
    """create sample flare events for testing."""
    from src.data.schema import FlareEvent

    # create flares at different times
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    flares = [
        FlareEvent(
            start_time=base_time + timedelta(hours=i),
            peak_time=base_time + timedelta(hours=i, minutes=15),
            end_time=base_time + timedelta(hours=i, minutes=30),
            flare_class="C1.2",
            class_category="C",
            class_magnitude=1.2,
            active_region=12345,
            source="test",
        )
        for i in range(5)
    ]

    # add one M-class flare
    flares.append(
        FlareEvent(
            start_time=base_time + timedelta(hours=10),
            peak_time=base_time + timedelta(hours=10, minutes=15),
            end_time=base_time + timedelta(hours=10, minutes=30),
            flare_class="M5.5",
            class_category="M",
            class_magnitude=5.5,
            active_region=12345,
            source="test",
        )
    )

    for flare in flares:
        db_session.add(flare)
    db_session.commit()

    return flares


@pytest.fixture
def sample_features():
    """create sample feature dataframe."""
    n_samples = 12
    timestamps = [datetime(2024, 1, 1, 12, 0, 0) + timedelta(hours=i) for i in range(n_samples)]
    features = {
        "timestamp": timestamps,
        "flux_short_mean_6h": np.random.rand(n_samples),
        "flux_long_mean_6h": np.random.rand(n_samples),
        "region_area": np.random.randint(10, 100, n_samples),
        "region_num_sunspots": np.random.randint(1, 20, n_samples),
    }
    return pd.DataFrame(features)


def test_flare_labeler_initialization():
    """test flare labeler initialization."""
    labeler = FlareLabeler()
    assert labeler.db is not None
    assert labeler.target_windows == [24, 48]
    assert labeler.target_classes == ["None", "C", "M", "X"]


def test_get_max_flare_class():
    """test getting maximum flare class."""
    labeler = FlareLabeler()

    # test with no flares
    flares_df = pd.DataFrame()
    assert labeler.get_max_flare_class(flares_df) == "None"

    # test with C-class flare
    flares_df = pd.DataFrame([{"class_category": "C"}])
    assert labeler.get_max_flare_class(flares_df) == "C"

    # test with multiple flares
    flares_df = pd.DataFrame([{"class_category": "C"}, {"class_category": "M"}, {"class_category": "C"}])
    assert labeler.get_max_flare_class(flares_df) == "M"

    # test with X-class flare
    flares_df = pd.DataFrame([{"class_category": "C"}, {"class_category": "X"}])
    assert labeler.get_max_flare_class(flares_df) == "X"


def test_create_labels_for_timestamp(sample_flare_events):
    """test creating labels for a timestamp."""
    labeler = FlareLabeler()

    # create label for timestamp before flares
    timestamp = datetime(2024, 1, 1, 12, 0, 0)
    labels = labeler.create_labels_for_timestamp(timestamp, windows=[24])

    assert "timestamp" in labels
    assert "label_24h" in labels
    assert labels["timestamp"] == timestamp


def test_create_labels(sample_flare_events):
    """test creating labels for multiple timestamps."""
    labeler = FlareLabeler()

    timestamps = [datetime(2024, 1, 1, 12, 0, 0) + timedelta(hours=i) for i in range(5)]
    labels_df = labeler.create_labels(timestamps, windows=[24])

    assert len(labels_df) == 5
    assert "label_24h" in labels_df.columns
    assert "timestamp" in labels_df.columns


def test_create_labels_from_features(sample_features, sample_flare_events):
    """test creating labels from feature dataframe."""
    labeler = FlareLabeler()

    labeled_df = labeler.create_labels_from_features(sample_features, windows=[24])

    assert len(labeled_df) == len(sample_features)
    assert "label_24h" in labeled_df.columns
    assert "timestamp" in labeled_df.columns


def test_model_trainer_initialization():
    """test model trainer initialization."""
    trainer = ModelTrainer(use_smote=True, cv_folds=5)
    assert trainer.use_smote is True
    assert trainer.cv_folds == 5


def test_prepare_features_and_labels(sample_features):
    """test preparing features and labels."""
    trainer = ModelTrainer()

    # add labels
    sample_features["label_24h"] = _balanced_labels(len(sample_features))

    X, y, feature_names = trainer.prepare_features_and_labels(sample_features, "label_24h")

    assert X.shape[0] == len(sample_features)
    assert len(feature_names) > 0
    assert len(y) == len(sample_features)


def test_train_logistic_regression(sample_features):
    """test training logistic regression model."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3)

    # add labels
    sample_features["label_24h"] = _balanced_labels(len(sample_features))

    X, y, _ = trainer.prepare_features_and_labels(sample_features, "label_24h")

    model, info = trainer.train_logistic_regression(X, y)

    assert model is not None
    assert "cv_mean" in info
    assert "model_type" in info
    assert info["model_type"] == "logistic_regression"


def test_train_gradient_boosting(sample_features):
    """test training gradient boosting model."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3)

    # add labels
    sample_features["label_24h"] = _balanced_labels(len(sample_features))

    X, y, _ = trainer.prepare_features_and_labels(sample_features, "label_24h")

    model, info = trainer.train_gradient_boosting(X, y)

    assert model is not None
    assert "cv_mean" in info
    assert "model_type" in info
    assert info["model_type"] == "gradient_boosting"


def test_train_baseline_models(sample_features):
    """test training baseline models."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3)

    # add labels
    sample_features["label_24h"] = _balanced_labels(len(sample_features))

    trained_models = trainer.train_baseline_models(sample_features, "label_24h", models=["logistic"])

    assert "logistic_regression" in trained_models
    model, info = trained_models["logistic_regression"]
    assert model is not None
    assert "cv_mean" in info


def test_model_evaluator_initialization():
    """test model evaluator initialization."""
    evaluator = ModelEvaluator()
    assert evaluator.label_encoder is None


def test_compute_brier_score():
    """test computing brier score."""
    evaluator = ModelEvaluator()

    y_true = np.array([0, 1, 2, 0, 1])
    y_prob = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])

    brier_scores = evaluator.compute_brier_score(y_true, y_prob, classes=["None", "C", "M"])

    assert "None" in brier_scores
    assert "C" in brier_scores
    assert "M" in brier_scores
    assert "macro_avg" in brier_scores


def test_compute_roc_auc_per_class():
    """test computing roc-auc per class."""
    evaluator = ModelEvaluator()

    y_true = np.array([0, 1, 2, 0, 1])
    y_prob = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])

    roc_auc_scores = evaluator.compute_roc_auc_per_class(y_true, y_prob, classes=["None", "C", "M"])

    assert "None" in roc_auc_scores
    assert "C" in roc_auc_scores
    assert "M" in roc_auc_scores
    assert "macro_avg" in roc_auc_scores


def test_calibrate_probabilities(sample_features):
    """test probability calibration."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3)

    # add labels - ensure at least 2 examples per class for 2-fold cv
    sample_features["label_24h"] = _balanced_labels(len(sample_features))

    X, y, _ = trainer.prepare_features_and_labels(sample_features, "label_24h")

    model, _ = trainer.train_logistic_regression(X, y)

    evaluator = ModelEvaluator()
    # use cv=2 for small dataset to avoid cross-validation error
    calibrated_model, info = evaluator.calibrate_probabilities(model, X, y, cv=2)

    assert calibrated_model is not None
    assert "method" in info


def test_evaluate_model(sample_features):
    """test comprehensive model evaluation."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3)

    # add labels
    sample_features["label_24h"] = _balanced_labels(len(sample_features))

    X, y, _ = trainer.prepare_features_and_labels(sample_features, "label_24h")

    model, _ = trainer.train_logistic_regression(X, y)

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, X, y, classes=["None", "C", "M", "X"], calibrate=False)

    assert "brier_score" in results
    assert "roc_auc" in results
    assert "classification_report" in results
    assert "confusion_matrix" in results


def test_classification_pipeline_initialization():
    """test classification pipeline initialization."""
    pipeline = ClassificationPipeline(use_smote=True, cv_folds=5)
    assert pipeline.use_smote is True
    assert pipeline.cv_folds == 5
    assert pipeline.feature_engineer is not None
    assert pipeline.labeler is not None
    assert pipeline.trainer is not None
    assert pipeline.evaluator is not None
