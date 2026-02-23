"""tests for short-term classification models."""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.models.labeling import FlareLabeler
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.models.pipeline import ClassificationPipeline
from src.models.ensemble import ProbabilityAveragingEnsemble


def _balanced_labels(n_samples: int):
    """generate labels with at least three samples per class for stratified CV."""
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
    n_samples = 20
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


def test_train_logistic_regression_with_single_class_early_folds():
    """test logistic training when early time-series folds have one class."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3)

    X = np.random.rand(20, 4)
    # first folds contain only class 0, later folds include class 1
    y = np.array([0] * 12 + [1] * 8)

    model, info = trainer.train_logistic_regression(X, y, use_class_weight=False)

    assert model is not None
    assert "cv_scores" in info
    assert len(info["cv_scores"]) >= 1


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
    assert "event_metrics" in results
    assert "f1" in results["event_metrics"]


def test_evaluate_model_with_missing_test_class(sample_features):
    """evaluation should handle test splits where one class is absent."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3)

    sample_features["label_24h"] = _balanced_labels(len(sample_features))

    X, y, _ = trainer.prepare_features_and_labels(sample_features, "label_24h")
    model, _ = trainer.train_logistic_regression(X, y)

    # remove one class from the evaluation subset to mimic temporal holdout gaps
    holdout_class = np.max(y)
    mask = y != holdout_class
    X_eval = X[mask]
    y_eval = y[mask]

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(
        model,
        X_eval,
        y_eval,
        classes=["None", "C", "M", "X"],
        calibrate=False,
    )

    assert "classification_report" in results
    assert "X" in results["classification_report"]
    assert results["classification_report"]["X"]["support"] == 0.0
    assert len(results["confusion_matrix"]) == 4


def test_train_lightgbm(sample_features):
    """test training lightgbm model."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3)

    sample_features["label_24h"] = _balanced_labels(len(sample_features))
    X, y, _ = trainer.prepare_features_and_labels(sample_features, "label_24h")

    model, info = trainer.train_lightgbm(X, y)

    assert model is not None
    assert "cv_mean" in info
    assert info["model_type"] == "lightgbm"


def test_train_random_forest(sample_features):
    """test training random forest model."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3)

    sample_features["label_24h"] = _balanced_labels(len(sample_features))
    X, y, _ = trainer.prepare_features_and_labels(sample_features, "label_24h")

    model, info = trainer.train_random_forest(X, y)

    assert model is not None
    assert "cv_mean" in info
    assert info["model_type"] == "random_forest"


def test_train_baseline_models_all_types(sample_features):
    """test training all model types via train_baseline_models."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3)

    sample_features["label_24h"] = _balanced_labels(len(sample_features))

    trained_models = trainer.train_baseline_models(
        sample_features,
        "label_24h",
        models=["logistic", "gradient_boosting", "lightgbm", "random_forest"],
    )

    assert "logistic" in trained_models
    assert "gradient_boosting" in trained_models
    assert "lightgbm" in trained_models
    assert "random_forest" in trained_models
    assert "best" in trained_models

    # verify best has selected_from key
    _, best_info = trained_models["best"]
    assert "selected_from" in best_info
    assert best_info["selected_from"] in ["logistic", "gradient_boosting", "lightgbm", "random_forest"]


def test_auto_select_best_model():
    """test select_best_model static method."""
    from unittest.mock import MagicMock

    mock_model_a = MagicMock()
    mock_model_b = MagicMock()

    trained_models = {
        "logistic": (mock_model_a, {"cv_mean": 0.80, "cv_f1_mean": 0.78}),
        "gradient_boosting": (mock_model_b, {"cv_mean": 0.85, "cv_f1_mean": 0.83}),
    }

    result = ModelTrainer.select_best_model(trained_models)
    assert result is not None
    best_model, best_info = result
    assert best_model is mock_model_b
    assert best_info["selected_from"] == "gradient_boosting"

    # test tie-break on F1
    trained_models_tie = {
        "logistic": (mock_model_a, {"cv_mean": 0.85, "cv_f1_mean": 0.90}),
        "gradient_boosting": (mock_model_b, {"cv_mean": 0.85, "cv_f1_mean": 0.83}),
    }

    result_tie = ModelTrainer.select_best_model(trained_models_tie)
    assert result_tie is not None
    _, tie_info = result_tie
    assert tie_info["selected_from"] == "logistic"

    # event-focused priority should beat higher raw accuracy
    trained_models_event = {
        "logistic": (
            mock_model_a,
            {"cv_mean": 0.82, "cv_f1_mean": 0.80, "cv_event_f1_mean": 0.72, "cv_event_recall_mean": 0.68},
        ),
        "gradient_boosting": (
            mock_model_b,
            {"cv_mean": 0.88, "cv_f1_mean": 0.86, "cv_event_f1_mean": 0.61, "cv_event_recall_mean": 0.59},
        ),
    }
    result_event = ModelTrainer.select_best_model(trained_models_event)
    assert result_event is not None
    _, event_info = result_event
    assert event_info["selected_from"] == "logistic"


def test_feature_selection(sample_features):
    """test feature selection with mutual information."""
    trainer = ModelTrainer(use_smote=False, cv_folds=3, use_feature_selection=True)

    sample_features["label_24h"] = _balanced_labels(len(sample_features))

    X, y, feature_names = trainer.prepare_features_and_labels(sample_features, "label_24h")

    X_sel, sel_names = trainer.select_features(X, y, feature_names)

    # should keep at least min_features=5 or all if fewer than 5
    assert len(sel_names) >= min(5, len(feature_names))
    assert X_sel.shape[1] == len(sel_names)
    assert X_sel.shape[0] == X.shape[0]

    # integration: verify train_baseline_models trains on pre-selected features
    # (feature selection is now called by the pipeline, not by train_baseline_models)
    selected_df = sample_features[sel_names + ["label_24h"]].copy()
    trained_models = trainer.train_baseline_models(
        selected_df,
        "label_24h",
        models=["logistic"],
    )

    assert "logistic" in trained_models
    _, info = trained_models["logistic"]
    assert "feature_names" in info
    assert len(info["feature_names"]) == len(sel_names)


def test_classification_pipeline_initialization():
    """test classification pipeline initialization."""
    pipeline = ClassificationPipeline(use_smote=True, cv_folds=5)
    assert pipeline.use_smote is True
    assert pipeline.cv_folds == 5
    assert pipeline.feature_engineer is not None
    assert pipeline.labeler is not None
    assert pipeline.trainer is not None
    assert pipeline.evaluator is not None


def test_classification_pipeline_chronological_split():
    """test that pipeline split preserves chronological order."""
    pipeline = ClassificationPipeline(use_smote=False, cv_folds=3)

    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20)

    X_train, X_test, y_train, y_test = pipeline._chronological_train_test_split(
        X,
        y,
        test_size=0.2,
    )

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert y_train.tolist() == list(range(16))
    assert y_test.tolist() == list(range(16, 20))


def test_probability_averaging_ensemble():
    """ensemble should average probabilities and expose argmax predictions."""

    class _MockModel:
        def __init__(self, probs):
            self._probs = np.asarray(probs, dtype=float)

        def predict_proba(self, X):
            return np.tile(self._probs, (len(X), 1))

    model_a = _MockModel([0.8, 0.2])
    model_b = _MockModel([0.4, 0.6])
    ensemble = ProbabilityAveragingEnsemble([model_a, model_b], model_names=["a", "b"])

    X = np.array([[1.0], [2.0]])
    probs = ensemble.predict_proba(X)
    preds = ensemble.predict(X)

    assert probs.shape == (2, 2)
    assert np.allclose(probs[0], np.array([0.6, 0.4]))
    assert preds.tolist() == [0, 0]


def test_classification_pipeline_predict_uses_feature_fill_values(monkeypatch):
    """predict should use per-feature fill defaults instead of hard-coded zeros."""
    from sklearn.preprocessing import LabelEncoder

    class _DummyModel:
        def __init__(self):
            self.last_X = None

        def predict(self, X):
            self.last_X = X
            return np.array([1])

        def predict_proba(self, X):
            self.last_X = X
            return np.array([[0.2, 0.8]])

    pipeline = ClassificationPipeline(use_smote=False, cv_folds=3)
    dummy_model = _DummyModel()
    encoder = LabelEncoder()
    encoder.fit(["None", "M"])

    target_time = datetime(2024, 1, 5, 0, 0, 0)
    pipeline.models = {
        "24h": {
            "best": {
                "model": dummy_model,
                "label_encoder": encoder,
                "feature_names": ["feature_a", "feature_b"],
                "feature_fill_values": {"feature_b": 7.5},
            }
        }
    }

    feature_frame = pd.DataFrame([{"timestamp": target_time, "feature_a": 1.25}])
    monkeypatch.setattr(pipeline.feature_engineer, "compute_features", lambda *args, **kwargs: feature_frame)

    result = pipeline.predict(timestamp=target_time, window=24, model_type="best")

    assert result["predicted_class"] in {"None", "M"}
    assert dummy_model.last_X is not None
    assert float(dummy_model.last_X[0, 1]) == pytest.approx(7.5)
