"""tests for api endpoints."""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch

from src.api.app import create_app
from src.models.pipeline import ClassificationPipeline
from src.models.survival_pipeline import SurvivalAnalysisPipeline


@pytest.fixture
def mock_classification_pipeline():
    """create mock classification pipeline."""
    pipeline = Mock(spec=ClassificationPipeline)
    pipeline.models = {
        "24h": {
            "gradient_boosting": {
                "model": Mock(),
                "label_encoder": Mock(),
                "feature_names": ["feature1", "feature2"],
            },
            "logistic": {
                "model": Mock(),
                "label_encoder": Mock(),
                "feature_names": ["feature1", "feature2"],
            },
        },
        "48h": {
            "gradient_boosting": {
                "model": Mock(),
                "label_encoder": Mock(),
                "feature_names": ["feature1", "feature2"],
            },
            "logistic": {
                "model": Mock(),
                "label_encoder": Mock(),
                "feature_names": ["feature1", "feature2"],
            },
        },
    }

    # create mock feature engineer that returns a DataFrame-like object
    import pandas as pd  # noqa: F401

    mock_features_df = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
    pipeline.feature_engineer = Mock()
    pipeline.feature_engineer.compute_features = Mock(return_value=mock_features_df)

    # make predict return value depend on window parameter
    def predict_side_effect(*args, **kwargs):
        window = kwargs.get("window", 24)
        return {
            "timestamp": datetime.now(),
            "window_hours": window,
            "predicted_class": "C",
            "class_probabilities": {"None": 0.5, "C": 0.3, "M": 0.15, "X": 0.05},
            "model_type": kwargs.get("model_type", "gradient_boosting"),
        }

    pipeline.predict = Mock(side_effect=predict_side_effect)
    return pipeline


@pytest.fixture
def mock_survival_pipeline():
    """create mock survival pipeline."""
    pipeline = Mock(spec=SurvivalAnalysisPipeline)
    pipeline.is_fitted = True
    pipeline.target_flare_class = "X"
    pipeline.max_time_hours = 168
    pipeline.cox_model = Mock()
    pipeline.gb_model = Mock()

    # create mock covariate engineer that returns a DataFrame-like object
    import pandas as pd  # noqa: F401

    mock_covariates_df = pd.DataFrame({"covariate1": [1.0], "covariate2": [2.0]})
    pipeline.covariate_engineer = Mock()
    pipeline.covariate_engineer.compute_time_varying_covariates = Mock(return_value=mock_covariates_df)

    # make predict return value depend on model_type parameter
    def predict_side_effect(*args, **kwargs):
        model_type = kwargs.get("model_type", "cox")
        return {
            "timestamp": datetime.now(),
            "model_type": model_type,
            "hazard_score": 120.5,
            "probability_distribution": {
                "0h-6h": 0.8,
                "6h-12h": 0.1,
                "12h-24h": 0.05,
                "24h-48h": 0.03,
                "48h-72h": 0.02,
            },
        }

    pipeline.predict_survival_probabilities = Mock(side_effect=predict_side_effect)
    return pipeline


@pytest.fixture
def app(mock_classification_pipeline, mock_survival_pipeline):
    """create flask app for testing."""
    app = create_app(
        classification_pipeline=mock_classification_pipeline,
        survival_pipeline=mock_survival_pipeline,
    )
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """create test client."""
    return app.test_client()


class TestHealthEndpoint:
    """tests for /health endpoint."""

    def test_health_check_success(self, client):
        """test health check returns success."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "status" in data
        assert "classification_available" in data
        assert "survival_available" in data

    def test_health_check_with_no_models(self, client):
        """test health check with no models (degraded status)."""
        # create app without models
        app_no_models = create_app()
        app_no_models.config["TESTING"] = True
        client_no_models = app_no_models.test_client()

        response = client_no_models.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "degraded"
        assert data["classification_available"] is False
        assert data["survival_available"] is False


class TestClassificationEndpoint:
    """tests for /predict/classification endpoint."""

    def test_classification_prediction_success(self, client, mock_classification_pipeline):
        """test successful classification prediction."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "window": 24,
            "model_type": "gradient_boosting",
        }

        response = client.post(
            "/predict/classification",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "predicted_class" in data
        assert "class_probabilities" in data
        assert data["window_hours"] == 24

        # verify pipeline was called
        mock_classification_pipeline.predict.assert_called_once()

    def test_classification_prediction_missing_timestamp(self, client):
        """test classification prediction without timestamp."""
        payload = {"window": 24}

        response = client.post(
            "/predict/classification",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "timestamp" in data["error"].lower()

    def test_classification_prediction_invalid_window(self, client):
        """test classification prediction with invalid window."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "window": 36,  # invalid window
        }

        response = client.post(
            "/predict/classification",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_classification_prediction_with_region(self, client, mock_classification_pipeline):
        """test classification prediction with region number."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "window": 48,
            "model_type": "logistic",
            "region_number": 1234,
        }

        response = client.post(
            "/predict/classification",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["window_hours"] == 48

        # verify region was passed
        call_args = mock_classification_pipeline.predict.call_args
        assert call_args[1]["region_number"] == 1234


class TestSurvivalEndpoint:
    """tests for /predict/survival endpoint."""

    def test_survival_prediction_success(self, client, mock_survival_pipeline):
        """test successful survival prediction."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "model_type": "cox",
        }

        response = client.post(
            "/predict/survival",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "probability_distribution" in data
        assert "hazard_score" in data
        assert data["model_type"] == "cox"

        # verify pipeline was called
        mock_survival_pipeline.predict_survival_probabilities.assert_called_once()

    def test_survival_prediction_missing_timestamp(self, client):
        """test survival prediction without timestamp."""
        payload = {"model_type": "cox"}

        response = client.post(
            "/predict/survival",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_survival_prediction_invalid_model_type(self, client):
        """test survival prediction with invalid model type."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "model_type": "invalid",
        }

        response = client.post(
            "/predict/survival",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_survival_prediction_with_custom_buckets(self, client, mock_survival_pipeline):
        """test survival prediction with custom time buckets."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "model_type": "gb",
            "time_buckets": [0, 6, 12, 24],
        }

        response = client.post(
            "/predict/survival",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 200

        # verify custom buckets were passed
        call_args = mock_survival_pipeline.predict_survival_probabilities.call_args
        assert call_args[1]["time_buckets"] == [0, 6, 12, 24]


class TestCombinedEndpoint:
    """tests for /predict/all endpoint."""

    def test_combined_prediction_success(self, client, mock_classification_pipeline, mock_survival_pipeline):
        """test successful combined prediction."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "classification_windows": [24, 48],
            "survival_model_type": "cox",
        }

        response = client.post(
            "/predict/all",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "classifications" in data
        assert "survival" in data
        assert "24h" in data["classifications"]
        assert "48h" in data["classifications"]

    def test_combined_prediction_missing_timestamp(self, client):
        """test combined prediction without timestamp."""
        payload = {}

        response = client.post(
            "/predict/all",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_combined_prediction_with_region(self, client, mock_classification_pipeline, mock_survival_pipeline):
        """test combined prediction with region number."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "region_number": 5678,
        }

        response = client.post(
            "/predict/all",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["region_number"] == 5678


class TestMonitoringIntegration:
    """tests for monitoring hooks integration."""

    @patch("src.api.service.InputDriftDetector")
    @patch("src.api.service.OutcomeLogger")
    def test_drift_detection_integration(self, mock_outcome_logger, mock_drift_detector, mock_classification_pipeline):
        """test drift detection is called during prediction."""
        mock_detector_instance = Mock()
        mock_detector_instance.detect_drift.return_value = {
            "drifted_features": ["feature1"],
            "overall_drift": True,
        }
        mock_drift_detector.return_value = mock_detector_instance

        # create service with drift detection
        from src.api.service import PredictionService  # noqa: F401

        service = PredictionService(
            classification_pipeline=mock_classification_pipeline,
            enable_drift_detection=True,
        )

        # make prediction
        timestamp = datetime.now()
        result = service.predict_classification(
            timestamp=timestamp,
            window=24,
        )

        # verify drift detection was called
        assert mock_detector_instance.detect_drift.called
        assert "drift_detection" in result

    @patch("src.api.service.OutcomeLogger")
    def test_outcome_logging_integration(self, mock_outcome_logger, mock_classification_pipeline):
        """test outcome logging is called during prediction."""
        mock_logger_instance = Mock()
        mock_outcome_logger.return_value = mock_logger_instance

        # create service with outcome logging
        from src.api.service import PredictionService  # noqa: F401

        service = PredictionService(
            classification_pipeline=mock_classification_pipeline,
            enable_outcome_logging=True,
        )

        # make prediction
        timestamp = datetime.now()
        service.predict_classification(
            timestamp=timestamp,
            window=24,
        )

        # verify outcome logging was called
        assert mock_logger_instance.log_prediction.called

    def test_health_check_includes_monitoring_status(self, client):
        """test health check includes monitoring status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "drift_detection_enabled" in data
        assert "outcome_logging_enabled" in data


class TestErrorHandling:
    """tests for error handling."""

    def test_invalid_json(self, client):
        """test handling of invalid json."""
        response = client.post(
            "/predict/classification",
            data="invalid json",
            content_type="application/json",
        )

        # flask should handle this gracefully
        assert response.status_code in [400, 500]

    def test_malformed_timestamp(self, client):
        """test handling of malformed timestamp."""
        payload = {
            "timestamp": "not-a-date",
            "window": 24,
        }

        response = client.post(
            "/predict/classification",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_prediction_internal_error(self, client, mock_classification_pipeline):
        """test handling of internal prediction errors."""
        # make pipeline raise exception
        mock_classification_pipeline.predict.side_effect = Exception("model error")

        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "window": 24,
        }

        response = client.post(
            "/predict/classification",
            data=json.dumps(payload),
            content_type="application/json",
        )

        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data
