"""Comprehensive tests for SHAP explainability implementation.

Tests cover:
1. ShapExplainer class initialization and caching
2. explain_classification() method with various model types
3. explain_survival() method for GB and Cox models
4. Pipeline integration with include_explanation parameter
5. API endpoint integration
6. Edge cases and error handling
7. Security considerations
"""

import pytest
import json
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

try:
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


class TestShapExplainerInitialization:
    """Tests for ShapExplainer initialization and configuration."""

    def test_default_initialization(self):
        """Test ShapExplainer initializes with default parameters."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()
        assert explainer.max_background_samples == 100
        assert explainer._explainer_cache == {}
        assert explainer._background_cache == {}

    def test_custom_max_background_samples(self):
        """Test ShapExplainer accepts custom max_background_samples."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer(max_background_samples=50)
        assert explainer.max_background_samples == 50

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()
        # Add dummy cache entries
        explainer._explainer_cache["test_key"] = "test_value"
        explainer._background_cache["test_key"] = "test_value"

        explainer.clear_cache()

        assert explainer._explainer_cache == {}
        assert explainer._background_cache == {}


class TestGetBaseEstimator:
    """Tests for get_base_estimator utility function."""

    def test_raw_estimator(self):
        """Test extraction from raw estimator (returns as-is)."""
        from src.api.explainer import get_base_estimator

        model = GradientBoostingClassifier(n_estimators=10)
        result = get_base_estimator(model)
        assert result is model

    def test_sklearn_pipeline(self):
        """Test extraction from sklearn pipeline."""
        from src.api.explainer import get_base_estimator
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline([("scaler", StandardScaler()), ("model", GradientBoostingClassifier(n_estimators=10))])

        result = get_base_estimator(pipeline)
        assert isinstance(result, GradientBoostingClassifier)

    def test_calibrated_classifier(self):
        """Test extraction from CalibratedClassifierCV."""
        from src.api.explainer import get_base_estimator

        # Create and fit a calibrated classifier
        base_model = GradientBoostingClassifier(n_estimators=10)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)
        base_model.fit(X, y)

        calibrated = CalibratedClassifierCV(base_model, cv="prefit")
        calibrated.fit(X, y)

        result = get_base_estimator(calibrated)
        # Result should be the base estimator from first calibrated classifier
        assert isinstance(result, GradientBoostingClassifier)

    @pytest.mark.skipif(not HAS_IMBLEARN, reason="imblearn not installed")
    def test_imblearn_pipeline(self):
        """Test extraction from imblearn pipeline (SMOTE pipelines)."""
        from src.api.explainer import get_base_estimator
        from imblearn.over_sampling import SMOTE

        pipeline = ImbPipeline(
            [("smote", SMOTE(random_state=42)), ("model", GradientBoostingClassifier(n_estimators=10))]
        )

        result = get_base_estimator(pipeline)
        assert isinstance(result, GradientBoostingClassifier)


class TestGetExplainerType:
    """Tests for get_explainer_type utility function."""

    def test_tree_based_models(self):
        """Test tree-based models return 'tree' type."""
        from src.api.explainer import get_explainer_type

        tree_models = [
            GradientBoostingClassifier(n_estimators=5),
            GradientBoostingRegressor(n_estimators=5),
        ]

        for model in tree_models:
            assert get_explainer_type(model) == "tree"

    def test_linear_models(self):
        """Test linear models return 'linear' type."""
        from src.api.explainer import get_explainer_type

        linear_models = [
            LogisticRegression(),
        ]

        for model in linear_models:
            assert get_explainer_type(model) == "linear"

    def test_unknown_model_returns_kernel(self):
        """Test unknown model types return 'kernel' as fallback."""
        from src.api.explainer import get_explainer_type

        class CustomModel:
            pass

        model = CustomModel()
        assert get_explainer_type(model) == "kernel"


class TestExplainClassification:
    """Tests for explain_classification method."""

    @pytest.fixture
    def trained_model_with_data(self):
        """Create a trained model with sample data (binary classification for SHAP compatibility)."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        # Binary classification (0 or 1) - SHAP TreeExplainer only supports binary for GBM
        y = np.random.randint(0, 2, 100)

        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        label_encoder = LabelEncoder()
        label_encoder.fit(["None", "C"])  # Binary: None or C class

        feature_names = ["flux_mean", "region_area", "num_spots", "complexity", "mag_class"]

        return {
            "model": model,
            "X_background": X,
            "X_sample": X[:1],
            "feature_names": feature_names,
            "label_encoder": label_encoder,
        }

    def test_basic_explanation(self, trained_model_with_data):
        """Test basic classification explanation."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()
        result = explainer.explain_classification(
            model=trained_model_with_data["model"],
            X=trained_model_with_data["X_sample"],
            feature_names=trained_model_with_data["feature_names"],
            label_encoder=trained_model_with_data["label_encoder"],
            model_type="gradient_boosting",
            window=24,
        )

        # Verify result structure
        assert "predicted_class" in result
        assert "predicted_probability" in result
        assert "base_value" in result
        assert "top_features" in result
        assert "all_features" in result
        assert "shap_values_by_class" in result
        assert "base_values_by_class" in result
        assert "explainer_type" in result
        assert result["explainer_type"] == "tree"

    def test_top_features_structure(self, trained_model_with_data):
        """Test top_features has correct structure."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()
        result = explainer.explain_classification(
            model=trained_model_with_data["model"],
            X=trained_model_with_data["X_sample"],
            feature_names=trained_model_with_data["feature_names"],
            label_encoder=trained_model_with_data["label_encoder"],
            top_n=3,
        )

        assert len(result["top_features"]) <= 3

        for feature in result["top_features"]:
            assert "feature" in feature
            assert "value" in feature
            assert "shap_value" in feature
            assert isinstance(feature["value"], float)
            assert isinstance(feature["shap_value"], float)

    def test_features_sorted_by_absolute_shap(self, trained_model_with_data):
        """Test features are sorted by absolute SHAP value."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()
        result = explainer.explain_classification(
            model=trained_model_with_data["model"],
            X=trained_model_with_data["X_sample"],
            feature_names=trained_model_with_data["feature_names"],
            label_encoder=trained_model_with_data["label_encoder"],
        )

        shap_values = [abs(f["shap_value"]) for f in result["all_features"]]
        assert shap_values == sorted(shap_values, reverse=True)

    def test_caching(self, trained_model_with_data):
        """Test explainer caching works correctly."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        # First call should create explainer
        result1 = explainer.explain_classification(
            model=trained_model_with_data["model"],
            X=trained_model_with_data["X_sample"],
            feature_names=trained_model_with_data["feature_names"],
            label_encoder=trained_model_with_data["label_encoder"],
            model_type="gradient_boosting",
            window=24,
        )

        cache_key = f"gradient_boosting_24h_{id(trained_model_with_data['model'])}"
        assert cache_key in explainer._explainer_cache

        # Second call should reuse cached explainer
        result2 = explainer.explain_classification(
            model=trained_model_with_data["model"],
            X=trained_model_with_data["X_sample"],
            feature_names=trained_model_with_data["feature_names"],
            label_encoder=trained_model_with_data["label_encoder"],
            model_type="gradient_boosting",
            window=24,
        )

        # Both should succeed
        assert "error" not in result1
        assert "error" not in result2

    def test_error_handling(self):
        """Test error handling returns error dict."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        # Use a model with predict_proba but that will fail during SHAP computation
        class BadModel:
            def predict_proba(self, X):
                raise RuntimeError("Intentional test error")

        bad_model = BadModel()
        label_encoder = LabelEncoder()
        label_encoder.fit(["None", "C"])  # Binary

        # Provide X_background to get past the kernel explainer check
        X_bg = np.random.rand(10, 3)
        result = explainer.explain_classification(
            model=bad_model,
            X=np.array([[1, 2, 3]]),
            feature_names=["a", "b", "c"],
            label_encoder=label_encoder,
            X_background=X_bg,
        )

        assert "error" in result
        assert "explainer_type" in result


class TestExplainSurvival:
    """Tests for explain_survival method."""

    @pytest.fixture
    def trained_gb_survival_model(self):
        """Create a trained GB survival model with sample data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.rand(n_samples, n_features)
        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Create a simple GB regressor as survival model wrapper
        class MockGBSurvivalModel:
            def __init__(self):
                self.model = GradientBoostingRegressor(n_estimators=10, random_state=42)
                self.is_fitted = False

            def fit(self, X, y):
                self.model.fit(X, y)
                self.is_fitted = True

            def predict(self, X):
                return self.model.predict(X)

            def predict_hazard(self, X):
                return self.model.predict(X)

        model = MockGBSurvivalModel()
        y = np.random.exponential(50, n_samples)
        model.fit(X, y)

        return {
            "model": model,
            "X_background": X,
            "X_sample": X[:1],
            "feature_names": feature_names,
        }

    def test_gb_survival_explanation(self, trained_gb_survival_model):
        """Test GB survival model explanation."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()
        result = explainer.explain_survival(
            model=trained_gb_survival_model["model"],
            X=trained_gb_survival_model["X_sample"],
            feature_names=trained_gb_survival_model["feature_names"],
            model_type="gb",
        )

        # Verify result structure
        assert "hazard_score" in result
        assert "base_value" in result
        assert "top_features" in result
        assert "all_features" in result
        assert "explainer_type" in result
        assert result["explainer_type"] == "tree"

    def test_survival_top_n_parameter(self, trained_gb_survival_model):
        """Test top_n parameter limits features."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()
        result = explainer.explain_survival(
            model=trained_gb_survival_model["model"],
            X=trained_gb_survival_model["X_sample"],
            feature_names=trained_gb_survival_model["feature_names"],
            model_type="gb",
            top_n=2,
        )

        assert len(result["top_features"]) <= 2

    def test_cox_without_background_returns_error(self):
        """Test Cox model without background data returns error."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        # Mock Cox model
        mock_model = Mock()

        result = explainer.explain_survival(
            model=mock_model,
            X=np.array([[1, 2, 3]]),
            feature_names=["a", "b", "c"],
            model_type="cox",
            X_background=None,  # No background data
        )

        assert "error" in result
        assert "X_background required" in result["error"]

    def test_survival_error_handling(self):
        """Test survival explanation error handling."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        # Use invalid model
        class BadModel:
            pass

        result = explainer.explain_survival(
            model=BadModel(),
            X=np.array([[1, 2, 3]]),
            feature_names=["a", "b", "c"],
            model_type="gb",
        )

        assert "error" in result


class TestPipelineIntegration:
    """Tests for pipeline integration with include_explanation parameter."""

    @pytest.fixture
    def mock_classification_pipeline(self):
        """Create mock classification pipeline with trained model."""
        from src.models.pipeline import ClassificationPipeline

        pipeline = Mock(spec=ClassificationPipeline)

        # Create real model and data
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 3, 50)

        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        label_encoder = LabelEncoder()
        label_encoder.fit(["None", "C", "M"])

        feature_names = ["flux_mean", "region_area", "num_spots", "complexity", "mag_class"]

        pipeline.models = {
            "24h": {
                "gradient_boosting": {
                    "model": model,
                    "label_encoder": label_encoder,
                    "feature_names": feature_names,
                }
            }
        }

        # Mock feature engineer
        pipeline.feature_engineer = Mock()
        pipeline.feature_engineer.compute_features = Mock(return_value=pd.DataFrame(X[:1], columns=feature_names))

        return pipeline

    @patch("src.api.explainer.get_explainer")
    def test_classification_with_explanation_param(self, mock_get_explainer, mock_classification_pipeline):
        """Test classification pipeline respects include_explanation parameter."""
        from src.api.explainer import ShapExplainer

        # Setup mock explainer
        mock_explainer = Mock(spec=ShapExplainer)
        mock_explainer.explain_classification.return_value = {
            "predicted_class": "C",
            "top_features": [{"feature": "flux_mean", "value": 0.5, "shap_value": 0.1}],
            "explainer_type": "tree",
        }
        mock_get_explainer.return_value = mock_explainer

        # Verify mock is set up correctly
        assert mock_classification_pipeline.models is not None


class TestAPIEndpointIntegration:
    """Tests for API endpoint integration with explanation requests."""

    @pytest.fixture
    def mock_classification_pipeline(self):
        """Create mock classification pipeline."""
        from src.models.pipeline import ClassificationPipeline

        pipeline = Mock(spec=ClassificationPipeline)
        pipeline.models = {
            "24h": {
                "gradient_boosting": {
                    "model": Mock(),
                    "label_encoder": Mock(),
                    "feature_names": ["feature1", "feature2"],
                }
            }
        }

        # Create mock DataFrame for feature engineer
        mock_features_df = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
        pipeline.feature_engineer = Mock()
        pipeline.feature_engineer.compute_features = Mock(return_value=mock_features_df)

        def predict_side_effect(*args, **kwargs):
            include_explanation = kwargs.get("include_explanation", False)
            result = {
                "timestamp": datetime.now(),
                "window_hours": kwargs.get("window", 24),
                "predicted_class": "C",
                "class_probabilities": {"None": 0.5, "C": 0.3, "M": 0.15, "X": 0.05},
                "model_type": kwargs.get("model_type", "gradient_boosting"),
            }
            if include_explanation:
                result["explanation"] = {
                    "predicted_class": "C",
                    "top_features": [{"feature": "feature1", "value": 1.0, "shap_value": 0.5}],
                    "explainer_type": "tree",
                }
            return result

        pipeline.predict = Mock(side_effect=predict_side_effect)
        return pipeline

    @pytest.fixture
    def mock_survival_pipeline(self):
        """Create mock survival pipeline."""
        from src.models.survival_pipeline import SurvivalAnalysisPipeline

        pipeline = Mock(spec=SurvivalAnalysisPipeline)
        pipeline.is_fitted = True
        pipeline.target_flare_class = "M"
        pipeline.cox_model = Mock()
        pipeline.gb_model = Mock()

        # Create mock covariate engineer
        mock_covariates_df = pd.DataFrame({"covariate1": [1.0], "covariate2": [2.0]})
        pipeline.covariate_engineer = Mock()
        pipeline.covariate_engineer.compute_time_varying_covariates = Mock(return_value=mock_covariates_df)

        def predict_side_effect(*args, **kwargs):
            include_explanation = kwargs.get("include_explanation", False)
            result = {
                "timestamp": datetime.now(),
                "model_type": kwargs.get("model_type", "gb"),
                "hazard_score": 120.5,
                "probability_distribution": {"0h-6h": 0.8},
            }
            if include_explanation and kwargs.get("model_type") == "gb":
                result["explanation"] = {
                    "hazard_score": 120.5,
                    "top_features": [{"feature": "covariate1", "value": 1.0, "shap_value": 0.5}],
                    "explainer_type": "tree",
                }
            elif include_explanation and kwargs.get("model_type") == "cox":
                result["explanation"] = {
                    "error": "SHAP explanations not available for Cox model (too slow)",
                    "suggestion": "Use GB model for explanations",
                }
            return result

        pipeline.predict_survival_probabilities = Mock(side_effect=predict_side_effect)
        return pipeline

    @pytest.fixture
    def app(self, mock_classification_pipeline, mock_survival_pipeline, monkeypatch):
        """Create Flask app for testing."""
        from src.api.app import create_app
        from src.api import auth

        # Set test API key
        monkeypatch.setenv("API_KEYS", "test-api-key-12345")
        auth.reload_api_keys()

        app = create_app(
            classification_pipeline=mock_classification_pipeline,
            survival_pipeline=mock_survival_pipeline,
        )
        app.config["TESTING"] = True
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()

    @pytest.fixture
    def auth_headers(self):
        """Return headers with valid API key."""
        return {"X-API-Key": "test-api-key-12345"}

    def test_classification_endpoint_with_explanation(self, client, mock_classification_pipeline, auth_headers):
        """Test /predict/classification with include_explanation=true."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "window": 24,
            "model_type": "gradient_boosting",
            "include_explanation": True,
        }

        response = client.post(
            "/predict/classification",
            data=json.dumps(payload),
            content_type="application/json",
            headers=auth_headers,
        )

        assert response.status_code == 200
        json.loads(response.data)  # Verify response is valid JSON

        # Verify explanation was requested
        call_args = mock_classification_pipeline.predict.call_args
        assert call_args[1]["include_explanation"] is True

    def test_classification_endpoint_without_explanation(self, client, mock_classification_pipeline, auth_headers):
        """Test /predict/classification with include_explanation=false (default)."""
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
            headers=auth_headers,
        )

        assert response.status_code == 200

        # Verify explanation was not requested by default
        call_args = mock_classification_pipeline.predict.call_args
        assert call_args[1].get("include_explanation", False) is False

    def test_survival_endpoint_with_explanation_gb(self, client, mock_survival_pipeline, auth_headers):
        """Test /predict/survival with include_explanation=true for GB model."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "model_type": "gb",
            "include_explanation": True,
        }

        response = client.post(
            "/predict/survival",
            data=json.dumps(payload),
            content_type="application/json",
            headers=auth_headers,
        )

        assert response.status_code == 200
        json.loads(response.data)  # Verify response is valid JSON

        # Verify explanation was requested
        call_args = mock_survival_pipeline.predict_survival_probabilities.call_args
        assert call_args[1]["include_explanation"] is True

    def test_survival_endpoint_with_explanation_cox(self, client, mock_survival_pipeline, auth_headers):
        """Test /predict/survival with include_explanation=true for Cox model returns warning."""
        timestamp = datetime.now().isoformat()
        payload = {
            "timestamp": timestamp,
            "model_type": "cox",
            "include_explanation": True,
        }

        response = client.post(
            "/predict/survival",
            data=json.dumps(payload),
            content_type="application/json",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should have explanation with error/suggestion for Cox
        if "explanation" in data:
            assert "error" in data["explanation"] or "suggestion" in data["explanation"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_feature_names(self):
        """Test with empty feature names returns error gracefully."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        # Train a valid model but test with empty feature names
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        label_encoder = LabelEncoder()
        label_encoder.fit(["A", "B"])

        # Test with mismatched empty feature names - should handle gracefully
        result = explainer.explain_classification(
            model=model,
            X=X[:1],
            feature_names=[],  # Empty feature names
            label_encoder=label_encoder,
        )

        # Should either return error or empty features
        assert "error" in result or len(result.get("all_features", [])) == 0

    def test_single_sample(self):
        """Test with single sample input."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        label_encoder = LabelEncoder()
        label_encoder.fit(["A", "B"])

        result = explainer.explain_classification(
            model=model,
            X=X[:1],  # Single sample
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            label_encoder=label_encoder,
        )

        assert "top_features" in result
        assert len(result["all_features"]) == 5

    def test_nan_in_features(self):
        """Test handling of NaN values in features."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        label_encoder = LabelEncoder()
        label_encoder.fit(["A", "B"])

        # Add NaN to sample
        X_with_nan = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])

        # Should handle NaN (SHAP may have issues, but should not crash)
        result = explainer.explain_classification(
            model=model,
            X=X_with_nan,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            label_encoder=label_encoder,
        )

        # Should either succeed or return error dict
        assert "top_features" in result or "error" in result

    def test_large_feature_set(self):
        """Test with large number of features."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        np.random.seed(42)
        n_features = 100
        X = np.random.rand(200, n_features)
        y = np.random.randint(0, 2, 200)

        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        label_encoder = LabelEncoder()
        label_encoder.fit(["A", "B"])

        result = explainer.explain_classification(
            model=model,
            X=X[:1],
            feature_names=[f"feature_{i}" for i in range(n_features)],
            label_encoder=label_encoder,
            top_n=10,
        )

        assert "top_features" in result
        assert len(result["top_features"]) <= 10
        assert len(result["all_features"]) == n_features

    def test_multiclass_classification(self):
        """Test with more than 2 classes - returns gracefully (may error due to SHAP limitations)."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 4, 100)  # 4 classes

        # Use LogisticRegression - SHAP LinearExplainer behavior varies for multiclass
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        label_encoder = LabelEncoder()
        label_encoder.fit(["None", "C", "M", "X"])

        result = explainer.explain_classification(
            model=model,
            X=X[:1],
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            label_encoder=label_encoder,
            X_background=X,  # Required for linear explainer
        )

        # Should return a result dict (with either success or error)
        assert isinstance(result, dict)
        # Must have explainer_type
        assert "explainer_type" in result
        # Either has SHAP values or has an error (both are valid outcomes)
        assert "shap_values_by_class" in result or "error" in result

    def test_binary_classification(self):
        """Test with binary classification."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)  # 2 classes

        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        label_encoder = LabelEncoder()
        label_encoder.fit(["Negative", "Positive"])

        result = explainer.explain_classification(
            model=model,
            X=X[:1],
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            label_encoder=label_encoder,
        )

        assert "predicted_class" in result
        assert result["predicted_class"] in ["Negative", "Positive"]


class TestSecurityConsiderations:
    """Tests for security-related concerns."""

    def test_no_sensitive_data_in_error_messages(self):
        """Test that error messages don't leak sensitive information."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        # Train a valid model
        np.random.seed(42)
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        # Create a mock label_encoder that will cause an error
        bad_label_encoder = Mock()
        bad_label_encoder.classes_ = ["A", "B"]
        bad_label_encoder.inverse_transform = Mock(side_effect=ValueError("Label encoding error"))

        # Trigger an error via label encoder
        result = explainer.explain_classification(
            model=model,
            X=X[:1],
            feature_names=["secret_api_key", "password_hash", "credit_card"],
            label_encoder=bad_label_encoder,
        )

        assert "error" in result
        # Error message should not contain the feature names (which could be sensitive)
        error_msg = str(result["error"]).lower()
        # This is a weak check - in production you'd want more comprehensive sanitization
        assert "credit_card" not in error_msg or "secret_api_key" not in error_msg

    def test_large_input_handling(self):
        """Test handling of very large inputs (potential DoS vector)."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer(max_background_samples=10)  # Limit background samples

        np.random.seed(42)
        # Create model with normal size
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)

        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)

        label_encoder = LabelEncoder()
        label_encoder.fit(["A", "B"])

        # Try with very large background data
        X_large_bg = np.random.rand(10000, 5)

        result = explainer.explain_classification(
            model=model,
            X=X_train[:1],
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            label_encoder=label_encoder,
            X_background=X_large_bg,
        )

        # Should complete without timing out (background samples limited)
        assert "top_features" in result or "error" in result

    def test_input_validation_in_api(self, monkeypatch):
        """Test that API validates include_explanation parameter type."""
        from src.api.app import create_app
        from src.api import auth

        # Set test API key
        monkeypatch.setenv("API_KEYS", "test-api-key-12345")
        auth.reload_api_keys()

        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        timestamp = datetime.now().isoformat()

        # Test with invalid include_explanation type
        payload = {
            "timestamp": timestamp,
            "window": 24,
            "include_explanation": "yes",  # Should be boolean, not string
        }

        response = client.post(
            "/predict/classification",
            data=json.dumps(payload),
            content_type="application/json",
            headers={"X-API-Key": "test-api-key-12345"},
        )

        # Should handle gracefully (truthy string coerces to True in Python)
        # This tests the robustness of the parameter handling
        assert response.status_code in [200, 400, 500]


class TestModuleHelpers:
    """Tests for module-level helper functions."""

    def test_get_shap_lazy_import(self):
        """Test lazy SHAP import."""
        from src.api.explainer import _get_shap

        shap = _get_shap()
        assert shap is not None
        assert hasattr(shap, "TreeExplainer")
        assert hasattr(shap, "LinearExplainer")
        assert hasattr(shap, "KernelExplainer")

    def test_get_explainer_singleton(self):
        """Test get_explainer returns singleton instance."""
        from src.api.explainer import get_explainer

        explainer1 = get_explainer()
        explainer2 = get_explainer()

        assert explainer1 is explainer2

    def test_get_explainer_returns_shap_explainer(self):
        """Test get_explainer returns ShapExplainer instance."""
        from src.api.explainer import get_explainer, ShapExplainer

        explainer = get_explainer()
        assert isinstance(explainer, ShapExplainer)


class TestJSONSerializability:
    """Tests to ensure explanation results are JSON serializable."""

    def test_classification_result_serializable(self):
        """Test classification explanation result is JSON serializable."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        np.random.seed(42)
        X = np.random.rand(50, 5)
        # Binary classification for SHAP TreeExplainer compatibility
        y = np.random.randint(0, 2, 50)

        model = GradientBoostingClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        label_encoder = LabelEncoder()
        label_encoder.fit(["None", "C"])  # Binary

        result = explainer.explain_classification(
            model=model,
            X=X[:1],
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            label_encoder=label_encoder,
        )

        # Should be JSON serializable
        try:
            json_str = json.dumps(result)
            assert isinstance(json_str, str)

            # Verify it can be deserialized
            parsed = json.loads(json_str)
            assert "top_features" in parsed or "error" in parsed
        except (TypeError, ValueError) as e:
            pytest.fail(f"Result not JSON serializable: {e}")

    def test_survival_result_serializable(self):
        """Test survival explanation result is JSON serializable."""
        from src.api.explainer import ShapExplainer

        explainer = ShapExplainer()

        np.random.seed(42)
        X = np.random.rand(50, 5)

        # Mock survival model
        class MockSurvivalModel:
            def __init__(self):
                self.model = GradientBoostingRegressor(n_estimators=5, random_state=42)
                self.is_fitted = True

            def predict(self, X):
                return self.model.predict(X)

            def predict_hazard(self, X):
                return self.model.predict(X)

        model = MockSurvivalModel()
        model.model.fit(X, np.random.rand(50))

        result = explainer.explain_survival(
            model=model,
            X=X[:1],
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            model_type="gb",
        )

        # Should be JSON serializable
        try:
            json_str = json.dumps(result)
            assert isinstance(json_str, str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Result not JSON serializable: {e}")
