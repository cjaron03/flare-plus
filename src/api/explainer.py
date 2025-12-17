"""SHAP explainability for flare prediction models."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# lazy import shap to avoid startup overhead
_shap = None


def _get_shap():
    """Lazy load SHAP library."""
    global _shap
    if _shap is None:
        try:
            import shap

            _shap = shap
        except ImportError:
            logger.error("shap library not installed. run: pip install shap")
            raise
    return _shap


def get_base_estimator(model: Any) -> Any:
    """
    Extract base estimator from sklearn wrappers.

    Handles:
    - imblearn.pipeline.Pipeline (SMOTE pipelines)
    - sklearn.calibration.CalibratedClassifierCV
    - sklearn.pipeline.Pipeline
    - Raw estimators (returned as-is)

    args:
        model: sklearn model or pipeline

    returns:
        base estimator
    """
    # check for imblearn or sklearn pipeline
    if hasattr(model, "named_steps"):
        # pipeline with named steps
        if "model" in model.named_steps:
            return model.named_steps["model"]
        # try to get the last step
        steps = list(model.named_steps.values())
        if steps:
            return steps[-1]

    # check for calibrated classifier
    if hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV - get base estimator from first calibrated classifier
        if model.calibrated_classifiers_:
            return model.calibrated_classifiers_[0].estimator

    # already a base estimator
    return model


def get_explainer_type(model: Any) -> str:
    """
    Determine the appropriate SHAP explainer type for a model.

    args:
        model: sklearn model

    returns:
        explainer type string ('tree', 'linear', 'kernel')
    """
    base_model = get_base_estimator(model)
    model_name = type(base_model).__name__

    # tree-based models
    tree_models = [
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
    ]

    # linear models
    linear_models = [
        "LogisticRegression",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SGDClassifier",
        "SGDRegressor",
    ]

    if model_name in tree_models:
        return "tree"
    elif model_name in linear_models:
        return "linear"
    else:
        return "kernel"


class ShapExplainer:
    """SHAP explanations for flare prediction models."""

    def __init__(self, max_background_samples: int = 100):
        """
        Initialize SHAP explainer.

        args:
            max_background_samples: maximum background samples for KernelExplainer
        """
        self.max_background_samples = max_background_samples
        self._explainer_cache: Dict[str, Any] = {}
        self._background_cache: Dict[str, np.ndarray] = {}

    def _get_cache_key(self, model: Any, model_type: str, window: int) -> str:
        """Generate cache key for explainer."""
        return f"{model_type}_{window}h_{id(model)}"

    def _create_explainer(
        self,
        model: Any,
        X_background: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Any:
        """
        Create appropriate SHAP explainer for model.

        args:
            model: sklearn model
            X_background: background data for KernelExplainer
            feature_names: feature names for display

        returns:
            SHAP explainer instance
        """
        shap = _get_shap()
        base_model = get_base_estimator(model)
        explainer_type = get_explainer_type(model)

        logger.info(f"creating {explainer_type} explainer for {type(base_model).__name__}")

        if explainer_type == "tree":
            return shap.TreeExplainer(base_model)

        elif explainer_type == "linear":
            if X_background is not None:
                return shap.LinearExplainer(base_model, X_background)
            else:
                # fallback to Explainer which auto-detects
                return shap.Explainer(base_model)

        else:  # kernel
            if X_background is None:
                raise ValueError("X_background required for KernelExplainer")
            # use k-means to summarize background data
            background = shap.kmeans(X_background, min(self.max_background_samples, len(X_background)))
            return shap.KernelExplainer(base_model.predict_proba, background)

    def explain_classification(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        label_encoder: Any,
        model_type: str = "gradient_boosting",
        window: int = 24,
        X_background: Optional[np.ndarray] = None,
        top_n: int = 15,
    ) -> Dict[str, Any]:
        """
        Compute SHAP explanation for classification prediction.

        args:
            model: trained sklearn model
            X: feature values (1 sample)
            feature_names: list of feature names
            label_encoder: label encoder for class names
            model_type: model type string
            window: prediction window hours
            X_background: background data for explainers that need it
            top_n: number of top features to return

        returns:
            dict with SHAP explanation data
        """
        cache_key = self._get_cache_key(model, model_type, window)

        # get or create explainer
        if cache_key not in self._explainer_cache:
            self._explainer_cache[cache_key] = self._create_explainer(model, X_background, feature_names)
        explainer = self._explainer_cache[cache_key]

        try:
            # compute SHAP values
            shap_values = explainer.shap_values(X)
            expected_value = explainer.expected_value

            # handle different SHAP output formats
            # TreeExplainer returns list of arrays for multiclass
            # LinearExplainer may return single array
            classes = list(label_encoder.classes_)

            if isinstance(shap_values, list):
                # multiclass: one array per class
                shap_by_class = {}
                base_by_class = {}
                for i, class_name in enumerate(classes):
                    shap_by_class[str(class_name)] = shap_values[i][0].tolist()
                    if isinstance(expected_value, (list, np.ndarray)):
                        base_by_class[str(class_name)] = float(expected_value[i])
                    else:
                        base_by_class[str(class_name)] = float(expected_value)
            else:
                # binary or single output
                if len(classes) == 2:
                    # binary classification - shap_values is for positive class
                    shap_by_class = {str(classes[1]): shap_values[0].tolist()}
                    base_by_class = {str(classes[1]): float(expected_value)}
                else:
                    shap_by_class = {str(classes[0]): shap_values[0].tolist()}
                    base_by_class = {str(classes[0]): float(expected_value)}

            # get predicted class
            y_pred = model.predict(X)[0]
            predicted_class = str(label_encoder.inverse_transform([y_pred])[0])

            # get SHAP values for predicted class
            if predicted_class in shap_by_class:
                shap_for_predicted = shap_by_class[predicted_class]
            else:
                # fallback to first class
                shap_for_predicted = list(shap_by_class.values())[0]

            # build feature contributions sorted by absolute value
            contributions = []
            for i, (name, shap_val) in enumerate(zip(feature_names, shap_for_predicted)):
                contributions.append(
                    {
                        "feature": name,
                        "value": float(X[0, i]),
                        "shap_value": float(shap_val),
                    }
                )

            # sort by absolute SHAP value
            contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            # get prediction probability
            y_prob = model.predict_proba(X)[0]
            predicted_prob = float(y_prob[list(label_encoder.classes_).index(predicted_class)])

            return {
                "predicted_class": predicted_class,
                "predicted_probability": predicted_prob,
                "base_value": base_by_class.get(predicted_class, list(base_by_class.values())[0]),
                "top_features": contributions[:top_n],
                "all_features": contributions,
                "shap_values_by_class": shap_by_class,
                "base_values_by_class": base_by_class,
                "explainer_type": get_explainer_type(model),
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}", exc_info=True)
            return {"error": str(e), "explainer_type": get_explainer_type(model)}

    def explain_survival(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        model_type: str = "gb",
        X_background: Optional[np.ndarray] = None,
        top_n: int = 15,
    ) -> Dict[str, Any]:
        """
        Compute SHAP explanation for survival model (hazard prediction).

        args:
            model: trained survival model (GB or Cox wrapper)
            X: feature values (1 sample)
            feature_names: list of feature names
            model_type: 'gb' or 'cox'
            X_background: background data for explainers
            top_n: number of top features to return

        returns:
            dict with SHAP explanation data
        """
        shap = _get_shap()

        try:
            if model_type == "gb":
                # GB survival model - use underlying regressor
                if hasattr(model, "model"):
                    base_model = model.model
                else:
                    base_model = model

                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(X)
                expected_value = explainer.expected_value

                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

            else:  # cox model
                # Cox model - need to use KernelExplainer with predict function
                if X_background is None:
                    return {
                        "error": "X_background required for Cox model explanation",
                        "explainer_type": "kernel",
                    }

                # create prediction function for partial hazard
                def predict_fn(X_batch):
                    df = pd.DataFrame(X_batch, columns=feature_names)
                    return model.predict_partial_hazard(df)

                background = shap.kmeans(X_background, min(self.max_background_samples, len(X_background)))
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(X)
                expected_value = explainer.expected_value

            # build feature contributions
            contributions = []
            shap_arr = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            for i, (name, shap_val) in enumerate(zip(feature_names, shap_arr)):
                contributions.append(
                    {
                        "feature": name,
                        "value": float(X[0, i]),
                        "shap_value": float(shap_val),
                    }
                )

            # sort by absolute SHAP value
            contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            # get hazard prediction
            if hasattr(model, "predict_hazard"):
                hazard = float(model.predict_hazard(X)[0])
            elif hasattr(model, "predict"):
                hazard = float(model.predict(X)[0])
            else:
                hazard = None

            return {
                "hazard_score": hazard,
                "base_value": (
                    float(expected_value) if not isinstance(expected_value, np.ndarray) else float(expected_value[0])
                ),
                "top_features": contributions[:top_n],
                "all_features": contributions,
                "explainer_type": "tree" if model_type == "gb" else "kernel",
            }

        except Exception as e:
            logger.error(f"survival SHAP explanation failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "explainer_type": "tree" if model_type == "gb" else "kernel",
            }

    def clear_cache(self):
        """Clear explainer cache."""
        self._explainer_cache.clear()
        self._background_cache.clear()
        logger.info("SHAP explainer cache cleared")


# module-level instance for reuse
_default_explainer: Optional[ShapExplainer] = None


def get_explainer() -> ShapExplainer:
    """Get or create default SHAP explainer instance."""
    global _default_explainer
    if _default_explainer is None:
        _default_explainer = ShapExplainer()
    return _default_explainer
