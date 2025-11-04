"""helper functions for ui dashboard: connection, formatting, throttling."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

import requests
import joblib

from src.models.pipeline import ClassificationPipeline
from src.models.survival_pipeline import SurvivalAnalysisPipeline

logger = logging.getLogger(__name__)


def get_prediction_service(
    api_url: str,
    classification_model_path: Optional[str] = None,
    survival_model_path: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """
    try api first, fallback to direct model loading.

    args:
        api_url: api server url (e.g., "http://127.0.0.1:5000")
        classification_model_path: path to classification model (joblib)
        survival_model_path: path to survival model (joblib)

    returns:
        tuple: (connection_mode, api_url_or_none, loaded_pipelines_dict)
            connection_mode: "api", "direct", or "error"
            api_url_or_none: api url if using api, None otherwise
            loaded_pipelines_dict: {"classification": pipeline, "survival": pipeline} if direct
    """
    # try api connection first
    try:
        health_url = f"{api_url}/health"
        response = requests.get(health_url, timeout=2)
        if response.status_code == 200:
            logger.info(f"connected to api at {api_url}")
            return "api", api_url, None
    except requests.Timeout:
        logger.warning(f"api timeout at {api_url}, falling back to direct loading")
    except requests.ConnectionError:
        logger.warning(f"api connection failed at {api_url}, falling back to direct loading")
    except requests.RequestException as e:
        logger.warning(f"api request error: {e}, falling back to direct loading")

    # fallback: direct model loading
    loaded_pipelines = {}
    errors = []

    if classification_model_path:
        try:
            logger.info(f"loading classification model from {classification_model_path}")
            pipeline_data = joblib.load(classification_model_path)
            pipeline = ClassificationPipeline()
            pipeline.models = pipeline_data.get("models", {})
            pipeline.evaluation_results = pipeline_data.get("evaluation_results", {})
            loaded_pipelines["classification"] = pipeline
            logger.info("classification model loaded successfully")
        except Exception as e:
            logger.error(f"failed to load classification model: {e}")
            errors.append(f"classification: {e}")

    if survival_model_path:
        try:
            logger.info(f"loading survival model from {survival_model_path}")
            pipeline_data = joblib.load(survival_model_path)
            pipeline = SurvivalAnalysisPipeline(
                target_flare_class=pipeline_data.get("target_flare_class", "X"),
                max_time_hours=pipeline_data.get("max_time_hours", 168),
            )
            pipeline.is_fitted = pipeline_data.get("is_fitted", False)
            if pipeline_data.get("cox_model"):
                pipeline.cox_model = pipeline_data["cox_model"]
            if pipeline_data.get("gb_model"):
                pipeline.gb_model = pipeline_data["gb_model"]
            loaded_pipelines["survival"] = pipeline
            logger.info("survival model loaded successfully")
        except Exception as e:
            logger.error(f"failed to load survival model: {e}")
            errors.append(f"survival: {e}")

    if not loaded_pipelines:
        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"failed to load any models: {error_msg}")
            return "error", None, None
        else:
            logger.warning("no model paths provided, cannot use direct loading")
            return "error", None, None

    logger.info(f"using direct model loading ({len(loaded_pipelines)} models)")
    return "direct", None, loaded_pipelines


def get_api_model_status(api_url: str) -> Dict[str, bool]:
    """
    query api health endpoint to get model availability status.

    args:
        api_url: api base url

    returns:
        dict with model availability: {"classification": bool, "survival": bool}
    """
    try:
        health_url = f"{api_url}/health"
        response = requests.get(health_url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return {
                "classification": data.get("classification_available", False),
                "survival": data.get("survival_available", False),
            }
    except Exception as e:
        logger.warning(f"failed to query api health: {e}")
    return {"classification": False, "survival": False}


def should_refresh(last_refresh: Optional[datetime], min_interval_minutes: int = 5) -> bool:
    """
    check if refresh should occur based on throttling interval.

    args:
        last_refresh: last refresh timestamp
        min_interval_minutes: minimum minutes between refreshes

    returns:
        true if refresh should occur
    """
    if last_refresh is None:
        return True

    time_since_refresh = datetime.now() - last_refresh
    return time_since_refresh > timedelta(minutes=min_interval_minutes)


def format_classification_prediction(prediction: Dict[str, Any]) -> str:
    """
    format classification prediction for display.

    args:
        prediction: prediction dict from pipeline or api

    returns:
        formatted string
    """
    predicted_class = prediction.get("predicted_class", "Unknown")
    class_probs = prediction.get("class_probabilities", {})
    window = prediction.get("window_hours", 24)

    if not class_probs:
        return f"Prediction unavailable for {window}h window"

    # format probabilities
    prob_lines = []
    for class_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
        prob_lines.append(f"  {class_name}: {prob*100:.1f}%")

    result = f"Predicted class: {predicted_class} (next {window}h)\n\n"
    result += "Interpretation: This predicts the maximum flare class expected.\n"
    result += "If 'C' is predicted, it means at least C-class (may include M or X).\n\n"
    result += "Probabilities:\n"
    result += "\n".join(prob_lines)

    return result


def format_survival_plain_language(prediction: Dict[str, Any]) -> str:
    """
    convert survival probabilities to plain language.

    args:
        prediction: survival prediction dict

    returns:
        plain language explanation
    """
    prob_dist = prediction.get("probability_distribution", {})
    target_class = prediction.get("target_flare_class", "X")
    hazard_score = prediction.get("hazard_score")

    if not prob_dist:
        return "Unable to generate survival prediction"

    # find highest probability bucket
    if len(prob_dist) == 0:
        return "No probability distribution available"

    max_item = max(prob_dist.items(), key=lambda x: x[1])
    bucket_range, max_prob = max_item

    # format bucket range
    bucket_str = bucket_range.replace("-", " to ").replace("h", "h")

    # create plain language statement
    statement = (
        f"There is a {max_prob*100:.1f}% chance of a {target_class}-class flare "
        f"within {bucket_str}.\n\n"
        f"Note: This survival model predicts C-class flare timing. "
        f"For M/X-class flares, use the Classification model."
    )

    # add hazard score interpretation if available
    if hazard_score is not None:
        if hazard_score > 1.5:
            hazard_desc = "high"
        elif hazard_score > 1.0:
            hazard_desc = "moderate"
        else:
            hazard_desc = "low"

        statement += f"\n\nHazard score: {hazard_score:.2f} ({hazard_desc} risk)"

    return statement


def format_survival_probability_distribution(prediction: Dict[str, Any]) -> str:
    """
    format survival probability distribution as text.

    args:
        prediction: survival prediction dict

    returns:
        formatted distribution text
    """
    prob_dist = prediction.get("probability_distribution", {})

    if not prob_dist:
        return "No probability distribution available"

    lines = ["Probability distribution over time buckets:"]
    lines.append("-" * 50)

    for bucket_range, prob in sorted(prob_dist.items(), key=lambda x: _parse_bucket_range(x[0])):
        lines.append(f"  {bucket_range:15s}  {prob*100:6.2f}%")

    return "\n".join(lines)


def _parse_bucket_range(bucket_str: str) -> int:
    """parse bucket range string to integer for sorting (e.g., '0h-6h' -> 0)."""
    try:
        start = bucket_str.split("-")[0].replace("h", "")
        return int(start)
    except (ValueError, IndexError):
        return 0


def make_api_request(
    api_url: str,
    endpoint: str,
    method: str = "GET",
    json_data: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    make api request with error handling.

    args:
        api_url: base api url
        endpoint: api endpoint path
        method: http method
        json_data: optional json data for post requests
        timeout: request timeout

    returns:
        tuple: (success, response_data, error_message)
    """
    try:
        url = f"{api_url}{endpoint}"
        if method.upper() == "POST":
            response = requests.post(url, json=json_data, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)

        response.raise_for_status()
        return True, response.json(), None

    except requests.Timeout:
        error_msg = f"request timeout for {endpoint}"
        logger.error(error_msg)
        return False, None, error_msg
    except requests.ConnectionError:
        error_msg = f"connection error for {endpoint}"
        logger.error(error_msg)
        return False, None, error_msg
    except requests.RequestException as e:
        error_msg = f"request error: {e}"
        logger.error(error_msg)
        return False, None, error_msg
    except ValueError as e:  # json decode error
        error_msg = f"invalid json response: {e}"
        logger.error(error_msg)
        return False, None, error_msg
