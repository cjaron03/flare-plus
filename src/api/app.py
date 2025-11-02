"""flask application for model serving."""

import logging
from datetime import datetime
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.api.service import PredictionService
from src.models.pipeline import ClassificationPipeline
from src.models.survival_pipeline import SurvivalAnalysisPipeline

logger = logging.getLogger(__name__)


def create_app(
    classification_pipeline: Optional[ClassificationPipeline] = None,
    survival_pipeline: Optional[SurvivalAnalysisPipeline] = None,
) -> Flask:
    """
    create flask application.

    args:
        classification_pipeline: optional pre-trained classification pipeline
        survival_pipeline: optional pre-trained survival analysis pipeline

    returns:
        configured flask app
    """
    app = Flask(__name__)
    CORS(app)  # enable cross-origin requests

    # initialize prediction service
    service = PredictionService(
        classification_pipeline=classification_pipeline,
        survival_pipeline=survival_pipeline,
    )

    @app.route("/health", methods=["GET"])
    def health():
        """health check endpoint."""
        status = service.health_check()
        return jsonify(status), 200

    @app.route("/predict/classification", methods=["POST"])
    def predict_classification():
        """
        classification prediction endpoint.

        request body:
            {
                "timestamp": "2024-01-01T12:00:00",
                "window": 24,
                "model_type": "gradient_boosting",
                "region_number": 1234  # optional
            }
        """
        try:
            data = request.get_json()

            if not data or "timestamp" not in data:
                return jsonify({"error": "timestamp is required"}), 400

            timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            window = data.get("window", 24)
            model_type = data.get("model_type", "gradient_boosting")
            region_number = data.get("region_number")

            if window not in [24, 48]:
                return jsonify({"error": "window must be 24 or 48"}), 400

            result = service.predict_classification(
                timestamp=timestamp,
                window=window,
                model_type=model_type,
                region_number=region_number,
            )

            return jsonify(result), 200

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"classification prediction error: {e}", exc_info=True)
            return jsonify({"error": "internal server error"}), 500

    @app.route("/predict/survival", methods=["POST"])
    def predict_survival():
        """
        survival prediction endpoint.

        request body:
            {
                "timestamp": "2024-01-01T12:00:00",
                "model_type": "cox",
                "region_number": 1234,  # optional
                "time_buckets": [0, 6, 12, 24]  # optional
            }
        """
        try:
            data = request.get_json()

            if not data or "timestamp" not in data:
                return jsonify({"error": "timestamp is required"}), 400

            timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            model_type = data.get("model_type", "cox")
            region_number = data.get("region_number")
            time_buckets = data.get("time_buckets")

            if model_type not in ["cox", "gb"]:
                return jsonify({"error": "model_type must be 'cox' or 'gb'"}), 400

            result = service.predict_survival(
                timestamp=timestamp,
                model_type=model_type,
                region_number=region_number,
                time_buckets=time_buckets,
            )

            return jsonify(result), 200

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"survival prediction error: {e}", exc_info=True)
            return jsonify({"error": "internal server error"}), 500

    @app.route("/predict/all", methods=["POST"])
    def predict_all():
        """
        combined prediction endpoint (classification + survival).

        request body:
            {
                "timestamp": "2024-01-01T12:00:00",
                "region_number": 1234,  # optional
                "classification_windows": [24, 48],  # optional
                "survival_model_type": "cox"  # optional
            }
        """
        try:
            data = request.get_json()

            if not data or "timestamp" not in data:
                return jsonify({"error": "timestamp is required"}), 400

            timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            region_number = data.get("region_number")
            classification_windows = data.get("classification_windows", [24, 48])
            survival_model_type = data.get("survival_model_type", "cox")

            result = service.predict_all(
                timestamp=timestamp,
                region_number=region_number,
                classification_windows=classification_windows,
                survival_model_type=survival_model_type,
            )

            return jsonify(result), 200

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"combined prediction error: {e}", exc_info=True)
            return jsonify({"error": "internal server error"}), 500

    return app
