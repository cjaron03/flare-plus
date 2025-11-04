"""flask application for model serving."""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.api.service import PredictionService
from src.models.pipeline import ClassificationPipeline
from src.models.survival_pipeline import SurvivalAnalysisPipeline
from src.data.ingestion import DataIngestionPipeline

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

    def determine_overall_status(results: Dict[str, Any]) -> str:
        """
        determine overall ingestion status based on individual data type statuses.

        args:
            results: ingestion results dict with xray_flux, solar_regions, magnetogram, flare_events

        returns:
            "success" (all succeeded), "partial" (some succeeded), or "failed" (all failed)
        """
        statuses = []

        # check each data type's status field
        for key in ["xray_flux", "solar_regions", "magnetogram", "flare_events"]:
            if key in results and results[key] is not None:
                result_data = results[key]
                status = result_data.get("status", "failed")
                statuses.append(status)

        if not statuses:
            return "failed"

        success_count = sum(1 for s in statuses if s == "success")
        total_count = len(statuses)

        if success_count == total_count:
            return "success"
        elif success_count > 0:
            return "partial"
        else:
            return "failed"

    @app.route("/ingest", methods=["POST"])
    def ingest():
        """
        data ingestion endpoint (day 2: with duration tracking).

        request body (optional):
            {
                "use_cache": true  # optional, defaults to true
            }

        returns:
            {
                "status": "success",
                "duration": 4.2,  # seconds
                "results": {...}
            }
        """
        # day 4: add partial failure logic
        start_time = datetime.utcnow()

        try:
            data = request.get_json() if request.is_json else {}
            use_cache = data.get("use_cache", True)

            logger.info("starting data ingestion")
            pipeline = DataIngestionPipeline()
            results = pipeline.run_incremental_update(use_cache=use_cache)

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # determine overall status by checking each data type's status field
            overall_status = determine_overall_status(results)

            # format results with status for each data type
            formatted_results = {}
            for key in ["xray_flux", "solar_regions", "magnetogram", "flare_events"]:
                if key in results and results[key] is not None:
                    result_data = results[key]
                    status = result_data.get("status", "failed")
                    formatted_results[key] = {"status": status}

                    # add records/error info based on status
                    if status == "success":
                        if key == "xray_flux":
                            formatted_results[key]["records"] = result_data.get("records_inserted", 0)
                        elif key == "solar_regions":
                            formatted_results[key]["records"] = result_data.get("records_inserted", 0)
                        elif key == "magnetogram":
                            formatted_results[key]["records"] = result_data.get("records_inserted", 0)
                        elif key == "flare_events":
                            formatted_results[key]["new"] = result_data.get("records_inserted", 0)
                            formatted_results[key]["duplicates"] = result_data.get("records_updated", 0)
                    else:
                        formatted_results[key]["error"] = result_data.get(
                            "error", result_data.get("error_message", "unknown error")
                        )

            # determine http status code
            http_status = 200 if overall_status in ["success", "partial"] else 500

            return (
                jsonify({"overall_status": overall_status, "duration": duration, "results": formatted_results}),
                http_status,
            )

        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"ingestion error: {e}", exc_info=True)
            return jsonify({"overall_status": "failed", "error": str(e), "duration": duration, "results": {}}), 500

    return app
