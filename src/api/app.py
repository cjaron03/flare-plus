"""flask application for model serving."""

import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional, Dict, Any

import sentry_sdk
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sentry_sdk.integrations.flask import FlaskIntegration

from src.api.service import PredictionService
from src.api.auth import require_api_key
from src.models.pipeline import ClassificationPipeline
from src.models.survival_pipeline import SurvivalAnalysisPipeline
from src.data.ingestion import DataIngestionPipeline
from src.data.database import get_database
from src.data.schema import SystemValidationLog
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# initialize sentry if dsn is configured
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[FlaskIntegration()],
        traces_sample_rate=0.1,  # 10% of transactions for performance monitoring
        profiles_sample_rate=0.1,  # 10% of transactions for profiling
        environment=os.getenv("ENVIRONMENT", "development"),
    )
    logger.info("sentry initialized for error monitoring")
else:
    logger.info("sentry dsn not configured, skipping initialization")


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

    # initialize rate limiter
    # uses in-memory storage by default (upgrade to Redis for production scaling)
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["100 per hour"],
        storage_uri="memory://",
    )

    # initialize prediction service
    service = PredictionService(
        classification_pipeline=classification_pipeline,
        survival_pipeline=survival_pipeline,
    )

    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # CSP for API - restrict to self
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

    @app.route("/health", methods=["GET"])
    @limiter.exempt  # no rate limit for health checks
    def health():
        """health check endpoint."""
        status = service.health_check()
        return jsonify(status), 200

    @app.route("/validate/system", methods=["POST"])
    @require_api_key
    def trigger_system_validation():
        """trigger end-to-end system validation run."""

        def truncate(text: str, limit: int = 6000) -> str:
            if text is None:
                return ""
            if len(text) <= limit:
                return text
            return text[:limit] + "\n...[truncated]..."

        try:
            payload = request.get_json(silent=True) or {}
            initiated_by = payload.get("initiated_by") or "api"

            env = os.environ.copy()
            env["VALIDATION_INITIATED_BY"] = initiated_by
            cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_system.py")]

            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=900,
                env=env,
            )

            response_data = {
                "returncode": completed.returncode,
                "stdout": truncate(completed.stdout),
                "stderr": truncate(completed.stderr),
            }

            status_code = 200 if completed.returncode == 0 else 400
            return jsonify(response_data), status_code
        except subprocess.TimeoutExpired:
            return jsonify({"error": "validation timed out"}), 504
        except Exception as e:
            logger.error(f"system validation trigger failed: {e}", exc_info=True)
            return jsonify({"error": "failed to trigger validation"}), 500

    @app.route("/validation/logs", methods=["GET"])
    @require_api_key
    def list_validation_logs():
        """return recent validation runs for admin dashboard."""
        try:
            limit_param = request.args.get("limit", default="10")
            try:
                limit = min(max(int(limit_param), 1), 100)
            except ValueError:
                limit = 10

            db = get_database()
            with db.get_session() as session:
                logs = (
                    session.query(SystemValidationLog)
                    .order_by(SystemValidationLog.run_timestamp.desc())
                    .limit(limit)
                    .all()
                )

                payload = [
                    {
                        "id": log.id,
                        "run_timestamp": log.run_timestamp.isoformat(),
                        "status": log.status,
                        "validation_type": log.validation_type,
                        "guardrail_triggered": log.guardrail_triggered,
                        "guardrail_reason": log.guardrail_reason,
                        "initiated_by": log.initiated_by,
                    }
                    for log in logs
                ]

            return jsonify({"logs": payload}), 200
        except Exception as e:
            logger.error(f"failed to fetch validation logs: {e}", exc_info=True)
            return jsonify({"error": "failed to fetch validation logs"}), 500

    @app.route("/predict/classification", methods=["POST"])
    @limiter.limit("20 per minute")  # stricter limit for compute-heavy endpoint
    @require_api_key
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
            # normalize to timezone-naive UTC for pandas compatibility
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)
            window = data.get("window", 24)
            model_type = data.get("model_type", "gradient_boosting")
            region_number = data.get("region_number")

            if window not in [24, 48]:
                return jsonify({"error": "window must be 24 or 48"}), 400

            include_explanation = data.get("include_explanation", False)

            result = service.predict_classification(
                timestamp=timestamp,
                window=window,
                model_type=model_type,
                region_number=region_number,
                include_explanation=include_explanation,
            )

            return jsonify(result), 200

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"classification prediction error: {e}", exc_info=True)
            return jsonify({"error": "internal server error"}), 500

    @app.route("/predict/survival", methods=["POST"])
    @limiter.limit("20 per minute")  # stricter limit for compute-heavy endpoint
    @require_api_key
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
            # normalize to timezone-naive UTC for pandas compatibility
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)
            model_type = data.get("model_type", "cox")
            region_number = data.get("region_number")
            time_buckets = data.get("time_buckets")

            if model_type not in ["cox", "gb"]:
                return jsonify({"error": "model_type must be 'cox' or 'gb'"}), 400

            include_explanation = data.get("include_explanation", False)

            result = service.predict_survival(
                timestamp=timestamp,
                model_type=model_type,
                region_number=region_number,
                time_buckets=time_buckets,
                include_explanation=include_explanation,
            )

            return jsonify(result), 200

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"survival prediction error: {e}", exc_info=True)
            return jsonify({"error": "internal server error"}), 500

    @app.route("/predict/all", methods=["POST"])
    @limiter.limit("10 per minute")  # strictest limit - runs both models
    @require_api_key
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
    @limiter.limit("5 per minute")  # limit expensive ingestion operations
    @require_api_key
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
