"""FastAPI-based UI backend that powers the Svelte dashboard."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.config import AdminConfig, CONFIG
from src.ui.utils.admin import fetch_validation_logs, trigger_validation_via_api
from src.ui.utils.data_queries import (
    calculate_data_freshness,
    get_latest_data_timestamps,
    query_historical_flares,
)
from src.ui.utils.helpers import (
    format_classification_prediction,
    format_survival_plain_language,
    format_survival_probability_distribution,
    get_api_health_status,
    get_api_model_status,
    get_prediction_service,
    make_api_request,
    should_refresh,
)
from src.ui.utils.ingestion import (
    can_trigger_ingestion,
    format_ingestion_summary,
    get_ingestion_state,
    run_ingestion_via_api,
    set_ingestion_in_progress,
    set_last_ingestion_time,
)

logger = logging.getLogger(__name__)


# Request models ----------------------------------------------------------------


class ClassificationRequest(BaseModel):
    """Payload for classification predictions."""

    timestamp: Optional[datetime] = None
    window: int = Field(default=24)
    region_number: Optional[int] = Field(default=None, alias="regionNumber")
    model_type: str = Field(default="gradient_boosting", alias="modelType")
    force_refresh: bool = Field(default=False, alias="forceRefresh")

    @field_validator("window")
    @classmethod
    def validate_window(cls, v: int) -> int:
        """validate window is 24 or 48."""
        if v not in [24, 48]:
            raise ValueError("window must be 24 or 48")
        return v


class SurvivalRequest(BaseModel):
    """Payload for survival predictions."""

    timestamp: Optional[datetime] = None
    region_number: Optional[int] = Field(default=None, alias="regionNumber")
    model_type: str = Field(default="cox", alias="modelType")
    force_refresh: bool = Field(default=False, alias="forceRefresh")


class TimelineRequest(BaseModel):
    """Payload for timeline queries."""

    start: Optional[datetime] = None
    end: Optional[datetime] = None
    min_class: str = Field(default="B", alias="minClass")
    region_number: Optional[int] = Field(default=None, alias="regionNumber")
    force_refresh: bool = Field(default=False, alias="forceRefresh")


class IngestionRequest(BaseModel):
    """Payload for ingestion trigger."""

    use_cache: bool = Field(default=True, alias="useCache")


class LoginRequest(BaseModel):
    """Payload for admin login."""

    username: str
    password: str


# Internal state ----------------------------------------------------------------


@dataclass
class ConnectionSnapshot:
    """Serialized connection state for the frontend."""

    mode: str
    apiUrl: Optional[str]
    modelsLoaded: List[str]
    confidence: Optional[str]
    guardrailActive: bool
    guardrailReason: Optional[str]
    lastValidation: Optional[str]
    adminIndicator: str
    adminDisabledReason: str


class UIState:
    """Holds shared UI state such as connection info and refresh throttles."""

    def __init__(
        self,
        api_url: str,
        classification_model_path: Optional[str],
        survival_model_path: Optional[str],
    ):
        self._api_url = api_url
        self._classification_model_path = classification_model_path
        self._survival_model_path = survival_model_path
        (
            self.connection_mode,
            self.api_url,
            self.pipelines,
        ) = get_prediction_service(api_url, classification_model_path, survival_model_path)

        self._refresh_lock = Lock()
        self._last_refresh: Optional[datetime] = None
        self._connection_lock = Lock()
        self._last_connection_check: Optional[datetime] = None

    def refresh_connection(self, force: bool = False):
        """
        re-check api connection and update connection mode.

        uses smart refresh strategy to avoid expensive model reloading:
        - if in error mode: always refresh
        - if in stable mode (api/direct): only refresh if >30s since last check or forced
        - if in api mode: use lightweight health check instead of full reload

        args:
            force: bypass time-based throttling
        """
        with self._connection_lock:
            now = datetime.utcnow()

            # determine if we should refresh
            should_refresh_now = force
            if not should_refresh_now:
                if self.connection_mode == "error":
                    # always try to recover from error state
                    should_refresh_now = True
                elif self._last_connection_check is None:
                    # first check after init
                    should_refresh_now = True
                else:
                    # in stable mode, only refresh if enough time passed
                    elapsed = (now - self._last_connection_check).total_seconds()
                    should_refresh_now = elapsed >= 30

            if not should_refresh_now:
                return

            # if already in api mode, try lightweight health check first
            if self.connection_mode == "api" and self.api_url:
                health = get_api_health_status(self.api_url)
                if health:
                    # api still healthy, no need to reload
                    self._last_connection_check = now
                    return
                # api became unhealthy, fall through to full reload

            # perform full connection refresh (may reload models from disk)
            (
                self.connection_mode,
                self.api_url,
                self.pipelines,
            ) = get_prediction_service(self._api_url, self._classification_model_path, self._survival_model_path)
            self._last_connection_check = now

    def snapshot(self) -> ConnectionSnapshot:
        """Return a serializable snapshot used by multiple endpoints."""
        confidence = None
        guardrail_active = False
        guardrail_reason = None
        last_validation = None
        models_loaded: List[str] = []

        if self.connection_mode == "api" and self.api_url:
            status = get_api_model_status(self.api_url)
            confidence = status.get("confidence_level")
            guardrail_active = bool(status.get("survival_guardrail"))
            guardrail_reason = status.get("guardrail_reason")
            last_validation = status.get("last_validation")
            models = []
            if status.get("classification"):
                models.append("Classification")
            if status.get("survival"):
                models.append("Survival")
            models_loaded = models
        elif self.pipelines:
            models_loaded = [name.title() for name in self.pipelines.keys()]

        return ConnectionSnapshot(
            mode=self.connection_mode,
            apiUrl=self.api_url,
            modelsLoaded=models_loaded,
            confidence=confidence.title() if isinstance(confidence, str) else confidence,
            guardrailActive=guardrail_active,
            guardrailReason=guardrail_reason,
            lastValidation=last_validation,
            adminIndicator=AdminConfig.status_indicator(),
            adminDisabledReason=AdminConfig.disabled_reason(),
        )

    def last_refresh(self) -> Optional[datetime]:
        with self._refresh_lock:
            return self._last_refresh

    def record_refresh(self):
        with self._refresh_lock:
            self._last_refresh = datetime.utcnow()


# Helper utilities ---------------------------------------------------------------


def _serialize_data_freshness() -> Dict[str, Any]:
    """Return structured data freshness payload."""
    timestamps = get_latest_data_timestamps()
    sections = {}
    for key, label in [
        ("flux", "Flux"),
        ("regions", "Regions"),
        ("flares", "Flares"),
    ]:
        latest = timestamps.get(f"{key}_latest")
        count = timestamps.get(f"{key}_count", 0)
        freshness = calculate_data_freshness(latest)
        sections[key] = {
            "label": label,
            "latest": latest.isoformat() if hasattr(latest, "isoformat") else None,
            "count": count,
            "hoursAgo": freshness.get("hours_ago"),
            "status": freshness.get("status"),
            "color": freshness.get("color"),
        }
    return sections


def _throttle_message(last_refresh: Optional[datetime], min_minutes: int) -> str:
    """Generate throttle copy similar to the Gradio implementation."""
    if last_refresh is None:
        return ""

    elapsed = datetime.utcnow() - last_refresh
    if elapsed >= timedelta(minutes=min_minutes):
        return ""

    minutes_remaining = max(0, int((timedelta(minutes=min_minutes) - elapsed).total_seconds() / 60))
    return f"Refresh throttled (wait {minutes_remaining} more minutes or use force refresh)."


def _serialize_classification_result(prediction: Dict[str, Any], window: int) -> Dict[str, Any]:
    """Return structured classification payload."""
    class_probs = prediction.get("class_probabilities", {}) or {}
    ordered = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in ordered]
    values = [round(item[1] * 100, 2) for item in ordered]

    return {
        "predictedClass": prediction.get("predicted_class", "Unknown"),
        "windowHours": prediction.get("window_hours", window),
        "probabilities": class_probs,
        "orderedLabels": labels,
        "orderedValues": values,
        "text": format_classification_prediction(prediction),
    }


def _serialize_survival_result(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Return structured survival payload."""
    prob_dist = prediction.get("probability_distribution", {}) or {}
    ordered = sorted(prob_dist.items(), key=lambda item: _parse_bucket(item[0]))

    return {
        "plainText": format_survival_plain_language(prediction),
        "distributionText": format_survival_probability_distribution(prediction),
        "probabilityDistribution": prob_dist,
        "orderedBuckets": [item[0] for item in ordered],
        "orderedValues": [round(item[1] * 100, 2) for item in ordered],
        "survivalCurve": prediction.get("survival_function", {}),
        "targetClass": prediction.get("target_flare_class", "M"),
        "hazardScore": prediction.get("hazard_score"),
    }


def _parse_bucket(bucket: str) -> int:
    """Utility for bucket sorting."""
    try:
        return int(bucket.split("h")[0].split("-")[0])
    except (ValueError, IndexError):
        return 0


def _format_guardrail_status(health: Optional[Dict[str, Any]]) -> str:
    """Human readable guardrail summary for the admin panel."""
    if not health:
        return "Unable to retrieve API health status."

    validation = health.get("validation") or {}
    guardrail_active = health.get("survival_guardrail", False)
    confidence = (health.get("confidence_level") or "unknown").title()
    service_status = (health.get("status") or "unknown").title()
    last_validation = validation.get("run_timestamp", "N/A")
    last_validation_status = (validation.get("status") or "unknown").title()
    guardrail_reason = validation.get("guardrail_reason")

    lines: List[str] = [
        f"**Service Status**: {service_status}",
        f"**Confidence**: {confidence}",
        f"**Last Validation**: {last_validation} ({last_validation_status})",
    ]

    if guardrail_active:
        lines.append("**Guardrail**: Active")
        if guardrail_reason:
            lines.append(f"**Reason**: {guardrail_reason}")
    else:
        lines.append("**Guardrail**: Not triggered")

    return "\n".join(lines)


def _serialize_validation_rows(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert validation log list to frontend friendly rows."""
    rows = []
    for log in logs:
        rows.append(
            {
                "runTime": log.get("run_timestamp"),
                "status": (log.get("status") or "").title(),
                "guardrail": "Yes" if log.get("guardrail_triggered") else "No",
                "reason": log.get("guardrail_reason") or "",
                "initiatedBy": log.get("initiated_by") or "-",
            }
        )
    return rows


# FastAPI application ------------------------------------------------------------


def create_app(
    api_url: str,
    classification_model_path: Optional[str] = None,
    survival_model_path: Optional[str] = None,
    static_dir: Optional[Path] = None,
) -> FastAPI:
    """Instantiate the FastAPI app and wire up routes."""
    state = UIState(api_url, classification_model_path, survival_model_path)

    app = FastAPI(title="Flare+ UI Backend", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    assets_dir = None
    index_file: Optional[Path] = None

    if static_dir:
        static_dir = Path(static_dir)
        if static_dir.exists():
            assets_dir = static_dir / "assets"
            if assets_dir.exists():
                app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
            index_file = static_dir / "index.html"
        else:
            logger.warning("Svelte build directory %s not found. Run `npm run build` in ui-frontend.", static_dir)

    # --- Core status endpoints -------------------------------------------------

    @app.get("/ui/api/status")
    def get_status() -> Dict[str, Any]:
        # refresh connection state to detect if api became available
        state.refresh_connection()
        snapshot = asdict(state.snapshot())
        data_freshness = _serialize_data_freshness()
        return {
            "connection": snapshot,
            "dataFreshness": data_freshness,
            "limitation": (
                "This prototype has limited training data. Predictions may be unreliable until more historical "
                "events are ingested."
            ),
            "lastRefresh": state.last_refresh().isoformat() if state.last_refresh() else None,
        }

    # --- Ingestion -------------------------------------------------------------

    @app.post("/ui/api/ingest")
    def trigger_ingestion(request: IngestionRequest) -> Dict[str, Any]:
        can_trigger, rate_message = can_trigger_ingestion()
        if not can_trigger:
            return {
                "success": False,
                "message": rate_message or "Ingestion recently completed. Try again later.",
                "dataFreshness": _serialize_data_freshness(),
            }

        in_progress, _ = get_ingestion_state()
        if in_progress:
            return {
                "success": False,
                "message": "Ingestion already in progress. Please wait for completion.",
                "dataFreshness": _serialize_data_freshness(),
            }

        # re-check api connection before rejecting (api may have become available since startup)
        state.refresh_connection(force=True)
        if state.connection_mode != "api" or not state.api_url:
            return {
                "success": False,
                "message": "API connection required. Start the API server or provide model paths.",
                "dataFreshness": _serialize_data_freshness(),
            }

        set_ingestion_in_progress(True)
        try:
            success, results, error, error_type = run_ingestion_via_api(state.api_url, use_cache=request.use_cache)
            if success and results:
                summary = format_ingestion_summary(results)
                duration = results.get("duration", 0.0)
                status = results.get("overall_status", "success")
                set_last_ingestion_time(datetime.utcnow())
                return {
                    "success": True,
                    "message": f"Ingestion {status}. Duration: {duration:.1f}s",
                    "summary": summary,
                    "dataFreshness": _serialize_data_freshness(),
                }

            message = error or "Ingestion failed."
            if error_type == "transient":
                message = f"Ingestion failed (temporary): {message}"
            elif error_type == "permanent":
                message = f"Ingestion failed (permanent): {message}"

            return {
                "success": False,
                "message": message,
                "dataFreshness": _serialize_data_freshness(),
            }
        finally:
            set_ingestion_in_progress(False)

    # --- Predictions -----------------------------------------------------------

    def _ensure_prediction_window(force_refresh: bool, min_minutes: int) -> Tuple[bool, str]:
        last = state.last_refresh()
        if force_refresh or should_refresh(last, min_minutes):
            return True, ""
        return False, _throttle_message(last, min_minutes)

    @app.post("/ui/api/predict/classification")
    def predict_classification(request: ClassificationRequest) -> JSONResponse:
        allowed, message = _ensure_prediction_window(request.force_refresh, min_minutes=5)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": message,
                    "throttled": True,
                },
            )

        timestamp = request.timestamp or datetime.utcnow()
        region = request.region_number

        prediction: Optional[Dict[str, Any]] = None

        if state.connection_mode == "api" and state.api_url:
            payload = {
                "timestamp": timestamp.isoformat(),
                "window": int(request.window),  # ensure integer type
                "model_type": request.model_type,
            }
            if region is not None:
                payload["region_number"] = region

            success, data, error = make_api_request(
                state.api_url, "/predict/classification", method="POST", json_data=payload
            )
            if success and data:
                prediction = data
            else:
                return JSONResponse(
                    status_code=502,
                    content={
                        "success": False,
                        "message": f"API error: {error or 'unknown error'}",
                    },
                )
        elif state.connection_mode == "direct" and state.pipelines:
            classification_pipeline = state.pipelines.get("classification")
            if not classification_pipeline:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "Classification model not available (only survival model loaded).",
                    },
                )
            try:
                prediction = classification_pipeline.predict(
                    timestamp=timestamp,
                    window=request.window,
                    model_type=request.model_type,
                    region_number=region,
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.exception("Classification prediction failed")
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "message": f"Prediction error: {exc}",
                    },
                )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No connection available. Start the API server or provide model paths.",
                },
            )

        if not prediction:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Unable to generate prediction."},
            )

        serialized = _serialize_classification_result(prediction, request.window)
        if request.force_refresh:
            state.record_refresh()
        return JSONResponse(
            content={
                "success": True,
                "status": "Prediction successful",
                "result": serialized,
            }
        )

    @app.post("/ui/api/predict/survival")
    def predict_survival(request: SurvivalRequest) -> JSONResponse:
        allowed, message = _ensure_prediction_window(request.force_refresh, min_minutes=5)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": message,
                    "throttled": True,
                },
            )

        timestamp = request.timestamp or datetime.utcnow()
        region = request.region_number
        prediction: Optional[Dict[str, Any]] = None

        if state.connection_mode == "api" and state.api_url:
            payload = {
                "timestamp": timestamp.isoformat(),
                "model_type": request.model_type,
            }
            if region is not None:
                payload["region_number"] = region

            success, data, error = make_api_request(
                state.api_url, "/predict/survival", method="POST", json_data=payload
            )
            if success and data:
                prediction = data
            else:
                return JSONResponse(
                    status_code=502,
                    content={
                        "success": False,
                        "message": f"API error: {error or 'unknown error'}",
                    },
                )
        elif state.connection_mode == "direct" and state.pipelines:
            survival_pipeline = state.pipelines.get("survival")
            if not survival_pipeline:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "Survival model not available (only classification model loaded).",
                    },
                )
            try:
                prediction = survival_pipeline.predict_survival_probabilities(
                    timestamp=timestamp,
                    region_number=region,
                    model_type=request.model_type,
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.exception("Survival prediction failed")
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "message": f"Prediction error: {exc}",
                    },
                )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No connection available. Start the API server or provide model paths.",
                },
            )

        if not prediction:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Unable to generate prediction."},
            )

        serialized = _serialize_survival_result(prediction)
        if request.force_refresh:
            state.record_refresh()
        return JSONResponse(
            content={
                "success": True,
                "status": "Prediction successful",
                "result": serialized,
            }
        )

    # --- Timeline --------------------------------------------------------------

    @app.post("/ui/api/timeline")
    def timeline(request: TimelineRequest) -> JSONResponse:
        allowed, message = _ensure_prediction_window(request.force_refresh, min_minutes=10)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": message or "Refresh throttled. Use force refresh to bypass.",
                    "events": [],
                },
            )

        start = request.start or (datetime.utcnow() - timedelta(days=30))
        end = request.end or datetime.utcnow()

        if start >= end:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Start date must be before end date.",
                    "events": [],
                },
            )

        min_class = request.min_class or "B"
        region = request.region_number

        flares_df = query_historical_flares(
            start_date=start,
            end_date=end,
            min_class=min_class,
            region_number=region,
        )

        events: List[Dict[str, Any]] = []
        if not flares_df.empty:
            import pandas as pd

            for _, row in flares_df.iterrows():
                # Handle NaN values (convert to None for JSON serialization)
                class_magnitude = row.get("class_magnitude")
                if pd.isna(class_magnitude):
                    class_magnitude = None

                active_region = row.get("active_region")
                if pd.isna(active_region):
                    active_region = None

                events.append(
                    {
                        "startTime": row.get("start_time").isoformat() if row.get("start_time") is not None else None,
                        "peakTime": row.get("peak_time").isoformat() if row.get("peak_time") is not None else None,
                        "endTime": row.get("end_time").isoformat() if row.get("end_time") is not None else None,
                        "flareClass": row.get("flare_class"),
                        "classCategory": row.get("class_category"),
                        "classMagnitude": class_magnitude,
                        "region": int(active_region) if active_region is not None else None,
                        "location": row.get("location"),
                        "source": row.get("source"),
                    }
                )

        status_message = f"Found {len(events)} flare events" if events else "No flares found in the specified range."

        if request.force_refresh:
            state.record_refresh()

        return JSONResponse(
            content={
                "success": True,
                "message": status_message,
                "events": events,
            }
        )

    # --- About ----------------------------------------------------------------

    @app.get("/ui/api/about")
    def about() -> Dict[str, Any]:
        endpoints = CONFIG.get("data_ingestion", {}).get("endpoints", {})
        return {
            "endpoints": endpoints,
            "knownIssues": [
                "Limited training data may lead to inaccurate predictions.",
                "Some features may be missing for specific periods.",
                "Model performance is not yet benchmarked against official forecasts.",
                "Real-time data updates depend on NOAA API availability.",
            ],
        }

    # --- Admin ----------------------------------------------------------------

    def _gather_admin_status() -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
        if not AdminConfig.has_access():
            reason = AdminConfig.disabled_reason()
            return reason or "Admin access locked.", [], reason

        if state.connection_mode != "api" or not state.api_url:
            return (
                "API connection required to retrieve system health.",
                [],
                "API connection required.",
            )

        health = get_api_health_status(state.api_url)
        guardrail_text = _format_guardrail_status(health)

        success, logs, error = fetch_validation_logs(state.api_url, limit=10)
        if not success:
            logger.warning("Failed to fetch validation logs: %s", error)
            return guardrail_text + "\n\n⚠ " + error, [], error

        return guardrail_text, _serialize_validation_rows(logs), None

    @app.get("/ui/api/admin/session")
    def admin_session() -> Dict[str, Any]:
        return {
            "hasAccess": AdminConfig.has_access(),
            "indicator": AdminConfig.status_indicator(),
            "disabledReason": AdminConfig.disabled_reason(),
        }

    @app.post("/ui/api/admin/login")
    def admin_login(request: LoginRequest) -> Dict[str, Any]:
        success, message = AdminConfig.validate_credentials(request.username.strip(), request.password)
        guardrail_text, rows, error = _gather_admin_status()
        return {
            "success": success,
            "message": message,
            "session": {
                "hasAccess": AdminConfig.has_access(),
                "indicator": AdminConfig.status_indicator(),
                "disabledReason": AdminConfig.disabled_reason(),
            },
            "guardrailStatus": guardrail_text if success else "",
            "validationHistory": rows if success else [],
            "error": error if not success else None,
        }

    @app.post("/ui/api/admin/logout")
    def admin_logout() -> Dict[str, Any]:
        AdminConfig.revoke_session_access()
        return {
            "success": True,
            "message": "Session cleared. Admin features locked.",
            "session": {
                "hasAccess": AdminConfig.has_access(),
                "indicator": AdminConfig.status_indicator(),
                "disabledReason": AdminConfig.disabled_reason(),
            },
        }

    @app.get("/ui/api/admin/panel")
    def admin_panel() -> Dict[str, Any]:
        guardrail_text, rows, error = _gather_admin_status()
        return {
            "guardrailStatus": guardrail_text,
            "validationHistory": rows,
            "error": error,
        }

    @app.post("/ui/api/admin/validate")
    def admin_validate() -> Dict[str, Any]:
        if not AdminConfig.has_access():
            return {
                "success": False,
                "message": AdminConfig.disabled_reason() or "Admin access required.",
            }

        if state.connection_mode != "api" or not state.api_url:
            return {
                "success": False,
                "message": "API connection required to run validation.",
            }

        initiated_by = "ui-admin"
        success, data, error = trigger_validation_via_api(state.api_url, initiated_by)
        guardrail_text, rows, _ = _gather_admin_status()

        # include validation output in response even if it failed
        validation_output = ""
        if data:
            returncode = data.get("returncode", 0)
            stdout = data.get("stdout", "")
            stderr = data.get("stderr", "")
            # format output nicely - prioritize stdout, include stderr if different
            if stdout:
                validation_output = stdout
                if stderr and stderr.strip() and stderr != stdout:
                    validation_output += f"\n\nSTDERR:\n{stderr}"
            elif stderr:
                validation_output = stderr
            else:
                validation_output = f"Validation completed with return code: {returncode}"

        if not success:
            # create a user-friendly message
            if validation_output:
                user_message = "Validation completed with issues. See output below for details."
            else:
                user_message = error or "Validation failed."
            return {
                "success": False,
                "message": user_message,
                "guardrailStatus": guardrail_text,
                "validationHistory": rows,
                "validationOutput": validation_output,
            }

        output_lines = []
        stdout = data.get("stdout", "")
        stderr = data.get("stderr", "")
        return_code = data.get("returncode", 0)
        output_lines.append(f"Return code: {return_code}")
        if stdout:
            output_lines.append("STDOUT:\n" + stdout)
        if stderr:
            output_lines.append("STDERR:\n" + stderr)

        return {
            "success": True,
            "message": "Validation triggered.",
            "output": "\n\n".join(output_lines),
            "guardrailStatus": guardrail_text,
            "validationHistory": rows,
        }

    # --- SPA fallback ----------------------------------------------------------

    if index_file and index_file.exists():

        @app.get("/", include_in_schema=False)
        def serve_index() -> FileResponse:
            return FileResponse(index_file)

        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_spa(full_path: str) -> FileResponse:
            if full_path.startswith("ui/api"):
                raise HTTPException(status_code=404)
            return FileResponse(index_file)

    else:

        @app.get("/", include_in_schema=False)
        def fallback_status() -> Dict[str, Any]:
            return {
                "status": "ui-backend-ready",
                "message": "Svelte build not found. Run `npm install && npm run build` inside ui-frontend/.",
            }

    return app


__all__ = ["create_app"]
