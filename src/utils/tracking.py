"""experiment tracking helpers (mlflow integration)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional
import tempfile

import joblib

from src.config import CONFIG

try:
    import mlflow
except ImportError:  # pragma: no cover - mlflow optional
    mlflow = None  # type: ignore[assignment]


def _mlflow_config() -> Dict[str, Any]:
    """return mlflow configuration block."""
    tracking_cfg = CONFIG.get("tracking", {})
    return tracking_cfg.get("mlflow", {}) or {}


def is_enabled() -> bool:
    """return true when mlflow tracking is enabled and available."""
    cfg = _mlflow_config()
    return bool(cfg.get("enabled")) and mlflow is not None


def _sanitize_param_value(value: Any) -> Optional[Any]:
    """coerce parameter values to mlflow-friendly representations."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _to_float(value: Any) -> Optional[float]:
    """convert value to float for metric logging."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_serializable(obj: Any) -> Any:
    """convert object to json-serializable structure."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    return str(obj)


@contextmanager
def start_run(run_name: str, tags: Optional[Dict[str, str]] = None):
    """context manager to start an mlflow run when enabled."""
    if not is_enabled():  # pragma: no cover - no-op when disabled
        yield None
        return

    cfg = _mlflow_config()
    tracking_uri = cfg.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = cfg.get("experiment_name", "flare-plus")
    artifact_location = cfg.get("artifact_location")
    if artifact_location:
        mlflow.set_experiment(
            experiment_name=experiment_name,
            artifact_location=artifact_location,
        )
    else:
        mlflow.set_experiment(experiment_name=experiment_name)

    base_tags = cfg.get("run_tags", {}) or {}
    combined_tags = {**base_tags, **(tags or {})}

    with mlflow.start_run(run_name=run_name, tags=combined_tags):
        yield mlflow


def log_params(params: Dict[str, Any]) -> None:
    """log parameter dictionary to mlflow."""
    if not is_enabled() or not params:
        return

    sanitized = {}
    for key, value in params.items():
        clean_value = _sanitize_param_value(value)
        if clean_value is not None:
            sanitized[str(key)] = clean_value

    if sanitized:
        mlflow.log_params(sanitized)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """log numeric metrics to mlflow."""
    if not is_enabled() or not metrics:
        return

    numeric_metrics = {}
    for key, value in metrics.items():
        numeric_value = _to_float(value)
        if numeric_value is not None:
            numeric_metrics[str(key)] = numeric_value

    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics, step=step)


def log_dict(data: Dict[str, Any], artifact_file: str) -> None:
    """log dictionary as json artifact."""
    if not is_enabled() or not data:
        return

    serializable = _to_serializable(data)
    mlflow.log_dict(serializable, artifact_file)


def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
    """log local file artifact."""
    if not is_enabled():
        return
    mlflow.log_artifact(path, artifact_path=artifact_path)


def log_joblib_artifact(
    obj: Any,
    artifact_name: str = "model.joblib",
    artifact_path: Optional[str] = None,
) -> None:
    """serialize object with joblib and log as artifact."""
    if not is_enabled():
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / artifact_name
        joblib.dump(obj, target)
        mlflow.log_artifact(str(target), artifact_path=artifact_path)
