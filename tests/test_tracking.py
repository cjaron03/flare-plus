"""tests for experiment tracking helper (no-op when disabled)."""

from src.utils import tracking


def test_tracking_disabled_behaviour():
    """ensure tracking helpers safely no-op when disabled."""
    assert tracking.is_enabled() is False

    # functions should not raise when mlflow disabled/not installed
    tracking.log_params({"alpha": 0.1})
    tracking.log_metrics({"accuracy": 0.95})
    tracking.log_dict({"foo": "bar"}, "artifacts/data.json")
    tracking.log_artifact("README.md", artifact_path="docs")  # file may not exist but function should handle disable
    tracking.log_joblib_artifact({"dummy": True}, artifact_name="dummy.joblib")

    # context manager should yield None and not raise
    with tracking.start_run("test-run"):
        tracking.log_params({"inside": True})
