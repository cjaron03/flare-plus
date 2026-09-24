"""unit tests for realtime NOAA monitor helper scripts."""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, script_filename: str):
    script_path = PROJECT_ROOT / "scripts" / script_filename
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_compute_metrics_known_values():
    report = _load_script_module("noaa_realtime_report_test", "noaa_realtime_report.py")
    rows = [
        {
            "actual_event": True,
            "flare_predicted_event": True,
            "noaa_predicted_event": True,
            "flare_probability": 0.9,
            "noaa_probability": 0.8,
        },
        {
            "actual_event": True,
            "flare_predicted_event": True,
            "noaa_predicted_event": False,
            "flare_probability": 0.7,
            "noaa_probability": 0.3,
        },
        {
            "actual_event": False,
            "flare_predicted_event": True,
            "noaa_predicted_event": False,
            "flare_probability": 0.8,
            "noaa_probability": 0.2,
        },
        {
            "actual_event": False,
            "flare_predicted_event": False,
            "noaa_predicted_event": True,
            "flare_probability": 0.1,
            "noaa_probability": 0.6,
        },
    ]

    metrics = report.compute_metrics(rows)

    assert metrics["n"] == 4
    assert metrics["flare_accuracy"] == pytest.approx(0.75)
    assert metrics["noaa_accuracy"] == pytest.approx(0.50)
    assert metrics["accuracy_delta_points"] == pytest.approx(0.25)
    assert metrics["flare_precision"] == pytest.approx(2 / 3)
    assert metrics["flare_recall"] == pytest.approx(1.0)
    assert metrics["flare_f1"] == pytest.approx(0.8)
    assert metrics["noaa_precision"] == pytest.approx(0.5)
    assert metrics["noaa_recall"] == pytest.approx(0.5)
    assert metrics["noaa_f1"] == pytest.approx(0.5)
    assert metrics["flare_brier"] == pytest.approx(0.1875)
    assert metrics["noaa_brier"] == pytest.approx(0.2325)
    assert metrics["mcnemar_n10_flare_only_correct"] == 2
    assert metrics["mcnemar_n01_noaa_only_correct"] == 1
    assert metrics["mcnemar_exact_p_value"] == pytest.approx(1.0)


def test_report_compute_metrics_empty_rows():
    report = _load_script_module("noaa_realtime_report_test_empty", "noaa_realtime_report.py")
    metrics = report.compute_metrics([])

    assert metrics["n"] == 0
    assert metrics["flare_accuracy"] is None
    assert metrics["noaa_accuracy"] is None
    assert metrics["mcnemar_exact_p_value"] is None


def test_status_accuracy_summary():
    status = _load_script_module("noaa_realtime_status_test", "noaa_realtime_status.py")
    rows = [
        (True, True, True),
        (True, True, False),
        (False, True, False),
        (False, False, True),
    ]

    metrics = status._accuracy(rows)
    assert metrics["n"] == 4
    assert metrics["flare_accuracy"] == pytest.approx(0.75)
    assert metrics["noaa_accuracy"] == pytest.approx(0.50)
    assert metrics["accuracy_delta_points"] == pytest.approx(0.25)


def test_status_latest_file_state_reads_last_line(tmp_path):
    status = _load_script_module("noaa_realtime_status_file_test", "noaa_realtime_status.py")
    file_path = tmp_path / "sample.log"
    file_path.write_text("line-1\nline-2\n", encoding="utf-8")

    state = status.latest_file_state(file_path)
    assert state["exists"] is True
    assert state["last_line"] == "line-2"
    assert state["size_bytes"] > 0

    missing = status.latest_file_state(tmp_path / "missing.log")
    assert missing["exists"] is False


def test_watchdog_monitor_processes_filters_by_model_tag(monkeypatch):
    watchdog = _load_script_module("noaa_realtime_watchdog_test", "noaa_realtime_watchdog.py")

    cmdline_map = {
        "/proc/100/cmdline": (
            b"python\x00scripts/monitor_noaa_realtime.py\x00--daemon\x00--model-tag\x00realtime-30d-live\x00"
        ),
        "/proc/200/cmdline": (
            b"python\x00scripts/monitor_noaa_realtime.py\x00--daemon\x00--model-tag\x00other-tag\x00"
        ),
        "/proc/300/cmdline": b"python\x00some_other_script.py\x00",
    }

    monkeypatch.setattr(watchdog.os, "listdir", lambda _: ["100", "200", "300", "not-a-pid"])

    def fake_read_bytes(path_obj):
        path_str = str(path_obj)
        if path_str not in cmdline_map:
            raise FileNotFoundError(path_str)
        return cmdline_map[path_str]

    monkeypatch.setattr(watchdog.Path, "read_bytes", fake_read_bytes, raising=False)

    processes = watchdog.monitor_processes("realtime-30d-live")
    assert len(processes) == 1
    assert processes[0]["pid"] == 100


def test_watchdog_build_command_includes_expected_flags():
    watchdog = _load_script_module("noaa_realtime_watchdog_cmd_test", "noaa_realtime_watchdog.py")
    args = argparse.Namespace(
        model_tag="realtime-30d-live",
        model_path="data/cache/noaa_realtime_model.joblib",
        model_type="best",
        target_class="M",
        horizon_days=1,
        flare_threshold=0.38,
        noaa_threshold=0.56,
        run_time_utc="00:10",
        monitor_poll_seconds=30,
        log_file="scripts/runtime/noaa_realtime_daemon.log",
        csv_path="scripts/runtime/noaa_realtime_monitor.csv",
        no_backfill=True,
        no_train_if_missing=True,
        train_start_date="2025-01-01",
        train_end_date="2025-12-31",
        train_lookback_days=180,
        sample_interval_hours=12,
        test_size=0.2,
        train_models=["logistic", "gradient_boosting"],
    )

    command = watchdog.build_monitor_command(args)

    assert command[0]
    assert command[1:3] == ["scripts/monitor_noaa_realtime.py", "--daemon"]
    assert "--model-tag" in command
    assert "--no-backfill" in command
    assert "--no-train-if-missing" in command
    assert "--train-models" in command
    models_idx = command.index("--train-models")
    assert command[models_idx + 1 : models_idx + 3] == ["logistic", "gradient_boosting"]


def test_watchdog_validate_args_rejects_invalid_threshold():
    watchdog = _load_script_module("noaa_realtime_watchdog_validate_test", "noaa_realtime_watchdog.py")
    args = argparse.Namespace(
        flare_threshold=0.38,
        noaa_threshold=1.5,
        monitor_poll_seconds=30,
        watchdog_poll_seconds=300,
        start_check_delay_seconds=2,
        train_lookback_days=180,
        sample_interval_hours=12,
        test_size=0.2,
    )

    with pytest.raises(ValueError, match="--noaa-threshold"):
        watchdog.validate_args(args)


def test_watchdog_run_once_skips_restart_when_alive(monkeypatch, tmp_path):
    watchdog = _load_script_module("noaa_realtime_watchdog_run_once_test", "noaa_realtime_watchdog.py")

    args = argparse.Namespace(
        model_tag="realtime-30d-live",
        watchdog_log_file=str(tmp_path / "watchdog.log"),
        dry_run=False,
        start_check_delay_seconds=1,
    )

    monkeypatch.setattr(
        watchdog,
        "monitor_processes",
        lambda model_tag: [{"pid": 42, "cmdline": "python scripts/monitor_noaa_realtime.py --daemon"}],
    )

    start_calls = []
    log_calls = []
    monkeypatch.setattr(watchdog, "start_monitor", lambda *_: start_calls.append(True) or 999)
    monkeypatch.setattr(watchdog, "log_watchdog", lambda message, log_file: log_calls.append((message, log_file)))

    changed = watchdog.run_once(args)
    assert changed is False
    assert start_calls == []
    assert len(log_calls) == 1
    assert "OK monitor alive" in log_calls[0][0]


def test_runtime_maintenance_truncate_file_tail_in_place(tmp_path):
    maintenance = _load_script_module(
        "noaa_realtime_runtime_maintenance_test",
        "noaa_realtime_runtime_maintenance.py",
    )
    target = tmp_path / "sample.log"
    target.write_bytes(b"0123456789")

    before, after = maintenance.truncate_file_tail_in_place(path=target, keep_bytes=4)
    assert before == 10
    assert after == 4
    assert target.read_bytes() == b"6789"


def test_runtime_maintenance_validate_args_rejects_invalid_retention():
    maintenance = _load_script_module(
        "noaa_realtime_runtime_maintenance_validate_test",
        "noaa_realtime_runtime_maintenance.py",
    )
    args = argparse.Namespace(
        retention_days=0,
        max_file_mb=20.0,
        keep_tail_mb=5.0,
        patterns=["*.log"],
    )

    with pytest.raises(ValueError, match="--retention-days"):
        maintenance.validate_args(args)


def test_staleness_evaluate_staleness_threshold():
    staleness = _load_script_module(
        "noaa_realtime_staleness_eval_test",
        "noaa_realtime_staleness_check.py",
    )
    now_utc = datetime(2026, 2, 12, 12, 0, 0)
    latest = datetime(2026, 2, 12, 2, 0, 0)

    is_stale, age_hours = staleness.evaluate_staleness(
        now_utc=now_utc,
        latest_generated_at=latest,
        stale_after_hours=8.0,
    )
    assert is_stale is True
    assert age_hours == pytest.approx(10.0)


def test_staleness_validate_args_rejects_invalid_threshold():
    staleness = _load_script_module(
        "noaa_realtime_staleness_validate_test",
        "noaa_realtime_staleness_check.py",
    )
    args = argparse.Namespace(stale_after_hours=0.0)
    with pytest.raises(ValueError, match="--stale-after-hours"):
        staleness.validate_args(args)
