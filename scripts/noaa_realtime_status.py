#!/usr/bin/env python
"""status view for realtime NOAA vs flare+ monitoring."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.database import get_database  # noqa: E402
from src.data.schema import NoaaRealtimeLog  # noqa: E402

MONITOR_CMD_FRAGMENT = "scripts/monitor_noaa_realtime.py --daemon"


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def monitor_processes(model_tag: str) -> List[Dict[str, object]]:
    processes: List[Dict[str, object]] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        cmdline_path = f"/proc/{entry}/cmdline"
        try:
            raw = Path(cmdline_path).read_bytes()
        except Exception:
            continue
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
        if MONITOR_CMD_FRAGMENT not in cmdline:
            continue
        if f"--model-tag {model_tag}" not in cmdline:
            continue
        processes.append(
            {
                "pid": int(entry),
                "cmdline": cmdline,
            }
        )
    processes.sort(key=lambda item: int(item["pid"]))
    return processes


def _accuracy(rows: Sequence[Tuple[bool, bool, bool]]) -> Dict[str, Optional[float]]:
    if not rows:
        return {
            "n": 0,
            "flare_accuracy": None,
            "noaa_accuracy": None,
            "accuracy_delta_points": None,
        }

    flare_hits = 0
    noaa_hits = 0
    for actual_event, flare_predicted_event, noaa_predicted_event in rows:
        actual = bool(actual_event)
        flare_pred = bool(flare_predicted_event)
        noaa_pred = bool(noaa_predicted_event)
        flare_hits += int(flare_pred == actual)
        noaa_hits += int(noaa_pred == actual)

    n = len(rows)
    flare_accuracy = flare_hits / n
    noaa_accuracy = noaa_hits / n
    return {
        "n": n,
        "flare_accuracy": flare_accuracy,
        "noaa_accuracy": noaa_accuracy,
        "accuracy_delta_points": flare_accuracy - noaa_accuracy,
    }


def _format_pct(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def latest_file_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"exists": False, "path": str(path)}

    line = ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                pass
    except Exception:
        line = ""

    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "last_line": line.strip(),
    }


def collect_db_status(args: argparse.Namespace) -> Dict[str, object]:
    def serialize_tuple(row: Optional[Tuple[object, ...]]) -> Optional[Dict[str, object]]:
        if row is None:
            return None
        (
            forecast_date,
            forecast_generated_at,
            noaa_probability,
            flare_probability,
            noaa_predicted_event,
            flare_predicted_event,
            actual_event,
            actual_event_count,
            actual_resolved_at,
            is_backfill,
            prediction_error,
        ) = row
        return {
            "forecast_date": forecast_date.isoformat() if forecast_date else None,
            "forecast_generated_at": forecast_generated_at.isoformat() if forecast_generated_at else None,
            "noaa_probability": noaa_probability,
            "flare_probability": flare_probability,
            "noaa_predicted_event": noaa_predicted_event,
            "flare_predicted_event": flare_predicted_event,
            "actual_event": actual_event,
            "actual_event_count": actual_event_count,
            "actual_resolved_at": actual_resolved_at.isoformat() if actual_resolved_at else None,
            "is_backfill": is_backfill,
            "prediction_error": prediction_error,
        }

    db = get_database()
    with db.get_session() as session:
        base_query = session.query(NoaaRealtimeLog).filter(
            NoaaRealtimeLog.model_tag == args.model_tag,
            NoaaRealtimeLog.target_class == args.target_class,
            NoaaRealtimeLog.horizon_days == args.horizon_days,
        )

        total_rows = base_query.count()
        resolved_rows = base_query.filter(NoaaRealtimeLog.actual_event.is_not(None)).count()
        unresolved_rows = base_query.filter(NoaaRealtimeLog.actual_event.is_(None)).count()
        realtime_rows = base_query.filter(NoaaRealtimeLog.is_backfill.is_(False)).count()
        realtime_resolved_rows = base_query.filter(
            NoaaRealtimeLog.is_backfill.is_(False), NoaaRealtimeLog.actual_event.is_not(None)
        ).count()

        latest_row = (
            session.query(
                NoaaRealtimeLog.forecast_date,
                NoaaRealtimeLog.forecast_generated_at,
                NoaaRealtimeLog.noaa_probability,
                NoaaRealtimeLog.flare_probability,
                NoaaRealtimeLog.noaa_predicted_event,
                NoaaRealtimeLog.flare_predicted_event,
                NoaaRealtimeLog.actual_event,
                NoaaRealtimeLog.actual_event_count,
                NoaaRealtimeLog.actual_resolved_at,
                NoaaRealtimeLog.is_backfill,
                NoaaRealtimeLog.prediction_error,
            )
            .filter(
                NoaaRealtimeLog.model_tag == args.model_tag,
                NoaaRealtimeLog.target_class == args.target_class,
                NoaaRealtimeLog.horizon_days == args.horizon_days,
            )
            .order_by(NoaaRealtimeLog.forecast_date.desc())
            .first()
        )
        latest_resolved_row = (
            session.query(
                NoaaRealtimeLog.forecast_date,
                NoaaRealtimeLog.forecast_generated_at,
                NoaaRealtimeLog.noaa_probability,
                NoaaRealtimeLog.flare_probability,
                NoaaRealtimeLog.noaa_predicted_event,
                NoaaRealtimeLog.flare_predicted_event,
                NoaaRealtimeLog.actual_event,
                NoaaRealtimeLog.actual_event_count,
                NoaaRealtimeLog.actual_resolved_at,
                NoaaRealtimeLog.is_backfill,
                NoaaRealtimeLog.prediction_error,
            )
            .filter(
                NoaaRealtimeLog.model_tag == args.model_tag,
                NoaaRealtimeLog.target_class == args.target_class,
                NoaaRealtimeLog.horizon_days == args.horizon_days,
                NoaaRealtimeLog.actual_event.is_not(None),
            )
            .order_by(NoaaRealtimeLog.forecast_date.desc())
            .first()
        )

        metrics_rows = (
            session.query(
                NoaaRealtimeLog.actual_event,
                NoaaRealtimeLog.flare_predicted_event,
                NoaaRealtimeLog.noaa_predicted_event,
            )
            .filter(
                NoaaRealtimeLog.model_tag == args.model_tag,
                NoaaRealtimeLog.target_class == args.target_class,
                NoaaRealtimeLog.horizon_days == args.horizon_days,
                NoaaRealtimeLog.actual_event.is_not(None),
                NoaaRealtimeLog.flare_predicted_event.is_not(None),
                NoaaRealtimeLog.noaa_predicted_event.is_not(None),
            )
            .order_by(NoaaRealtimeLog.forecast_date.asc())
            .all()
        )
        realtime_metrics_rows = (
            session.query(
                NoaaRealtimeLog.actual_event,
                NoaaRealtimeLog.flare_predicted_event,
                NoaaRealtimeLog.noaa_predicted_event,
            )
            .filter(
                NoaaRealtimeLog.model_tag == args.model_tag,
                NoaaRealtimeLog.target_class == args.target_class,
                NoaaRealtimeLog.horizon_days == args.horizon_days,
                NoaaRealtimeLog.is_backfill.is_(False),
                NoaaRealtimeLog.actual_event.is_not(None),
                NoaaRealtimeLog.flare_predicted_event.is_not(None),
                NoaaRealtimeLog.noaa_predicted_event.is_not(None),
            )
            .order_by(NoaaRealtimeLog.forecast_date.asc())
            .all()
        )

    return {
        "counts": {
            "total_rows": total_rows,
            "resolved_rows": resolved_rows,
            "unresolved_rows": unresolved_rows,
            "realtime_rows": realtime_rows,
            "realtime_resolved_rows": realtime_resolved_rows,
        },
        "latest_row": serialize_tuple(latest_row),
        "latest_resolved_row": serialize_tuple(latest_resolved_row),
        "metrics_all": _accuracy(metrics_rows),
        "metrics_realtime": _accuracy(realtime_metrics_rows),
    }


def write_json(path_str: Optional[str], payload: Dict[str, object]) -> Optional[Path]:
    if not path_str:
        return None
    output_path = resolve_path(path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="status view for realtime NOAA vs flare+ monitor")
    parser.add_argument("--model-tag", type=str, default="realtime-30d-live")
    parser.add_argument("--target-class", type=str, default="M", choices=["C", "M", "X"])
    parser.add_argument("--horizon-days", type=int, default=1, choices=[1, 2])
    parser.add_argument("--log-file", type=str, default="scripts/runtime/noaa_realtime_daemon.log")
    parser.add_argument("--csv-path", type=str, default="scripts/runtime/noaa_realtime_monitor.csv")
    parser.add_argument("--json-out", type=str, default="scripts/runtime/noaa_realtime_status.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processes = monitor_processes(args.model_tag)
    db_status = collect_db_status(args)
    log_state = latest_file_state(resolve_path(args.log_file))
    csv_state = latest_file_state(resolve_path(args.csv_path))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_tag": args.model_tag,
        "target_class": args.target_class,
        "horizon_days": args.horizon_days,
        "monitor_running": len(processes) > 0,
        "monitor_process_count": len(processes),
        "monitor_processes": processes,
        "database": db_status,
        "log_file": log_state,
        "csv_file": csv_state,
    }
    out_path = write_json(args.json_out, payload)

    print("=" * 88)
    print("REALTIME NOAA VS FLARE+ STATUS")
    print("=" * 88)
    print(f"Model tag: {args.model_tag}")
    print(f"Target: >= {args.target_class}-class in next {args.horizon_days} day(s)")
    print(f"Monitor running: {'yes' if payload['monitor_running'] else 'no'}")
    print(f"Monitor process count: {payload['monitor_process_count']}")
    if processes:
        print(f"Primary PID: {processes[0]['pid']}")
    print("-" * 88)
    counts = db_status["counts"]
    print(
        f"Rows total/resolved/unresolved: {counts['total_rows']}/{counts['resolved_rows']}/{counts['unresolved_rows']}"
    )
    print("Realtime rows/resolved: " f"{counts['realtime_rows']}/{counts['realtime_resolved_rows']}")
    metrics_all = db_status["metrics_all"]
    metrics_realtime = db_status["metrics_realtime"]
    print(
        "Accuracy (all rows): "
        f"flare={_format_pct(metrics_all['flare_accuracy'])} "
        f"noaa={_format_pct(metrics_all['noaa_accuracy'])} "
        f"delta={_format_pct(metrics_all['accuracy_delta_points'])}"
    )
    print(
        "Accuracy (realtime only): "
        f"flare={_format_pct(metrics_realtime['flare_accuracy'])} "
        f"noaa={_format_pct(metrics_realtime['noaa_accuracy'])} "
        f"delta={_format_pct(metrics_realtime['accuracy_delta_points'])}"
    )
    latest = db_status["latest_row"]
    latest_date = latest["forecast_date"] if latest else None
    latest_resolved = db_status["latest_resolved_row"]
    latest_resolved_date = latest_resolved["forecast_date"] if latest_resolved else None
    print(f"Latest forecast_date in DB: {latest_date}")
    print(f"Latest resolved forecast_date: {latest_resolved_date}")
    print("-" * 88)
    print(
        "Log file: "
        f"{log_state['path']} "
        f"(exists={log_state['exists']}, modified={log_state.get('modified_at_utc', 'n/a')})"
    )
    print(
        "CSV file: "
        f"{csv_state['path']} "
        f"(exists={csv_state['exists']}, modified={csv_state.get('modified_at_utc', 'n/a')})"
    )
    if out_path is not None:
        print(f"JSON status: {out_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
