#!/usr/bin/env python
"""staleness check for realtime NOAA vs flare+ monitoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def monitor_running_for_model_tag(model_tag: str) -> bool:
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        cmdline_path = f"/proc/{entry}/cmdline"
        try:
            raw = Path(cmdline_path).read_bytes()
        except Exception:
            continue
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        if MONITOR_CMD_FRAGMENT not in cmdline:
            continue
        if f"--model-tag {model_tag}" not in cmdline:
            continue
        return True
    return False


def latest_row_timestamps(args: argparse.Namespace) -> Optional[Tuple[datetime, datetime, datetime]]:
    db = get_database()
    with db.get_session() as session:
        row = (
            session.query(
                NoaaRealtimeLog.forecast_date,
                NoaaRealtimeLog.forecast_generated_at,
                NoaaRealtimeLog.last_sync_at,
            )
            .filter(
                NoaaRealtimeLog.model_tag == args.model_tag,
                NoaaRealtimeLog.target_class == args.target_class,
                NoaaRealtimeLog.horizon_days == args.horizon_days,
            )
            .order_by(NoaaRealtimeLog.forecast_generated_at.desc())
            .first()
        )
    if row is None:
        return None
    return row[0], row[1], row[2]


def evaluate_staleness(
    now_utc: datetime,
    latest_generated_at: Optional[datetime],
    stale_after_hours: float,
) -> Tuple[bool, Optional[float]]:
    if latest_generated_at is None:
        return True, None
    age_hours = (now_utc - latest_generated_at).total_seconds() / 3600.0
    return age_hours > stale_after_hours, age_hours


def write_json(path_str: Optional[str], payload: Dict[str, object]) -> Optional[Path]:
    if not path_str:
        return None
    output_path = resolve_path(path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="staleness check for realtime NOAA monitor")
    parser.add_argument("--model-tag", type=str, default="realtime-30d-live")
    parser.add_argument("--target-class", type=str, default="M", choices=["C", "M", "X"])
    parser.add_argument("--horizon-days", type=int, default=1, choices=[1, 2])
    parser.add_argument("--stale-after-hours", type=float, default=24.0)
    parser.add_argument("--require-monitor-process", action="store_true")
    parser.add_argument("--json-out", type=str, default="scripts/runtime/noaa_realtime_staleness.json")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.stale_after_hours <= 0:
        raise ValueError("--stale-after-hours must be > 0")


def main() -> None:
    args = parse_args()
    validate_args(args)

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    latest = latest_row_timestamps(args)
    latest_forecast_date = latest[0] if latest else None
    latest_generated_at = latest[1] if latest else None
    latest_sync_at = latest[2] if latest else None

    stale, age_hours = evaluate_staleness(
        now_utc=now_utc,
        latest_generated_at=latest_generated_at,
        stale_after_hours=args.stale_after_hours,
    )
    monitor_running = monitor_running_for_model_tag(args.model_tag)

    status = "ok"
    exit_code = 0
    reason = "fresh"

    if latest_generated_at is None:
        status = "stale"
        reason = "no_rows"
        exit_code = 2
    elif stale:
        status = "stale"
        reason = "age_exceeded"
        exit_code = 1
    elif args.require_monitor_process and not monitor_running:
        status = "stale"
        reason = "monitor_process_missing"
        exit_code = 3

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_tag": args.model_tag,
        "target_class": args.target_class,
        "horizon_days": args.horizon_days,
        "stale_after_hours": args.stale_after_hours,
        "status": status,
        "reason": reason,
        "monitor_running": monitor_running,
        "latest_forecast_date": latest_forecast_date.isoformat() if latest_forecast_date else None,
        "latest_forecast_generated_at": latest_generated_at.isoformat() if latest_generated_at else None,
        "latest_last_sync_at": latest_sync_at.isoformat() if latest_sync_at else None,
        "age_hours_since_latest_generated_at": age_hours,
        "exit_code": exit_code,
    }
    out_path = write_json(args.json_out, payload)

    print("=" * 88)
    print("REALTIME NOAA STALENESS CHECK")
    print("=" * 88)
    print(f"Model tag: {args.model_tag}")
    print(f"Status: {status} ({reason})")
    print(f"Stale threshold: {args.stale_after_hours}h")
    print(f"Age since latest generated row: {age_hours if age_hours is not None else 'n/a'}")
    print(f"Latest forecast date: {payload['latest_forecast_date']}")
    print(f"Latest generated at: {payload['latest_forecast_generated_at']}")
    print(f"Monitor running: {monitor_running}")
    if out_path is not None:
        print(f"JSON report: {out_path}")
    print("=" * 88)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
