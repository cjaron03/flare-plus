#!/usr/bin/env python
"""watchdog that keeps realtime NOAA vs flare+ monitor running."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def monitor_processes(model_tag: str) -> List[Dict[str, object]]:
    target = "scripts/monitor_noaa_realtime.py --daemon"
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
        if target not in cmdline:
            continue
        if f"--model-tag {model_tag}" not in cmdline:
            continue
        processes.append({"pid": int(entry), "cmdline": cmdline})

    processes.sort(key=lambda item: int(item["pid"]))
    return processes


def build_monitor_command(args: argparse.Namespace) -> List[str]:
    command: List[str] = [
        sys.executable,
        "scripts/monitor_noaa_realtime.py",
        "--daemon",
        "--model-tag",
        args.model_tag,
        "--model-path",
        args.model_path,
        "--model-type",
        args.model_type,
        "--target-class",
        args.target_class,
        "--horizon-days",
        str(args.horizon_days),
        "--flare-threshold",
        str(args.flare_threshold),
        "--noaa-threshold",
        str(args.noaa_threshold),
        "--run-time-utc",
        args.run_time_utc,
        "--poll-seconds",
        str(args.monitor_poll_seconds),
        "--log-file",
        args.log_file,
        "--csv-path",
        args.csv_path,
    ]

    if args.no_backfill:
        command.append("--no-backfill")
    if args.no_train_if_missing:
        command.append("--no-train-if-missing")
    if args.train_start_date:
        command.extend(["--train-start-date", args.train_start_date])
    if args.train_end_date:
        command.extend(["--train-end-date", args.train_end_date])
    if args.train_lookback_days is not None:
        command.extend(["--train-lookback-days", str(args.train_lookback_days)])
    if args.sample_interval_hours is not None:
        command.extend(["--sample-interval-hours", str(args.sample_interval_hours)])
    if args.test_size is not None:
        command.extend(["--test-size", str(args.test_size)])
    if args.train_models:
        command.extend(["--train-models", *args.train_models])
    return command


def start_monitor(args: argparse.Namespace) -> Optional[int]:
    command = build_monitor_command(args)
    log_path = resolve_path(args.watchdog_log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as handle:
        timestamp = datetime.now(timezone.utc).isoformat()
        handle.write(f"{timestamp} START monitor command: {' '.join(command)}\n")

    if args.dry_run:
        return None

    proc = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.pid


def log_watchdog(message: str, log_file: str) -> None:
    log_path = resolve_path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {message}\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def run_once(args: argparse.Namespace) -> bool:
    processes = monitor_processes(args.model_tag)
    if processes:
        log_watchdog(
            f"OK monitor alive model_tag={args.model_tag} pid={processes[0]['pid']} count={len(processes)}",
            args.watchdog_log_file,
        )
        print(
            f"watchdog: monitor already running for model_tag={args.model_tag} "
            f"(count={len(processes)}, pid={processes[0]['pid']})"
        )
        return False

    started_pid = start_monitor(args)
    if args.dry_run:
        log_watchdog(
            f"DRY-RUN monitor restart requested model_tag={args.model_tag}",
            args.watchdog_log_file,
        )
        print(f"watchdog: monitor missing for model_tag={args.model_tag}; dry-run restart only")
        return True

    # allow /proc to reflect new process
    time.sleep(max(1, args.start_check_delay_seconds))
    after = monitor_processes(args.model_tag)
    if after:
        log_watchdog(
            f"RESTARTED monitor model_tag={args.model_tag} pid={after[0]['pid']} " f"launcher_pid={started_pid}",
            args.watchdog_log_file,
        )
        print(
            f"watchdog: restarted monitor for model_tag={args.model_tag} "
            f"(pid={after[0]['pid']}, launcher_pid={started_pid})"
        )
        return True

    log_watchdog(
        f"ERROR restart attempted but monitor still missing model_tag={args.model_tag} launcher_pid={started_pid}",
        args.watchdog_log_file,
    )
    print(
        f"watchdog: restart attempted for model_tag={args.model_tag}, "
        "but monitor process was not detected after launch"
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="watchdog for realtime NOAA monitor")
    parser.add_argument("--model-tag", type=str, default="realtime-30d-live")
    parser.add_argument("--model-path", type=str, default="data/cache/noaa_realtime_model.joblib")
    parser.add_argument("--model-type", type=str, default="best")
    parser.add_argument("--target-class", type=str, default="M", choices=["C", "M", "X"])
    parser.add_argument("--horizon-days", type=int, default=1, choices=[1, 2])
    parser.add_argument("--flare-threshold", type=float, default=0.38)
    parser.add_argument("--noaa-threshold", type=float, default=0.56)
    parser.add_argument("--log-file", type=str, default="scripts/runtime/noaa_realtime_daemon.log")
    parser.add_argument("--csv-path", type=str, default="scripts/runtime/noaa_realtime_monitor.csv")
    parser.add_argument("--run-time-utc", type=str, default="00:10")
    parser.add_argument("--monitor-poll-seconds", type=int, default=30)
    parser.add_argument("--no-backfill", action="store_true")

    parser.add_argument("--no-train-if-missing", action="store_true")
    parser.add_argument("--train-start-date", type=str, default=None)
    parser.add_argument("--train-end-date", type=str, default=None)
    parser.add_argument("--train-lookback-days", type=int, default=180)
    parser.add_argument("--sample-interval-hours", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--train-models",
        type=str,
        nargs="+",
        default=["logistic"],
        choices=["logistic", "gradient_boosting", "lightgbm", "random_forest"],
    )

    parser.add_argument("--daemon", action="store_true", help="run watchdog loop")
    parser.add_argument("--watchdog-poll-seconds", type=int, default=300)
    parser.add_argument("--start-check-delay-seconds", type=int, default=2)
    parser.add_argument("--watchdog-log-file", type=str, default="scripts/runtime/noaa_realtime_watchdog.log")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.flare_threshold <= 1.0):
        raise ValueError("--flare-threshold must be between 0 and 1")
    if not (0.0 <= args.noaa_threshold <= 1.0):
        raise ValueError("--noaa-threshold must be between 0 and 1")
    if args.monitor_poll_seconds <= 0:
        raise ValueError("--monitor-poll-seconds must be > 0")
    if args.watchdog_poll_seconds <= 0:
        raise ValueError("--watchdog-poll-seconds must be > 0")
    if args.start_check_delay_seconds <= 0:
        raise ValueError("--start-check-delay-seconds must be > 0")
    if args.train_lookback_days <= 0:
        raise ValueError("--train-lookback-days must be > 0")
    if args.sample_interval_hours <= 0:
        raise ValueError("--sample-interval-hours must be > 0")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1")


def run_watchdog_daemon(args: argparse.Namespace) -> None:
    print(f"watchdog: daemon mode active for model_tag={args.model_tag}; " f"poll every {args.watchdog_poll_seconds}s")
    while True:
        run_once(args)
        time.sleep(args.watchdog_poll_seconds)


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.daemon:
        run_watchdog_daemon(args)
        return

    run_once(args)


if __name__ == "__main__":
    main()
