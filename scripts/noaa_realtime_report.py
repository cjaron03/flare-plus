#!/usr/bin/env python
"""metrics report for realtime NOAA vs flare+ tracking."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support

# add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.database import get_database  # noqa: E402
from src.data.schema import NoaaRealtimeLog  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def load_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    db = get_database()
    with db.get_session() as session:
        query = (
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
            )
            .filter(
                NoaaRealtimeLog.model_tag == args.model_tag,
                NoaaRealtimeLog.target_class == args.target_class,
                NoaaRealtimeLog.horizon_days == args.horizon_days,
                NoaaRealtimeLog.actual_event.is_not(None),
                NoaaRealtimeLog.noaa_probability.is_not(None),
                NoaaRealtimeLog.flare_probability.is_not(None),
                NoaaRealtimeLog.noaa_predicted_event.is_not(None),
                NoaaRealtimeLog.flare_predicted_event.is_not(None),
            )
            .order_by(NoaaRealtimeLog.forecast_date.asc())
        )
        if args.realtime_only:
            query = query.filter(NoaaRealtimeLog.is_backfill.is_(False))
        rows = query.all()

    records: List[Dict[str, object]] = []
    for row in rows:
        records.append(
            {
                "forecast_date": row[0],
                "forecast_generated_at": row[1],
                "noaa_probability": float(row[2]),
                "flare_probability": float(row[3]),
                "noaa_predicted_event": bool(row[4]),
                "flare_predicted_event": bool(row[5]),
                "actual_event": bool(row[6]),
                "actual_event_count": int(row[7]) if row[7] is not None else None,
                "actual_resolved_at": row[8],
                "is_backfill": bool(row[9]),
            }
        )
    return records


def compute_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "n": 0,
            "flare_accuracy": None,
            "noaa_accuracy": None,
            "accuracy_delta_points": None,
            "flare_precision": None,
            "flare_recall": None,
            "flare_f1": None,
            "noaa_precision": None,
            "noaa_recall": None,
            "noaa_f1": None,
            "flare_brier": None,
            "noaa_brier": None,
            "mcnemar_n10_flare_only_correct": 0,
            "mcnemar_n01_noaa_only_correct": 0,
            "mcnemar_exact_p_value": None,
            "flare_wins_rate": None,
        }

    y_true = [1 if bool(r["actual_event"]) else 0 for r in rows]
    flare_pred = [1 if bool(r["flare_predicted_event"]) else 0 for r in rows]
    noaa_pred = [1 if bool(r["noaa_predicted_event"]) else 0 for r in rows]
    flare_prob = [float(r["flare_probability"]) for r in rows]
    noaa_prob = [float(r["noaa_probability"]) for r in rows]

    flare_hits = sum(int(p == y) for p, y in zip(flare_pred, y_true))
    noaa_hits = sum(int(p == y) for p, y in zip(noaa_pred, y_true))
    n = len(rows)

    flare_precision, flare_recall, flare_f1, _ = precision_recall_fscore_support(
        y_true, flare_pred, average="binary", zero_division=0
    )
    noaa_precision, noaa_recall, noaa_f1, _ = precision_recall_fscore_support(
        y_true, noaa_pred, average="binary", zero_division=0
    )

    # McNemar exact test via binomial test on discordant pairs.
    # n10: flare correct and NOAA wrong
    # n01: NOAA correct and flare wrong
    n10 = 0
    n01 = 0
    for fp, npred, y in zip(flare_pred, noaa_pred, y_true):
        flare_correct = fp == y
        noaa_correct = npred == y
        if flare_correct and not noaa_correct:
            n10 += 1
        elif noaa_correct and not flare_correct:
            n01 += 1
    discordant = n10 + n01
    mcnemar_p = None
    if discordant > 0:
        mcnemar_p = float(binomtest(min(n10, n01), n=discordant, p=0.5, alternative="two-sided").pvalue)

    return {
        "n": n,
        "flare_accuracy": _safe_div(flare_hits, n),
        "noaa_accuracy": _safe_div(noaa_hits, n),
        "accuracy_delta_points": _safe_div(flare_hits, n) - _safe_div(noaa_hits, n),
        "flare_precision": float(flare_precision),
        "flare_recall": float(flare_recall),
        "flare_f1": float(flare_f1),
        "noaa_precision": float(noaa_precision),
        "noaa_recall": float(noaa_recall),
        "noaa_f1": float(noaa_f1),
        "flare_brier": float(brier_score_loss(y_true, flare_prob)),
        "noaa_brier": float(brier_score_loss(y_true, noaa_prob)),
        "mcnemar_n10_flare_only_correct": int(n10),
        "mcnemar_n01_noaa_only_correct": int(n01),
        "mcnemar_exact_p_value": mcnemar_p,
        "flare_wins_rate": _safe_div(n10, n),
    }


def format_pct(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def write_json(path_str: Optional[str], payload: Dict[str, object]) -> Optional[Path]:
    if not path_str:
        return None
    output_path = Path(path_str)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="realtime NOAA vs flare+ metrics report")
    parser.add_argument("--model-tag", type=str, default="realtime-30d-live")
    parser.add_argument("--target-class", type=str, default="M", choices=["C", "M", "X"])
    parser.add_argument("--horizon-days", type=int, default=1, choices=[1, 2])
    parser.add_argument("--realtime-only", action="store_true")
    parser.add_argument("--json-out", type=str, default="scripts/runtime/noaa_realtime_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    metrics = compute_metrics(rows)

    latest_forecast_date = rows[-1]["forecast_date"] if rows else None
    latest_generated_at = rows[-1]["forecast_generated_at"] if rows else None
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_tag": args.model_tag,
        "target_class": args.target_class,
        "horizon_days": args.horizon_days,
        "realtime_only": bool(args.realtime_only),
        "latest_forecast_date": latest_forecast_date,
        "latest_generated_at": latest_generated_at,
        "metrics": metrics,
    }
    out_path = write_json(args.json_out, report)

    print("=" * 88)
    print("REALTIME NOAA VS FLARE+ REPORT")
    print("=" * 88)
    print(f"Model tag: {args.model_tag}")
    print(f"Target: >= {args.target_class}-class in next {args.horizon_days} day(s)")
    print(f"Realtime-only mode: {args.realtime_only}")
    print(f"Rows scored: {metrics['n']}")
    print(f"Latest forecast date: {latest_forecast_date}")
    print(f"Latest generated at: {latest_generated_at}")
    print("-" * 88)
    print(f"flare+ accuracy: {format_pct(metrics['flare_accuracy'])}")
    print(f"NOAA accuracy:  {format_pct(metrics['noaa_accuracy'])}")
    print(f"Delta:          {format_pct(metrics['accuracy_delta_points'])}")
    print("-" * 88)
    print(
        "flare+ metrics: "
        f"precision={format_pct(metrics['flare_precision'])} "
        f"recall={format_pct(metrics['flare_recall'])} "
        f"f1={format_pct(metrics['flare_f1'])} "
        f"brier={metrics['flare_brier'] if metrics['flare_brier'] is not None else 'n/a'}"
    )
    print(
        "NOAA metrics:   "
        f"precision={format_pct(metrics['noaa_precision'])} "
        f"recall={format_pct(metrics['noaa_recall'])} "
        f"f1={format_pct(metrics['noaa_f1'])} "
        f"brier={metrics['noaa_brier'] if metrics['noaa_brier'] is not None else 'n/a'}"
    )
    print("-" * 88)
    print(
        "McNemar (exact): "
        f"n10(flare only correct)={metrics['mcnemar_n10_flare_only_correct']}, "
        f"n01(noaa only correct)={metrics['mcnemar_n01_noaa_only_correct']}, "
        f"p={metrics['mcnemar_exact_p_value'] if metrics['mcnemar_exact_p_value'] is not None else 'n/a'}"
    )
    if out_path is not None:
        print(f"JSON report: {out_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
