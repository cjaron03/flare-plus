#!/usr/bin/env python
"""realtime monitor for flare+ vs noaa/swpc forecasts."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import pandas as pd
import requests
from sqlalchemy import and_

# add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.database import get_database  # noqa: E402
from src.data.schema import FlareEvent, NoaaRealtimeLog  # noqa: E402
from src.models.pipeline import ClassificationPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NOAA_SOLAR_PROBABILITIES_URL = "https://services.swpc.noaa.gov/json/solar_probabilities.json"
CLASS_ORDER = {"None": 0, "C": 1, "M": 2, "X": 3}
EVENT_CLASS_ORDER = {"C": 1, "M": 2, "X": 3}


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def parse_utc_time(value: str) -> Tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("--run-time-utc must use HH:MM format")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23):
        raise ValueError("run hour must be between 0 and 23")
    if not (0 <= minute <= 59):
        raise ValueError("run minute must be between 0 and 59")
    return hour, minute


def utc_day(value: datetime) -> datetime:
    return datetime(value.year, value.month, value.day)


def noaa_probability_column(target_class: str, horizon_days: int) -> str:
    return f"{target_class.lower()}_class_{horizon_days}_day"


def event_categories_for_target(target_class: str) -> List[str]:
    threshold = EVENT_CLASS_ORDER[target_class]
    return [name for name, rank in EVENT_CLASS_ORDER.items() if rank >= threshold]


def aggregate_model_probability(class_probabilities: Dict[str, float], target_class: str) -> float:
    threshold = CLASS_ORDER[target_class]
    return float(sum(prob for cls, prob in class_probabilities.items() if CLASS_ORDER.get(str(cls), -1) >= threshold))


def fetch_noaa_probabilities(target_class: str, horizon_days: int, endpoint: str) -> pd.DataFrame:
    col = noaa_probability_column(target_class, horizon_days)
    response = requests.get(endpoint, timeout=30)
    response.raise_for_status()

    payload = response.json()
    df = pd.DataFrame(payload)
    if "date" not in df.columns:
        raise ValueError("NOAA response missing 'date' column")
    if col not in df.columns:
        raise ValueError(f"NOAA response missing expected probability column: {col}")

    df["forecast_date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.floor("D")
    df["noaa_probability"] = pd.to_numeric(df[col], errors="coerce") / 100.0
    df["noaa_probability"] = df["noaa_probability"].clip(lower=0.0, upper=1.0)
    df = df.dropna(subset=["forecast_date", "noaa_probability"]).copy()
    df = df.sort_values("forecast_date").drop_duplicates(subset=["forecast_date"], keep="last")
    return df[["forecast_date", "noaa_probability"]].reset_index(drop=True)


def load_pipeline(model_path: Path) -> ClassificationPipeline:
    model_data = joblib.load(model_path)

    if isinstance(model_data, ClassificationPipeline):
        return model_data

    if isinstance(model_data, dict) and "models" in model_data:
        pipeline = ClassificationPipeline(use_mlflow=False, use_smote=False)
        pipeline.models = model_data.get("models", {})
        pipeline.evaluation_results = model_data.get("evaluation_results", {})
        return pipeline

    raise ValueError(f"unsupported model payload format in {model_path}")


def train_pipeline(
    model_path: Path,
    train_start: datetime,
    train_end: datetime,
    sample_interval_hours: int,
    test_size: float,
    models: List[str],
) -> ClassificationPipeline:
    logger.info("training fixed realtime model from %s to %s", train_start.date(), train_end.date())
    pipeline = ClassificationPipeline(use_mlflow=False, use_smote=False)
    dataset = pipeline.prepare_dataset(
        start_date=train_start,
        end_date=train_end,
        sample_interval_hours=sample_interval_hours,
    )
    if len(dataset) == 0:
        raise ValueError("training dataset is empty")

    logger.info("training dataset size: %d", len(dataset))
    pipeline.train_and_evaluate(
        dataset=dataset,
        test_size=test_size,
        models=models,
        run_name=f"realtime_noaa_monitor_{train_start.strftime('%Y%m%d')}_{train_end.strftime('%Y%m%d')}",
    )
    if not pipeline.models:
        raise ValueError("training produced no usable models")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "models": pipeline.models,
        "evaluation_results": pipeline.evaluation_results,
        "target_windows": [24, 48],
        "trained_at": datetime.utcnow().isoformat(),
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "use_smote": False,
    }
    joblib.dump(model_data, model_path)
    logger.info("saved fixed realtime model to %s", model_path)
    return pipeline


def resolve_training_window(
    now_utc: datetime,
    train_start_str: Optional[str],
    train_end_str: Optional[str],
    train_lookback_days: int,
) -> Tuple[datetime, datetime]:
    train_end = parse_date(train_end_str)
    if train_end is None:
        train_end = utc_day(now_utc) - timedelta(days=1)
    train_start = parse_date(train_start_str)
    if train_start is None:
        train_start = train_end - timedelta(days=train_lookback_days - 1)
    if train_start >= train_end:
        raise ValueError("train start date must be before train end date")
    return train_start, train_end


def ensure_pipeline(args: argparse.Namespace, now_utc: datetime) -> Tuple[ClassificationPipeline, Path]:
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if model_path.exists():
        logger.info("loading fixed realtime model from %s", model_path)
        pipeline = load_pipeline(model_path)
        return pipeline, model_path

    if not args.train_if_missing:
        raise ValueError(f"model file not found and training disabled: {model_path}")

    train_start, train_end = resolve_training_window(
        now_utc=now_utc,
        train_start_str=args.train_start_date,
        train_end_str=args.train_end_date,
        train_lookback_days=args.train_lookback_days,
    )
    pipeline = train_pipeline(
        model_path=model_path,
        train_start=train_start,
        train_end=train_end,
        sample_interval_hours=args.sample_interval_hours,
        test_size=args.test_size,
        models=args.train_models,
    )
    return pipeline, model_path


def select_forecast_dates(
    noaa_df: pd.DataFrame,
    today_utc: datetime,
    no_backfill: bool,
    max_forecast_age_days: int,
) -> List[datetime]:
    eligible = noaa_df[noaa_df["forecast_date"] <= today_utc].copy()
    if eligible.empty:
        return []

    if no_backfill:
        today_rows = eligible[eligible["forecast_date"] == today_utc]
        if not today_rows.empty:
            return [today_utc]
        return [eligible["forecast_date"].max()]

    oldest = today_utc - timedelta(days=max_forecast_age_days - 1)
    eligible = eligible[eligible["forecast_date"] >= oldest]
    return sorted(eligible["forecast_date"].tolist())


def upsert_forecasts(
    pipeline: ClassificationPipeline,
    model_path: Path,
    noaa_df: pd.DataFrame,
    forecast_dates: Iterable[datetime],
    run_ts: datetime,
    args: argparse.Namespace,
) -> Dict[str, int]:
    noaa_by_date = {row["forecast_date"]: float(row["noaa_probability"]) for _, row in noaa_df.iterrows()}
    today_utc = utc_day(run_ts)
    inserted = 0
    skipped_existing = 0
    errored = 0

    db = get_database()
    with db.get_session() as session:
        for forecast_date in forecast_dates:
            existing = (
                session.query(NoaaRealtimeLog)
                .filter(
                    NoaaRealtimeLog.model_tag == args.model_tag,
                    NoaaRealtimeLog.forecast_date == forecast_date,
                    NoaaRealtimeLog.target_class == args.target_class,
                    NoaaRealtimeLog.horizon_days == args.horizon_days,
                )
                .first()
            )
            if existing is not None:
                skipped_existing += 1
                continue

            prediction_error = None
            flare_probability = None
            flare_predicted_class = None
            flare_predicted_event = None
            try:
                prediction = pipeline.predict(
                    timestamp=forecast_date,
                    window=args.horizon_days * 24,
                    model_type=args.model_type,
                )
                class_probs = prediction.get("class_probabilities", {})
                flare_probability = aggregate_model_probability(class_probs, target_class=args.target_class)
                flare_predicted_class = prediction.get("predicted_class")
                flare_predicted_event = bool(flare_probability >= args.flare_threshold)
            except Exception as exc:
                prediction_error = str(exc)
                logger.warning("prediction failed for %s: %s", forecast_date.date(), prediction_error)
                errored += 1

            noaa_probability = noaa_by_date.get(forecast_date)
            if noaa_probability is None:
                logger.warning("missing NOAA probability for %s", forecast_date.date())
                continue

            row = NoaaRealtimeLog(
                model_tag=args.model_tag,
                model_type=args.model_type,
                model_path=str(model_path),
                target_class=args.target_class,
                horizon_days=args.horizon_days,
                forecast_date=forecast_date,
                forecast_generated_at=run_ts,
                last_sync_at=run_ts,
                is_backfill=bool((not args.no_backfill) and (forecast_date < today_utc)),
                noaa_endpoint=args.noaa_endpoint,
                noaa_probability=float(noaa_probability),
                flare_probability=None if flare_probability is None else float(flare_probability),
                noaa_threshold=float(args.noaa_threshold),
                flare_threshold=float(args.flare_threshold),
                noaa_predicted_event=bool(noaa_probability >= args.noaa_threshold),
                flare_predicted_event=flare_predicted_event,
                flare_predicted_class=flare_predicted_class,
                prediction_error=prediction_error,
                outcome_window_start=forecast_date,
                outcome_window_end=forecast_date + timedelta(days=args.horizon_days),
            )
            session.add(row)
            inserted += 1

    return {
        "inserted": inserted,
        "skipped_existing": skipped_existing,
        "prediction_errors": errored,
    }


def resolve_outcomes(run_ts: datetime, args: argparse.Namespace) -> int:
    categories = event_categories_for_target(args.target_class)
    resolved = 0

    db = get_database()
    with db.get_session() as session:
        pending = (
            session.query(NoaaRealtimeLog)
            .filter(
                NoaaRealtimeLog.model_tag == args.model_tag,
                NoaaRealtimeLog.target_class == args.target_class,
                NoaaRealtimeLog.horizon_days == args.horizon_days,
                NoaaRealtimeLog.actual_event.is_(None),
                NoaaRealtimeLog.outcome_window_end <= run_ts,
            )
            .all()
        )
        for row in pending:
            event_query = (
                session.query(FlareEvent)
                .filter(
                    FlareEvent.start_time >= row.outcome_window_start,
                    FlareEvent.start_time < row.outcome_window_end,
                    FlareEvent.class_category.in_(categories),
                )
                .order_by(FlareEvent.start_time.asc())
            )
            first_event = event_query.first()
            count = event_query.count()
            row.actual_event = bool(count > 0)
            row.actual_event_count = int(count)
            row.actual_first_event_time = None if first_event is None else first_event.start_time
            row.actual_resolved_at = run_ts
            row.last_sync_at = run_ts
            resolved += 1

    return resolved


def compute_accuracy_summary(args: argparse.Namespace, realtime_only: bool) -> Dict[str, Optional[float]]:
    db = get_database()
    with db.get_session() as session:
        filters = [
            NoaaRealtimeLog.model_tag == args.model_tag,
            NoaaRealtimeLog.target_class == args.target_class,
            NoaaRealtimeLog.horizon_days == args.horizon_days,
            NoaaRealtimeLog.actual_event.is_not(None),
            NoaaRealtimeLog.noaa_probability.is_not(None),
            NoaaRealtimeLog.flare_probability.is_not(None),
        ]
        if realtime_only:
            filters.append(NoaaRealtimeLog.is_backfill.is_(False))

        rows = (
            session.query(
                NoaaRealtimeLog.actual_event,
                NoaaRealtimeLog.flare_predicted_event,
                NoaaRealtimeLog.noaa_predicted_event,
            )
            .filter(and_(*filters))
            .order_by(NoaaRealtimeLog.forecast_date.asc())
            .all()
        )

    if not rows:
        return {
            "rows": 0,
            "flare_accuracy": None,
            "noaa_accuracy": None,
            "accuracy_delta_points": None,
            "flare_win_rate_vs_noaa": None,
        }

    flare_hits = 0
    noaa_hits = 0
    flare_wins = 0
    for actual_event, flare_predicted_event, noaa_predicted_event in rows:
        actual = bool(actual_event)
        flare_pred = bool(flare_predicted_event)
        noaa_pred = bool(noaa_predicted_event)
        flare_correct = flare_pred == actual
        noaa_correct = noaa_pred == actual
        flare_hits += int(flare_correct)
        noaa_hits += int(noaa_correct)
        flare_wins += int(flare_correct and not noaa_correct)

    n = len(rows)
    flare_accuracy = flare_hits / n
    noaa_accuracy = noaa_hits / n
    return {
        "rows": n,
        "flare_accuracy": flare_accuracy,
        "noaa_accuracy": noaa_accuracy,
        "accuracy_delta_points": flare_accuracy - noaa_accuracy,
        "flare_win_rate_vs_noaa": flare_wins / n,
    }


def export_csv(args: argparse.Namespace) -> Path:
    output_path = Path(args.csv_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db = get_database()
    with db.get_session() as session:
        rows = (
            session.query(NoaaRealtimeLog)
            .filter(
                NoaaRealtimeLog.model_tag == args.model_tag,
                NoaaRealtimeLog.target_class == args.target_class,
                NoaaRealtimeLog.horizon_days == args.horizon_days,
            )
            .order_by(NoaaRealtimeLog.forecast_date.asc())
            .all()
        )
        records: List[Dict[str, object]] = []
        for row in rows:
            records.append(
                {
                    "id": row.id,
                    "model_tag": row.model_tag,
                    "model_type": row.model_type,
                    "model_path": row.model_path,
                    "target_class": row.target_class,
                    "horizon_days": row.horizon_days,
                    "forecast_date": row.forecast_date.isoformat() if row.forecast_date else None,
                    "forecast_generated_at": (
                        row.forecast_generated_at.isoformat() if row.forecast_generated_at else None
                    ),
                    "last_sync_at": row.last_sync_at.isoformat() if row.last_sync_at else None,
                    "is_backfill": row.is_backfill,
                    "noaa_endpoint": row.noaa_endpoint,
                    "noaa_probability": row.noaa_probability,
                    "flare_probability": row.flare_probability,
                    "noaa_threshold": row.noaa_threshold,
                    "flare_threshold": row.flare_threshold,
                    "noaa_predicted_event": row.noaa_predicted_event,
                    "flare_predicted_event": row.flare_predicted_event,
                    "flare_predicted_class": row.flare_predicted_class,
                    "prediction_error": row.prediction_error,
                    "outcome_window_start": row.outcome_window_start.isoformat() if row.outcome_window_start else None,
                    "outcome_window_end": row.outcome_window_end.isoformat() if row.outcome_window_end else None,
                    "actual_event": row.actual_event,
                    "actual_event_count": row.actual_event_count,
                    "actual_first_event_time": (
                        row.actual_first_event_time.isoformat() if row.actual_first_event_time else None
                    ),
                    "actual_resolved_at": row.actual_resolved_at.isoformat() if row.actual_resolved_at else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
            )

    pd.DataFrame(records).to_csv(output_path, index=False)
    return output_path


def run_cycle(args: argparse.Namespace) -> None:
    run_ts = datetime.utcnow().replace(microsecond=0)
    today_utc = utc_day(run_ts)
    logger.info("starting realtime monitor cycle at %s UTC", run_ts.isoformat())

    db = get_database()
    db.connect()
    db.create_tables()

    pipeline, model_path = ensure_pipeline(args=args, now_utc=run_ts)
    noaa_df = fetch_noaa_probabilities(
        target_class=args.target_class,
        horizon_days=args.horizon_days,
        endpoint=args.noaa_endpoint,
    )
    if noaa_df.empty:
        raise ValueError("no NOAA forecast rows available")

    forecast_dates = select_forecast_dates(
        noaa_df=noaa_df,
        today_utc=today_utc,
        no_backfill=args.no_backfill,
        max_forecast_age_days=args.max_forecast_age_days,
    )
    upsert_stats = upsert_forecasts(
        pipeline=pipeline,
        model_path=model_path,
        noaa_df=noaa_df,
        forecast_dates=forecast_dates,
        run_ts=run_ts,
        args=args,
    )
    resolved_count = resolve_outcomes(run_ts=run_ts, args=args)
    csv_path = export_csv(args=args)

    overall = compute_accuracy_summary(args=args, realtime_only=False)
    realtime = compute_accuracy_summary(args=args, realtime_only=True)

    logger.info(
        "cycle complete: inserted=%d, existing=%d, prediction_errors=%d, outcomes_resolved=%d, csv=%s",
        upsert_stats["inserted"],
        upsert_stats["skipped_existing"],
        upsert_stats["prediction_errors"],
        resolved_count,
        csv_path,
    )
    logger.info(
        "all rows accuracy: n=%s flare=%.4f noaa=%.4f delta=%.4f",
        overall["rows"],
        overall["flare_accuracy"] if overall["flare_accuracy"] is not None else float("nan"),
        overall["noaa_accuracy"] if overall["noaa_accuracy"] is not None else float("nan"),
        overall["accuracy_delta_points"] if overall["accuracy_delta_points"] is not None else float("nan"),
    )
    logger.info(
        "realtime-only accuracy: n=%s flare=%.4f noaa=%.4f delta=%.4f",
        realtime["rows"],
        realtime["flare_accuracy"] if realtime["flare_accuracy"] is not None else float("nan"),
        realtime["noaa_accuracy"] if realtime["noaa_accuracy"] is not None else float("nan"),
        realtime["accuracy_delta_points"] if realtime["accuracy_delta_points"] is not None else float("nan"),
    )


def next_run_utc(now_utc: datetime, run_hour: int, run_minute: int) -> datetime:
    candidate = datetime(now_utc.year, now_utc.month, now_utc.day, run_hour, run_minute)
    if now_utc >= candidate:
        candidate += timedelta(days=1)
    return candidate


def run_daemon(args: argparse.Namespace) -> None:
    run_hour, run_minute = parse_utc_time(args.run_time_utc)
    logger.info("daemon mode enabled; target daily run time is %02d:%02d UTC", run_hour, run_minute)

    # run immediately once so today's forecast is captured.
    run_cycle(args=args)

    next_run = next_run_utc(datetime.utcnow(), run_hour, run_minute)
    logger.info("next scheduled run: %s UTC", next_run.isoformat())
    while True:
        now = datetime.utcnow()
        if now >= next_run:
            try:
                run_cycle(args=args)
            except Exception as exc:
                logger.exception("realtime monitor cycle failed: %s", exc)
            next_run += timedelta(days=1)
            logger.info("next scheduled run: %s UTC", next_run.isoformat())
        time.sleep(max(5, args.poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="realtime monitor for flare+ vs NOAA forecasts")
    parser.add_argument("--model-path", type=str, default="data/cache/noaa_realtime_model.joblib")
    parser.add_argument("--model-tag", type=str, default="realtime-30d-live")
    parser.add_argument("--model-type", type=str, default="best")
    parser.add_argument("--target-class", type=str, default="M", choices=["C", "M", "X"])
    parser.add_argument("--horizon-days", type=int, default=1, choices=[1, 2])
    parser.add_argument("--flare-threshold", type=float, default=0.38)
    parser.add_argument("--noaa-threshold", type=float, default=0.56)
    parser.add_argument("--noaa-endpoint", type=str, default=NOAA_SOLAR_PROBABILITIES_URL)
    parser.add_argument("--csv-path", type=str, default="data/cache/noaa_realtime_monitor.csv")
    parser.add_argument("--no-backfill", action="store_true")
    parser.add_argument("--max-forecast-age-days", type=int, default=30)

    parser.add_argument("--no-train-if-missing", action="store_true")
    parser.add_argument("--train-start-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--train-end-date", type=str, help="YYYY-MM-DD")
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

    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--run-time-utc", type=str, default="00:10", help="daily run time in UTC, HH:MM")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--log-file", type=str, default=None, help="optional log file path")

    args = parser.parse_args()
    args.train_if_missing = not args.no_train_if_missing

    if not (0.0 <= args.flare_threshold <= 1.0):
        raise ValueError("--flare-threshold must be between 0 and 1")
    if not (0.0 <= args.noaa_threshold <= 1.0):
        raise ValueError("--noaa-threshold must be between 0 and 1")
    if args.max_forecast_age_days <= 0:
        raise ValueError("--max-forecast-age-days must be > 0")
    if args.train_lookback_days <= 0:
        raise ValueError("--train-lookback-days must be > 0")
    if args.sample_interval_hours <= 0:
        raise ValueError("--sample-interval-hours must be > 0")
    if not (0 < args.test_size < 1):
        raise ValueError("--test-size must be between 0 and 1")
    parse_utc_time(args.run_time_utc)
    return args


def configure_file_logging(log_file: Optional[str]) -> None:
    if not log_file:
        return

    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    logger.info("file logging enabled at %s", log_path)


def main() -> None:
    args = parse_args()
    configure_file_logging(args.log_file)
    if args.daemon:
        run_daemon(args=args)
    else:
        run_cycle(args=args)


if __name__ == "__main__":
    main()
