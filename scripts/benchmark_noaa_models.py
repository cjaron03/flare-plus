#!/usr/bin/env python
"""Benchmark flare+ classification predictions against NOAA/SWPC solar probabilities."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score

# add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.database import get_database  # noqa: E402
from src.data.schema import FlareEvent, GOESXRayFlux  # noqa: E402
from src.models.pipeline import ClassificationPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NOAA_SOLAR_PROBABILITIES_URL = "https://services.swpc.noaa.gov/json/solar_probabilities.json"
CLASS_ORDER = {"None": 0, "C": 1, "M": 2, "X": 3}
EVENT_CLASS_ORDER = {"C": 1, "M": 2, "X": 3}


@dataclass
class MetricSummary:
    source: str
    n_samples: int
    positives: int
    positive_rate: float
    brier_score: Optional[float]
    roc_auc: Optional[float]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]


@dataclass
class ThresholdSelection:
    source: str
    objective: str
    threshold: float
    objective_score: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def noaa_probability_column(target_class: str, horizon_days: int) -> str:
    return f"{target_class.lower()}_class_{horizon_days}_day"


def event_categories_for_target(target_class: str) -> List[str]:
    threshold = EVENT_CLASS_ORDER[target_class]
    return [name for name, rank in EVENT_CLASS_ORDER.items() if rank >= threshold]


def aggregate_model_probability(class_probabilities: Dict[str, float], target_class: str) -> float:
    threshold = CLASS_ORDER[target_class]
    return float(sum(prob for cls, prob in class_probabilities.items() if CLASS_ORDER.get(str(cls), -1) >= threshold))


def fetch_noaa_probabilities(
    target_class: str,
    horizon_days: int,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    endpoint: str = NOAA_SOLAR_PROBABILITIES_URL,
) -> pd.DataFrame:
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

    if start_date is not None:
        df = df[df["forecast_date"] >= start_date]
    if end_date is not None:
        df = df[df["forecast_date"] <= end_date]

    return df[["forecast_date", "noaa_probability"]].reset_index(drop=True)


def load_pipeline(model_path: Path) -> ClassificationPipeline:
    model_data = joblib.load(model_path)

    if isinstance(model_data, ClassificationPipeline):
        return model_data

    if isinstance(model_data, dict) and "models" in model_data:
        pipeline = ClassificationPipeline(use_mlflow=False)
        pipeline.models = model_data.get("models", {})
        pipeline.evaluation_results = model_data.get("evaluation_results", {})
        return pipeline

    raise ValueError(f"unsupported model payload format in {model_path}")


def train_pipeline(
    train_start: datetime,
    train_end: datetime,
    sample_interval_hours: int,
    test_size: float,
    models: List[str],
) -> ClassificationPipeline:
    logger.info("training classification pipeline from %s to %s", train_start.date(), train_end.date())
    # Disable SMOTE for benchmark runs; time-ordered folds can contain too-few
    # minority samples for synthetic oversampling and would fail training.
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
        run_name=f"noaa_benchmark_{train_start.strftime('%Y%m%d')}_{train_end.strftime('%Y%m%d')}",
    )
    if not pipeline.models:
        raise ValueError("training produced no usable models")
    return pipeline


def build_actual_outcomes(forecast_dates: Iterable[datetime], target_class: str, horizon_days: int) -> pd.DataFrame:
    categories = event_categories_for_target(target_class)
    records = []
    db = get_database()

    with db.get_session() as session:
        for forecast_date in forecast_dates:
            window_start = forecast_date
            window_end = forecast_date + timedelta(days=horizon_days)

            count = (
                session.query(FlareEvent)
                .filter(
                    FlareEvent.start_time >= window_start,
                    FlareEvent.start_time < window_end,
                    FlareEvent.class_category.in_(categories),
                )
                .count()
            )
            records.append(
                {
                    "forecast_date": forecast_date,
                    "actual_event": int(count > 0),
                    "actual_event_count": int(count),
                }
            )

    return pd.DataFrame(records)


def build_flux_coverage(
    forecast_dates: Iterable[datetime],
    horizon_days: int,
    min_flux_records: int,
) -> pd.DataFrame:
    records = []
    db = get_database()

    with db.get_session() as session:
        for forecast_date in forecast_dates:
            window_start = forecast_date
            window_end = forecast_date + timedelta(days=horizon_days)
            flux_count = (
                session.query(GOESXRayFlux)
                .filter(
                    GOESXRayFlux.timestamp >= window_start,
                    GOESXRayFlux.timestamp < window_end,
                )
                .count()
            )
            records.append(
                {
                    "forecast_date": forecast_date,
                    "flux_records_window": int(flux_count),
                    "coverage_ok": bool(flux_count >= min_flux_records),
                }
            )

    return pd.DataFrame(records)


def build_model_probabilities(
    pipeline: ClassificationPipeline,
    forecast_dates: Iterable[datetime],
    target_class: str,
    horizon_days: int,
    model_type: str,
) -> pd.DataFrame:
    window_hours = horizon_days * 24
    if window_hours not in (24, 48):
        raise ValueError("flare+ classification supports only 24h or 48h windows")

    rows = []
    for forecast_date in forecast_dates:
        try:
            prediction = pipeline.predict(
                timestamp=forecast_date,
                window=window_hours,
                model_type=model_type,
            )
            class_probs = prediction.get("class_probabilities", {})
            model_probability = aggregate_model_probability(class_probs, target_class=target_class)
            rows.append(
                {
                    "forecast_date": forecast_date,
                    "flare_probability": float(model_probability),
                    "flare_predicted_class": prediction.get("predicted_class"),
                }
            )
        except Exception as exc:
            logger.warning("prediction failed for %s: %s", forecast_date.isoformat(), exc)
            rows.append(
                {
                    "forecast_date": forecast_date,
                    "flare_probability": np.nan,
                    "flare_predicted_class": None,
                }
            )

    return pd.DataFrame(rows)


def compute_metrics(source: str, y_true: np.ndarray, y_prob: np.ndarray, decision_threshold: float) -> MetricSummary:
    mask = ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_true) == 0:
        return MetricSummary(
            source=source,
            n_samples=0,
            positives=0,
            positive_rate=0.0,
            brier_score=None,
            roc_auc=None,
            accuracy=None,
            precision=None,
            recall=None,
            f1_score=None,
        )

    y_pred = (y_prob >= decision_threshold).astype(int)
    positives = int(y_true.sum())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    roc_auc = None
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))

    return MetricSummary(
        source=source,
        n_samples=int(len(y_true)),
        positives=positives,
        positive_rate=float(positives / len(y_true)),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        roc_auc=roc_auc,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
    )


def optimize_threshold(
    source: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> Optional[ThresholdSelection]:
    """Find the best threshold for a source using a sweep over [threshold_min, threshold_max]."""
    mask = ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_true) == 0:
        return None

    thresholds = np.arange(threshold_min, threshold_max + (threshold_step / 2), threshold_step)
    if len(thresholds) == 0:
        return None

    best: Optional[ThresholdSelection] = None
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        accuracy = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        objective_score = float(f1 if objective == "f1" else accuracy)
        candidate = ThresholdSelection(
            source=source,
            objective=objective,
            threshold=float(threshold),
            objective_score=objective_score,
            accuracy=accuracy,
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
        )

        if best is None:
            best = candidate
            continue

        # deterministic tie-break: objective > f1 > recall > accuracy > precision
        candidate_key = (
            candidate.objective_score,
            candidate.f1_score,
            candidate.recall,
            candidate.accuracy,
            candidate.precision,
        )
        best_key = (
            best.objective_score,
            best.f1_score,
            best.recall,
            best.accuracy,
            best.precision,
        )
        if candidate_key > best_key:
            best = candidate

    return best


def apply_forecast_window(
    noaa_df: pd.DataFrame,
    explicit_start: Optional[datetime],
    explicit_end: Optional[datetime],
    default_lookback_days: int,
) -> pd.DataFrame:
    """Apply explicit or default date windowing to NOAA rows."""
    filtered = noaa_df.copy()
    if explicit_start is not None:
        filtered = filtered[filtered["forecast_date"] >= explicit_start]
    if explicit_end is not None:
        filtered = filtered[filtered["forecast_date"] <= explicit_end]

    if explicit_start is None and explicit_end is None and default_lookback_days > 0 and not filtered.empty:
        latest = filtered["forecast_date"].max()
        window_start = latest - timedelta(days=default_lookback_days - 1)
        filtered = filtered[filtered["forecast_date"] >= window_start]

    return filtered.sort_values("forecast_date").reset_index(drop=True)


def split_threshold_fit_and_holdout(
    eval_df: pd.DataFrame,
    holdout_days: int,
    validation_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Split covered rows into threshold-fit and holdout windows.

    Thresholds are optimized on the validation window immediately before holdout.
    Final metrics are always reported on holdout rows when split succeeds.
    """
    notes: List[str] = []
    if eval_df.empty:
        return eval_df, eval_df, notes

    if holdout_days <= 0:
        return eval_df, eval_df, notes

    max_date = eval_df["forecast_date"].max()
    holdout_start = max_date - timedelta(days=holdout_days - 1)
    holdout_df = eval_df[eval_df["forecast_date"] >= holdout_start].copy()
    if holdout_df.empty:
        notes.append("Holdout split skipped: no rows in requested holdout window.")
        return eval_df, eval_df, notes
    if len(holdout_df) == len(eval_df):
        notes.append("Holdout split skipped: holdout consumed all rows.")
        return eval_df, eval_df, notes

    prior_df = eval_df[eval_df["forecast_date"] < holdout_start].copy()
    if prior_df.empty:
        notes.append("Holdout split skipped: no pre-holdout rows for threshold fitting.")
        return eval_df, eval_df, notes

    if validation_days > 0:
        validation_start = holdout_start - timedelta(days=validation_days)
        fit_df = prior_df[prior_df["forecast_date"] >= validation_start].copy()
        if fit_df.empty:
            fit_df = prior_df
    else:
        fit_df = prior_df

    notes.append(
        "Threshold fit window: "
        f"{fit_df['forecast_date'].min().date()} to {fit_df['forecast_date'].max().date()} "
        f"({len(fit_df)} rows)"
    )
    notes.append(
        "Frozen-threshold holdout: "
        f"{holdout_df['forecast_date'].min().date()} to {holdout_df['forecast_date'].max().date()} "
        f"({len(holdout_df)} rows)"
    )
    return fit_df, holdout_df, notes


def format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def print_summary(
    summaries: List[MetricSummary],
    target_class: str,
    horizon_days: int,
    default_decision_threshold: float,
    min_date: datetime,
    max_date: datetime,
    threshold_by_source: Dict[str, float],
    covered_rows: int,
    total_rows: int,
    optimization_notes: Optional[List[str]] = None,
) -> None:
    print("\n" + "=" * 88)
    print("FLARE+ VS NOAA/SWPC FORECAST BENCHMARK")
    print("=" * 88)
    print(f"Target: >= {target_class}-class flare in next {horizon_days} day(s)")
    print(f"Default decision threshold: {default_decision_threshold:.2f}")
    print(f"Evaluation window: {min_date.date()} to {max_date.date()}")
    print(f"Coverage rows: {covered_rows}/{total_rows} ({(covered_rows / total_rows) * 100:.1f}%)")
    if optimization_notes:
        for note in optimization_notes:
            print(note)
    print("-" * 88)
    print(
        f"{'Source':<12} {'Thr':>6} {'N':>5} {'PosRate':>8} {'Brier':>10} {'ROC-AUC':>10} "
        f"{'Acc':>10} {'Prec':>10} {'Recall':>10} {'F1':>10}"
    )
    print("-" * 88)
    for item in summaries:
        print(
            f"{item.source:<12} {threshold_by_source.get(item.source, np.nan):>6.2f} "
            f"{item.n_samples:>5d} {item.positive_rate:>8.3f} "
            f"{format_metric(item.brier_score):>10} {format_metric(item.roc_auc):>10} "
            f"{format_metric(item.accuracy):>10} {format_metric(item.precision):>10} "
            f"{format_metric(item.recall):>10} {format_metric(item.f1_score):>10}"
        )
    print("=" * 88 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark flare+ classification predictions against NOAA/SWPC")
    parser.add_argument(
        "--model-path",
        type=str,
        help="existing classification model file (joblib dict with 'models', or ClassificationPipeline object)",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="train a classification model before benchmarking",
    )
    parser.add_argument(
        "--train-start-date",
        type=str,
        help="training start date (YYYY-MM-DD), required with --train-model",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        help="training end date (YYYY-MM-DD), required with --train-model",
    )
    parser.add_argument(
        "--train-models",
        type=str,
        nargs="+",
        default=["logistic", "gradient_boosting", "lightgbm", "random_forest"],
        choices=["logistic", "gradient_boosting", "lightgbm", "random_forest"],
        help="model types to train when --train-model is used",
    )
    parser.add_argument(
        "--sample-interval-hours",
        type=int,
        default=1,
        help="sample interval for dataset generation during training (default: 1h)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="test split fraction during training (default: 0.2)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="best",
        help="model type to use for flare+ predictions (default: best)",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="M",
        choices=["C", "M", "X"],
        help="forecast class threshold to evaluate (default: M)",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=1,
        choices=[1, 2],
        help="forecast horizon in days for both NOAA and flare+ (default: 1)",
    )
    parser.add_argument(
        "--forecast-start-date",
        type=str,
        help="benchmark start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--forecast-end-date",
        type=str,
        help="benchmark end date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--default-forecast-lookback-days",
        type=int,
        default=180,
        help=(
            "when no forecast start/end is provided, evaluate latest N days from NOAA feed "
            "(default: 180; capped by feed availability)"
        ),
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="binary decision threshold for precision/recall metrics (default: 0.5)",
    )
    parser.add_argument(
        "--optimize-threshold",
        type=str,
        default="none",
        choices=["none", "flare", "noaa", "both"],
        help=("optimize decision threshold for flare+, noaa, or both using eval rows " "(default: none)"),
    )
    parser.add_argument(
        "--threshold-objective",
        type=str,
        default="f1",
        choices=["f1", "accuracy"],
        help="objective used for threshold sweep (default: f1)",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.01,
        help="minimum threshold considered for optimization (default: 0.01)",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.99,
        help="maximum threshold considered for optimization (default: 0.99)",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="step size for threshold sweep (default: 0.01)",
    )
    parser.add_argument(
        "--threshold-holdout-days",
        type=int,
        default=14,
        help=(
            "freeze optimized thresholds and score only on the latest N days " "(default: 14, 0 disables holdout split)"
        ),
    )
    parser.add_argument(
        "--threshold-validation-days",
        type=int,
        default=60,
        help=(
            "days used to fit threshold immediately before holdout window " "(default: 60, 0 uses all pre-holdout rows)"
        ),
    )
    parser.add_argument(
        "--min-flux-records",
        type=int,
        default=0,
        help=(
            "minimum GOES flux records required in each forecast window to include that row "
            "(default: 0 disables strict flux coverage filter)"
        ),
    )
    parser.add_argument(
        "--min-covered-rows",
        type=int,
        default=0,
        help="fail benchmark when covered rows are below this count (default: 0 disabled)",
    )
    parser.add_argument(
        "--noaa-endpoint",
        type=str,
        default=NOAA_SOLAR_PROBABILITIES_URL,
        help=f"NOAA solar probabilities endpoint (default: {NOAA_SOLAR_PROBABILITIES_URL})",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/noaa_benchmark_details.csv",
        help="path to save row-level benchmark details",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.train_model and not args.model_path:
        raise ValueError("provide --model-path or use --train-model")

    if args.train_model and (not args.train_start_date or not args.train_end_date):
        raise ValueError("--train-model requires --train-start-date and --train-end-date")

    if not (0.0 < args.decision_threshold < 1.0):
        raise ValueError("--decision-threshold must be between 0 and 1")
    if not (0.0 < args.threshold_min < 1.0 and 0.0 < args.threshold_max < 1.0):
        raise ValueError("--threshold-min and --threshold-max must be between 0 and 1")
    if args.threshold_min >= args.threshold_max:
        raise ValueError("--threshold-min must be less than --threshold-max")
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be > 0")
    if args.default_forecast_lookback_days < 0:
        raise ValueError("--default-forecast-lookback-days must be >= 0")
    if args.threshold_holdout_days < 0:
        raise ValueError("--threshold-holdout-days must be >= 0")
    if args.threshold_validation_days < 0:
        raise ValueError("--threshold-validation-days must be >= 0")
    if args.min_flux_records < 0:
        raise ValueError("--min-flux-records must be >= 0")

    forecast_start = parse_date(args.forecast_start_date)
    forecast_end = parse_date(args.forecast_end_date)

    noaa_df = fetch_noaa_probabilities(
        target_class=args.target_class,
        horizon_days=args.horizon_days,
        start_date=None,
        end_date=None,
        endpoint=args.noaa_endpoint,
    )
    noaa_df = apply_forecast_window(
        noaa_df=noaa_df,
        explicit_start=forecast_start,
        explicit_end=forecast_end,
        default_lookback_days=args.default_forecast_lookback_days,
    )
    if noaa_df.empty:
        raise ValueError("no NOAA forecast rows available after filtering")
    if forecast_start is None and forecast_end is None and args.default_forecast_lookback_days > 0:
        if len(noaa_df) < args.default_forecast_lookback_days:
            logger.warning(
                "NOAA feed returned only %d rows (requested lookback %d days).",
                len(noaa_df),
                args.default_forecast_lookback_days,
            )

    if args.train_model:
        train_start = parse_date(args.train_start_date)
        train_end = parse_date(args.train_end_date)
        if train_start is None or train_end is None:
            raise ValueError("invalid training date format, expected YYYY-MM-DD")
        pipeline = train_pipeline(
            train_start=train_start,
            train_end=train_end,
            sample_interval_hours=args.sample_interval_hours,
            test_size=args.test_size,
            models=args.train_models,
        )
    else:
        pipeline = load_pipeline(Path(args.model_path))

    forecast_dates = list(noaa_df["forecast_date"])
    actuals_df = build_actual_outcomes(
        forecast_dates=forecast_dates,
        target_class=args.target_class,
        horizon_days=args.horizon_days,
    )
    coverage_df = build_flux_coverage(
        forecast_dates=forecast_dates,
        horizon_days=args.horizon_days,
        min_flux_records=args.min_flux_records,
    )
    model_df = build_model_probabilities(
        pipeline=pipeline,
        forecast_dates=forecast_dates,
        target_class=args.target_class,
        horizon_days=args.horizon_days,
        model_type=args.model_type,
    )

    merged = (
        noaa_df.merge(actuals_df, on="forecast_date", how="left")
        .merge(coverage_df, on="forecast_date", how="left")
        .merge(model_df, on="forecast_date", how="left")
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    eval_df = merged[merged["coverage_ok"].fillna(False)].copy()
    logger.info(
        "using %d/%d forecast rows after coverage filter (min_flux_records=%d)",
        len(eval_df),
        len(merged),
        args.min_flux_records,
    )
    if args.min_flux_records == 0:
        logger.warning("flux coverage filter disabled (--min-flux-records=0); all forecast rows are eligible")
    if len(eval_df) < 14:
        logger.warning(
            "only %d covered rows passed quality checks; benchmark may be unstable",
            len(eval_df),
        )
    if args.min_covered_rows > 0 and len(eval_df) < args.min_covered_rows:
        raise ValueError(f"covered rows {len(eval_df)} below required minimum {args.min_covered_rows}")
    if eval_df.empty:
        raise ValueError("no benchmark rows passed coverage checks")

    threshold_fit_df, score_df, split_notes = split_threshold_fit_and_holdout(
        eval_df=eval_df,
        holdout_days=args.threshold_holdout_days if args.optimize_threshold != "none" else 0,
        validation_days=args.threshold_validation_days,
    )
    if score_df.empty:
        raise ValueError("no rows available for scoring after holdout split")

    y_true_fit = threshold_fit_df["actual_event"].fillna(0).astype(int).to_numpy()
    noaa_prob_fit = threshold_fit_df["noaa_probability"].astype(float).to_numpy()
    flare_prob_fit = pd.to_numeric(threshold_fit_df["flare_probability"], errors="coerce").to_numpy(dtype=float)

    y_true_score = score_df["actual_event"].fillna(0).astype(int).to_numpy()
    noaa_prob_score = score_df["noaa_probability"].astype(float).to_numpy()
    flare_prob_score = pd.to_numeric(score_df["flare_probability"], errors="coerce").to_numpy(dtype=float)

    threshold_by_source: Dict[str, float] = {
        "NOAA/SWPC": args.decision_threshold,
        "flare+": args.decision_threshold,
    }
    optimization_notes: List[str] = list(split_notes)

    optimize_sources: List[str] = []
    if args.optimize_threshold in ("flare", "both"):
        optimize_sources.append("flare+")
    if args.optimize_threshold in ("noaa", "both"):
        optimize_sources.append("NOAA/SWPC")

    for source in optimize_sources:
        probs = flare_prob_fit if source == "flare+" else noaa_prob_fit
        optimized = optimize_threshold(
            source=source,
            y_true=y_true_fit,
            y_prob=probs,
            objective=args.threshold_objective,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        if optimized is None:
            logger.warning("threshold optimization skipped for %s (no valid rows)", source)
            continue

        threshold_by_source[source] = optimized.threshold
        optimization_notes.append(
            f"Optimized {source} threshold ({args.threshold_objective}): "
            f"{optimized.threshold:.2f} "
            f"(objective={optimized.objective_score:.4f}, f1={optimized.f1_score:.4f}, "
            f"recall={optimized.recall:.4f}, accuracy={optimized.accuracy:.4f})"
        )

    summaries = [
        compute_metrics("NOAA/SWPC", y_true_score, noaa_prob_score, threshold_by_source["NOAA/SWPC"]),
        compute_metrics("flare+", y_true_score, flare_prob_score, threshold_by_source["flare+"]),
    ]

    print_summary(
        summaries=summaries,
        target_class=args.target_class,
        horizon_days=args.horizon_days,
        default_decision_threshold=args.decision_threshold,
        min_date=score_df["forecast_date"].min(),
        max_date=score_df["forecast_date"].max(),
        threshold_by_source=threshold_by_source,
        covered_rows=len(score_df),
        total_rows=len(merged),
        optimization_notes=optimization_notes if optimization_notes else None,
    )

    output_path = PROJECT_ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info("saved row-level benchmark results to %s", output_path)


if __name__ == "__main__":
    main()
