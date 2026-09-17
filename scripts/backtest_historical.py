#!/usr/bin/env python
"""backtest trained models against historical flare events."""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402
import joblib  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.data.database import get_database  # noqa: E402
from src.data.schema import FlareEvent  # noqa: E402
from src.features.pipeline import FeatureEngineer  # noqa: E402
from src.models.survival_pipeline import SurvivalAnalysisPipeline  # noqa: E402
from src.models.backtest_scoring import compute_backtest_metrics  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def run_backtest(
    model_path: Path,
    start_date: datetime,
    end_date: datetime,
    target_class: str = "C",
    sample_interval_hours: int = 24,
    horizon_hours: int = 24,
    threshold: float = 0.5,
) -> Tuple[List[Dict], Dict[str, float]]:
    """run backtest on historical data."""
    logger.info(f"backtesting from {start_date} to {end_date}")

    # load model
    model = load_survival_model(model_path)

    # the model has no opinion past the last bucket it was fitted over, so a
    # horizon beyond that would be scored against extrapolation.
    if horizon_hours > model.max_time_hours:
        logger.warning(
            f"horizon {horizon_hours}h exceeds the model's {model.max_time_hours}h "
            f"range, clamping to {model.max_time_hours}h"
        )
        horizon_hours = model.max_time_hours
    logger.info(f"horizon {horizon_hours}h, decision threshold {threshold}")
    feature_engineer = FeatureEngineer()
    db = get_database()

    # get historical flares
    historical_flares = get_historical_flares(db, start_date, end_date, target_class)
    logger.info(f"found {len(historical_flares)} historical {target_class}+ flares")

    # historical_flares already returns dicts in the correct format
    actuals = historical_flares

    # generate predictions at regular intervals
    predictions = []
    current_time = start_date

    while current_time <= end_date:
        try:
            # compute features
            features_df = feature_engineer.compute_features(current_time)

            if features_df is None or len(features_df) == 0:
                logger.warning(f"no features at {current_time}, skipping")
                current_time += timedelta(hours=sample_interval_hours)
                continue

            # make prediction
            result = model.predict_survival_probabilities(timestamp=current_time)

            predictions.append(
                {
                    "timestamp": current_time,
                    "window_hours": model.max_time_hours,
                    "probability_distribution": result.get("probability_distribution", {}),
                    "hazard_score": result.get("hazard_score"),
                }
            )

        except Exception as e:
            logger.error(f"prediction failed at {current_time}: {e}")

        current_time += timedelta(hours=sample_interval_hours)

    logger.info(f"generated {len(predictions)} predictions")

    # compute metrics
    metrics = compute_backtest_metrics(
        predictions, actuals, horizon_hours=horizon_hours, threshold=threshold
    )

    return predictions, metrics


def generate_backtest_report(
    metrics: Dict[str, float],
    model_path: Path,
    start_date: datetime,
    end_date: datetime,
    output_path: Path = None,
    train_window: Tuple[datetime, datetime] = None,
) -> str:
    """generate human readable backtest report."""
    if "error" in metrics:
        return f"backtest produced no usable result: {metrics['error']}"

    base = metrics["base_rate"]
    bss = metrics["brier_skill_score"]

    report_lines = [
        "=" * 70,
        "FLARE+ BACKTEST REPORT",
        "=" * 70,
        "",
        f"Model: {model_path.name}",
        f"Period: {start_date.date()} to {end_date.date()}",
        f"Duration: {(end_date - start_date).days} days",
        f"Forecast horizon: {metrics['horizon_hours']}h",
        f"Decision threshold: {metrics['threshold']:.2f}",
        "",
        "=" * 70,
        "WHAT HAPPENED",
        "=" * 70,
        "",
        f"Forecasts issued:         {metrics['n_predictions']}",
        f"Flares in period:         {metrics['n_actuals']}",
        f"Windows containing one:   {metrics['n_windows_with_event']} "
        f"({base:.1%} of windows)",
        "",
        f"True positives:           {metrics['true_positives']}",
        f"False positives:          {metrics['false_positives']}",
        f"False negatives:          {metrics['false_negatives']}",
        f"True negatives:           {metrics['true_negatives']}",
        "",
        "=" * 70,
        "THE MODEL AGAINST THE FREE ANSWERS",
        "=" * 70,
        "",
        "                          model      always yes    always no",
        f"Precision                 {metrics['precision']:.3f}      "
        f"{metrics['always_yes_precision']:.3f}         n/a",
        f"Recall                    {metrics['recall']:.3f}      "
        f"{metrics['always_yes_recall']:.3f}         0.000",
        f"F1                        {metrics['f1_score']:.3f}      "
        f"{metrics['always_yes_f1']:.3f}         0.000",
        "",
        f"Brier score               {metrics['brier_score']:.4f}",
        f"Brier, climatology        {metrics['brier_climatology']:.4f}",
        f"Brier skill score         {bss:+.4f}",
        "",
        f"Accuracy of always no     {metrics['always_no_accuracy']:.1%}",
        "",
        "=" * 70,
        "INTERPRETATION",
        "=" * 70,
        "",
    ]

    # the only interpretation that means anything is the one relative to
    # climatology. an absolute cutoff on precision would call a model good
    # for having noticed that the sun was active.
    if bss > 0.15:
        report_lines.append(
            f"MODEL BEATS CLIMATOLOGY: it removed {bss:.1%} of the error that "
            "knowing the base rate alone would have left."
        )
    elif bss > 0.0:
        report_lines.append(
            f"MARGINAL: the model is {bss:.1%} better than repeating the base "
            "rate. That is not yet a useful forecast."
        )
    else:
        report_lines.append(
            "NO SKILL: the model scores no better than a forecaster who "
            "ignores every input and repeats the base rate. Whatever the "
            "precision figure says, nothing has been demonstrated here."
        )

    if metrics["f1_score"] <= metrics["always_yes_f1"]:
        report_lines.append(
            f"F1 WARNING: saying yes every single time scores "
            f"{metrics['always_yes_f1']:.3f}, which is at least as good as the "
            f"model's {metrics['f1_score']:.3f}. F1 is flattering the model "
            "because events are common in this window."
        )

    if metrics["true_positives"] + metrics["false_positives"] == 0:
        report_lines.append(
            "NOTE: the model never crossed the threshold, so precision is "
            "undefined rather than zero."
        )

    # a threshold sweep, because one threshold is one arbitrary choice and a
    # model that separates the classes does it at more than one of them.
    report_lines.extend(
        [
            "",
            "=" * 70,
            "THRESHOLD SWEEP",
            "=" * 70,
            "",
            "  thresh   issued   precision   recall      f1",
        ]
    )
    for row in metrics["threshold_sweep"]:
        if row["n_positive"] == 0 and row["threshold"] > 0:
            continue
        report_lines.append(
            f"   {row['threshold']:.2f}     {row['n_positive']:4d}      "
            f"{row['precision']:.3f}     {row['recall']:.3f}   {row['f1']:.3f}"
        )

    report_lines.extend(
        [
            "",
            "=" * 70,
            "NOTES",
            "=" * 70,
            "",
            "- Precision: of the forecasts that said yes, the fraction that",
            "  were followed by a flare inside the horizon.",
            "- Recall: of the windows that did contain a flare, the fraction",
            "  the model called.",
            "- Brier score: mean squared error of the probability itself.",
            "  Lower is better, but it is only meaningful next to the",
            "  climatology row below it.",
            "- Brier skill score: the fraction of climatology's error the",
            "  model removed. Zero means the model added nothing over knowing",
            "  how often flares happen. This is the number that matters.",
            "",
            "Rare events make every absolute score look good. During a quiet",
            "period a forecaster who always says no is right almost always and",
            "is worth nothing; during an active one a forecaster who always",
            "says yes looks precise for the same reason. Only the comparison",
            "against climatology survives either case.",
            "",
        ]
    )

    if train_window:
        t0, t1 = train_window
        if t0 <= end_date and start_date <= t1:
            report_lines.extend(
                [
                    "WARNING: this backtest window overlaps the model's training",
                    f"window ({t0.date()} to {t1.date()}). The model has already",
                    "seen some of this data, so these scores are optimistic and",
                    "are not an estimate of performance on new observations.",
                    "",
                ]
            )

    report_lines.extend(
        [
            "Historical backtesting validates against past solar activity and",
            "does not guarantee future performance. Solar cycles vary.",
            "",
            f"Report generated: {datetime.utcnow().isoformat()}",
            "=" * 70,
        ]
    )

    report_text = "\n".join(report_lines)

    # save to file if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report_text)
        logger.info(f"report saved to {output_path}")

    return report_text


def main():
    """main entry point for backtesting."""
    import argparse

    parser = argparse.ArgumentParser(description="backtest flare+ models")
    parser.add_argument(
        "--model",
        type=str,
        default="models/survival_model_c_class.joblib",
        help="path to model file (default: survival_model_c_class.joblib)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="start date for backtesting (YYYY-MM-DD), default: 6 months ago",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="end date for backtesting (YYYY-MM-DD), default: today",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="C",
        choices=["X", "M", "C", "B"],
        help="minimum flare class to consider (default: C)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=24,
        help="prediction interval in hours (default: 24)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="forecast horizon in hours, the 'by when' of the forecast "
             "(default: 24)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="probability at or above which the forecast counts as a yes "
             "(default: 0.5)",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        help="start of the model's training window (YYYY-MM-DD). given, the "
             "report warns when the backtest overlaps it",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        help="end of the model's training window (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/backtest_report.txt",
        help="output path for report (default: data/backtest_report.txt)",
    )

    args = parser.parse_args()

    # parse dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.utcnow()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=180)  # 6 months

    # get model path
    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        logger.error(f"model not found: {model_path}")
        sys.exit(1)

    # run backtest
    logger.info("starting backtest...")
    predictions, metrics = run_backtest(
        model_path=model_path,
        start_date=start_date,
        end_date=end_date,
        target_class=args.target_class,
        sample_interval_hours=args.interval,
        horizon_hours=args.horizon,
        threshold=args.threshold,
    )

    # generate report
    train_window = None
    if args.train_start and args.train_end:
        train_window = (
            datetime.strptime(args.train_start, "%Y-%m-%d"),
            datetime.strptime(args.train_end, "%Y-%m-%d"),
        )

    report = generate_backtest_report(
        metrics=metrics,
        model_path=model_path,
        start_date=start_date,
        end_date=end_date,
        output_path=PROJECT_ROOT / args.output,
        train_window=train_window,
    )

    print("\n" + report)

    # the gate reads skill, not f1. f1 rises on its own when events are
    # common, so gating on it would pass a model for having been pointed at
    # an active fortnight.
    bss = metrics.get("brier_skill_score", 0)
    if bss > 0.15:
        logger.info(f"BACKTEST PASSED: brier skill score {bss:+.3f} over climatology")
        sys.exit(0)
    elif bss > 0:
        logger.warning(f"BACKTEST MARGINAL: brier skill score {bss:+.3f}, barely over climatology")
        sys.exit(0)
    else:
        logger.error(f"BACKTEST FAILED: brier skill score {bss:+.3f}, no better than climatology")
        sys.exit(1)


if __name__ == "__main__":
    main()
