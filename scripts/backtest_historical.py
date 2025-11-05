#!/usr/bin/env python
"""backtest trained models against historical flare events."""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402
import numpy as np  # noqa: E402
import joblib  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.data.database import get_database  # noqa: E402
from src.data.schema import FlareEvent  # noqa: E402
from src.features.pipeline import FeatureEngineer  # noqa: E402
from src.models.survival_pipeline import SurvivalAnalysisPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def reconstruct_survival_pipeline(pipeline_data: Dict) -> SurvivalAnalysisPipeline:
    """reconstruct survival pipeline from saved dict."""
    pipeline = SurvivalAnalysisPipeline(
        target_flare_class=pipeline_data.get("target_flare_class", "X"),
        max_time_hours=pipeline_data.get("max_time_hours", 168),
    )
    pipeline.is_fitted = pipeline_data.get("is_fitted", False)

    # restore model state
    if pipeline_data.get("cox_model"):
        pipeline.cox_model = pipeline_data["cox_model"]
    if pipeline_data.get("gb_model"):
        pipeline.gb_model = pipeline_data["gb_model"]

    return pipeline


def load_survival_model(model_path: Path) -> SurvivalAnalysisPipeline:
    """load survival model from joblib file."""
    logger.info(f"loading model from {model_path}")

    model_data = joblib.load(model_path)

    # reconstruct pipeline if saved as dict
    if isinstance(model_data, dict):
        return reconstruct_survival_pipeline(model_data)

    return model_data


def get_historical_flares(
    db,
    start_date: datetime,
    end_date: datetime,
    min_class: str = "C",
) -> List[Dict]:
    """retrieve historical flare events from database."""
    with db.get_session() as session:
        flares = (
            session.query(FlareEvent)
            .filter(
                FlareEvent.peak_time >= start_date,
                FlareEvent.peak_time <= end_date,
                FlareEvent.flare_class.isnot(None),
            )
            .order_by(FlareEvent.peak_time)
            .all()
        )

        # filter by minimum class and convert to dicts while in session
        class_order = {"B": 0, "C": 1, "M": 2, "X": 3}
        min_class_value = class_order.get(min_class, 0)

        filtered = []
        for flare in flares:
            flare_class = flare.flare_class[0] if flare.flare_class else "B"
            if class_order.get(flare_class, 0) >= min_class_value:
                filtered.append(
                    {
                        "peak_time": flare.peak_time,
                        "flare_class": flare.flare_class,
                        "class_magnitude": flare.class_magnitude,
                    }
                )

    return filtered


def compute_backtest_metrics(predictions: List[Dict], actuals: List[Dict]) -> Dict[str, float]:
    """compute performance metrics for backtesting."""
    if not predictions or not actuals:
        return {
            "error": "insufficient data",
            "n_predictions": len(predictions),
            "n_actuals": len(actuals),
        }

    # match predictions to actuals
    matches = []
    for pred in predictions:
        pred_time = pred["timestamp"]
        pred_window_end = pred_time + timedelta(hours=pred.get("window_hours", 48))

        # find actuals in prediction window
        window_actuals = [a for a in actuals if pred_time <= a["peak_time"] <= pred_window_end]

        matches.append(
            {
                "prediction": pred,
                "actuals": window_actuals,
                "hit": len(window_actuals) > 0,
            }
        )

    # calculate metrics
    n_predictions = len(matches)
    n_hits = sum(1 for m in matches if m["hit"])
    n_misses = n_predictions - n_hits

    # false negatives: actuals not predicted
    predicted_times = {m["actuals"][0]["peak_time"] for m in matches if m["actuals"]}
    false_negatives = len([a for a in actuals if a["peak_time"] not in predicted_times])

    # precision and recall
    precision = n_hits / n_predictions if n_predictions > 0 else 0
    recall = n_hits / (n_hits + false_negatives) if (n_hits + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # calculate brier score if probabilities available
    brier_scores = []
    for match in matches:
        pred = match["prediction"]
        if "probability_distribution" in pred:
            probs = pred["probability_distribution"]
            # true outcome: 1 if flare occurred, 0 otherwise
            true_outcome = 1 if match["hit"] else 0
            # predicted probability (sum of probabilities in window)
            predicted_prob = sum(probs.values())
            brier = (predicted_prob - true_outcome) ** 2
            brier_scores.append(brier)

    avg_brier = np.mean(brier_scores) if brier_scores else None

    return {
        "n_predictions": n_predictions,
        "n_actuals": len(actuals),
        "n_hits": n_hits,
        "n_misses": n_misses,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "brier_score": avg_brier,
    }


def run_backtest(
    model_path: Path,
    start_date: datetime,
    end_date: datetime,
    target_class: str = "C",
    sample_interval_hours: int = 24,
) -> Tuple[List[Dict], Dict[str, float]]:
    """run backtest on historical data."""
    logger.info(f"backtesting from {start_date} to {end_date}")

    # load model
    model = load_survival_model(model_path)
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
    metrics = compute_backtest_metrics(predictions, actuals)

    return predictions, metrics


def generate_backtest_report(
    metrics: Dict[str, float],
    model_path: Path,
    start_date: datetime,
    end_date: datetime,
    output_path: Path = None,
) -> str:
    """generate human-readable backtest report."""
    report_lines = [
        "=" * 70,
        "FLARE+ BACKTEST REPORT",
        "=" * 70,
        "",
        f"Model: {model_path.name}",
        f"Period: {start_date.date()} to {end_date.date()}",
        f"Duration: {(end_date - start_date).days} days",
        "",
        "=" * 70,
        "PERFORMANCE METRICS",
        "=" * 70,
        "",
        f"Total predictions:        {metrics.get('n_predictions', 0)}",
        f"Actual flares:            {metrics.get('n_actuals', 0)}",
        f"Correct predictions:      {metrics.get('n_hits', 0)}",
        f"Missed predictions:       {metrics.get('n_misses', 0)}",
        f"False negatives:          {metrics.get('false_negatives', 0)}",
        "",
        f"Precision:                {metrics.get('precision', 0):.3f}",
        f"Recall:                   {metrics.get('recall', 0):.3f}",
        f"F1 Score:                 {metrics.get('f1_score', 0):.3f}",
    ]

    if metrics.get("brier_score") is not None:
        report_lines.append(f"Brier Score:              {metrics['brier_score']:.3f}")

    report_lines.extend(
        [
            "",
            "=" * 70,
            "INTERPRETATION",
            "=" * 70,
            "",
        ]
    )

    # add interpretation
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)

    if precision > 0.7:
        report_lines.append("✓ HIGH PRECISION: Model rarely issues false alarms")
    elif precision > 0.5:
        report_lines.append("~ MODERATE PRECISION: Some false alarms expected")
    else:
        report_lines.append("✗ LOW PRECISION: High false alarm rate")

    if recall > 0.7:
        report_lines.append("✓ HIGH RECALL: Model catches most flares")
    elif recall > 0.5:
        report_lines.append("~ MODERATE RECALL: Model misses some flares")
    else:
        report_lines.append("✗ LOW RECALL: Model misses many flares")

    if metrics.get("brier_score") and metrics["brier_score"] < 0.15:
        report_lines.append("✓ GOOD CALIBRATION: Probabilities are well-calibrated")
    elif metrics.get("brier_score") and metrics["brier_score"] < 0.25:
        report_lines.append("~ FAIR CALIBRATION: Probabilities somewhat calibrated")
    elif metrics.get("brier_score"):
        report_lines.append("✗ POOR CALIBRATION: Probabilities need improvement")

    report_lines.extend(
        [
            "",
            "=" * 70,
            "NOTES",
            "=" * 70,
            "",
            "- Precision: Fraction of predictions that were correct (1 - false alarm rate)",
            "- Recall: Fraction of actual flares that were predicted (detection rate)",
            "- F1 Score: Harmonic mean of precision and recall",
            "- Brier Score: Lower is better (0 = perfect, 1 = worst)",
            "",
            "Historical backtesting provides validation on past solar activity but",
            "does not guarantee future performance. Solar cycles vary significantly.",
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
    )

    # generate report
    report = generate_backtest_report(
        metrics=metrics,
        model_path=model_path,
        start_date=start_date,
        end_date=end_date,
        output_path=PROJECT_ROOT / args.output,
    )

    print("\n" + report)

    # summary
    if metrics.get("f1_score", 0) > 0.6:
        logger.info("✓ BACKTEST PASSED - Model shows good performance")
        sys.exit(0)
    elif metrics.get("f1_score", 0) > 0.4:
        logger.warning("~ BACKTEST MARGINAL - Model shows moderate performance")
        sys.exit(0)
    else:
        logger.error("✗ BACKTEST FAILED - Model shows poor performance")
        sys.exit(1)


if __name__ == "__main__":
    main()
