#!/usr/bin/env python
"""train survival analysis model and make predictions for x-class flares."""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import json

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging  # noqa: E402

import pandas as pd  # noqa: E402
from sqlalchemy import func  # noqa: E402

from src.data.database import get_database  # noqa: E402
from src.data.schema import FlareEvent  # noqa: E402
from src.models.survival_pipeline import SurvivalAnalysisPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_historical_timestamps(
    start_date: datetime,
    end_date: datetime,
    interval_hours: int = 12,
    use_data_availability: bool = True,
) -> list:
    """
    generate timestamps at regular intervals for training.

    args:
        start_date: start of historical period
        end_date: end of historical period
        interval_hours: hours between timestamps
        use_data_availability: only generate timestamps where flux data exists

    returns:
        list of timestamps
    """
    if not use_data_availability:
        # simple interval-based generation
        timestamps = []
        current = start_date
        while current <= end_date:
            timestamps.append(current)
            current += timedelta(hours=interval_hours)
        return timestamps

    # generate timestamps based on available flux data
    from src.data.persistence import DataPersister

    persister = DataPersister()

    try:
        # get flux data range
        flux_df = persister.get_xray_flux_range(start_date, end_date)

        if len(flux_df) == 0:
            logger.warning("no flux data in date range, using simple interval generation")
            timestamps = []
            current = start_date
            while current <= end_date:
                timestamps.append(current)
                current += timedelta(hours=interval_hours)
            return timestamps

        # generate timestamps aligned with available flux data
        timestamps = []
        flux_timestamps = pd.to_datetime(flux_df["timestamp"]).sort_values()

        # sample at intervals from available data
        current_idx = 0
        while current_idx < len(flux_timestamps):
            timestamp = flux_timestamps.iloc[current_idx]
            if start_date <= timestamp <= end_date:
                timestamps.append(timestamp.to_pydatetime())
            # advance by interval_hours worth of samples (assuming ~5min samples)
            samples_per_hour = 12  # approximately 5-minute intervals
            current_idx += max(1, interval_hours * samples_per_hour)

        logger.info(f"generated {len(timestamps)} timestamps from available flux data")
        return timestamps

    except Exception as e:
        logger.warning(f"error getting flux data range: {e}, using simple interval generation")
        timestamps = []
        current = start_date
        while current <= end_date:
            timestamps.append(current)
            current += timedelta(hours=interval_hours)
        return timestamps


def check_data_availability(start_date: datetime, end_date: datetime, target_class: str = "X") -> dict:
    """
    check if we have enough data for training.

    args:
        start_date: start of period
        end_date: end of period
        target_class: target flare class to check

    returns:
        dict with data availability stats
    """
    db = get_database()
    stats = {
        "c_class_flares": 0,
        "m_class_flares": 0,
        "x_class_flares": 0,
        "target_class_flares": 0,
        "total_flares": 0,
        "period_days": (end_date - start_date).days,
        "sufficient": False,
    }

    try:
        with db.get_session() as session:
            # count flares by class
            for flare_class in ["C", "M", "X"]:
                count = (
                    session.query(FlareEvent)
                    .filter(
                        FlareEvent.start_time >= start_date,
                        FlareEvent.start_time <= end_date,
                        FlareEvent.class_category == flare_class,
                    )
                    .count()
                )
                stats[f"{flare_class.lower()}_class_flares"] = count

            # count total flares
            total_flares = (
                session.query(FlareEvent)
                .filter(
                    FlareEvent.start_time >= start_date,
                    FlareEvent.start_time <= end_date,
                )
                .count()
            )

            stats["total_flares"] = total_flares
            stats["target_class_flares"] = stats[f"{target_class.lower()}_class_flares"]

            # check if sufficient for training
            min_required = {"C": 200, "M": 50, "X": 10}
            stats["sufficient"] = stats["target_class_flares"] >= min_required.get(target_class, 10)

    except Exception as e:
        logger.error(f"error checking data availability: {e}")

    return stats


def train_model(
    start_date: datetime,
    end_date: datetime,
    interval_hours: int = 12,
    region_number: int = None,
    model_types: list = None,
    target_flare_class: str = "X",
) -> SurvivalAnalysisPipeline:
    """
    train survival analysis model on historical data.

    args:
        start_date: start of training period
        end_date: end of training period
        interval_hours: hours between observation timestamps
        region_number: optional region number to focus on
        model_types: list of models to train (["cox", "gb"] or None for both)

    returns:
        trained pipeline
    """
    logger.info(f"training survival model from {start_date} to {end_date}")
    logger.info(f"target flare class: {target_flare_class}")

    # check data availability
    stats = check_data_availability(start_date, end_date, target_flare_class)

    print("\n" + "=" * 70)
    print("DATA AVAILABILITY CHECK")
    print("=" * 70)
    print(f"Period: {start_date.date()} to {end_date.date()} ({stats['period_days']} days)")
    print(f"C-class flares: {stats['c_class_flares']}")
    print(f"M-class flares: {stats['m_class_flares']}")
    print(f"X-class flares: {stats['x_class_flares']}")
    print(f"Target ({target_flare_class}-class): {stats['target_class_flares']}")
    print("=" * 70)

    min_required = {"C": 200, "M": 50, "X": 10}
    required = min_required.get(target_flare_class, 10)

    if not stats["sufficient"]:
        print("\nWARNING: INSUFFICIENT TRAINING DATA")
        print(f"You have {stats['target_class_flares']} {target_flare_class}-class flares")
        print(f"Minimum recommended: {required}")
        print("\nThis model will likely OVERFIT and produce unreliable predictions.")
        print("Predictions are for DEMONSTRATION/LEARNING purposes only.")
        print("\nSee docs/DATA_COLLECTION_GUIDE.md for how to get more data.")
        print("=" * 70)
        print("\nProceeding with training (demonstration mode)...")
        print("=" * 70 + "\n")
    else:
        print(f"\n[OK] Sufficient data for {target_flare_class}-class training")
        print("=" * 70 + "\n")

    # check what classes we actually have
    db = get_database()
    try:
        with db.get_session() as session:
            class_counts = (
                session.query(FlareEvent.class_category, func.count(FlareEvent.id))
                .filter(
                    FlareEvent.start_time >= start_date,
                    FlareEvent.start_time <= end_date,
                )
                .group_by(FlareEvent.class_category)
                .all()
            )
            if class_counts:
                logger.info("flare class breakdown:")
                for class_cat, count in class_counts:
                    logger.info(f"  {class_cat}-class: {count}")
    except Exception as e:
        logger.debug(f"error getting class breakdown: {e}")

    # use stats from check_data_availability
    target_flares = stats["target_class_flares"]
    total_flares = stats["total_flares"]

    if target_flares < 5:
        logger.warning(
            f"low {target_flare_class}-class flare count ({target_flares}). "
            "model may not train well. consider using --target-class C (most common)."
        )

        # suggest using C-class if we have any
        if total_flares > 0 and target_flares == 0:
            logger.warning(
                f"detected {total_flares} flares but none are {target_flare_class}-class. "
                "try --target-class C instead."
            )

    # create pipeline with target class
    pipeline = SurvivalAnalysisPipeline(
        target_flare_class=target_flare_class,
        max_time_hours=168,
    )

    # get actual data range first (when we have flux data)
    from src.data.persistence import DataPersister

    persister = DataPersister()
    flux_df = persister.get_xray_flux_range(start_date, end_date)

    if len(flux_df) == 0:
        raise ValueError("no flux data available in date range - run data ingestion first")

    data_start = pd.to_datetime(flux_df["timestamp"].min()).to_pydatetime()
    data_end = pd.to_datetime(flux_df["timestamp"].max()).to_pydatetime()

    # max lookback needed is 24 hours (from time_varying_covariates config)
    max_lookback = timedelta(hours=24)
    min_observation_time = data_start + max_lookback

    logger.info(f"flux data available from {data_start} to {data_end}")
    logger.info(
        f"generating observations from {min_observation_time} (to allow {max_lookback.total_seconds()/3600}h lookback)"
    )

    # generate timestamps only within data coverage window
    timestamps = []

    # start from min_observation_time (after data start + lookback)
    # end at data_end or end_date, whichever is earlier
    effective_end = min(data_end, end_date)

    # generate regular interval timestamps
    current = min_observation_time
    while current <= effective_end:
        timestamps.append(current)
        current += timedelta(hours=interval_hours)

    # also add timestamps before detected flares (for event observations)
    try:
        with db.get_session() as session:
            flare_times = (
                session.query(FlareEvent.start_time)
                .filter(
                    FlareEvent.start_time >= min_observation_time,
                    FlareEvent.start_time <= effective_end,
                )
                .order_by(FlareEvent.start_time)
                .all()
            )

            if flare_times:
                # add observation timestamps 12h, 24h, 48h before each flare
                for (flare_time,) in flare_times:
                    # ensure flare_time is datetime
                    if hasattr(flare_time, "to_pydatetime"):
                        flare_time = flare_time.to_pydatetime()

                    # add observations before flares
                    for hours_before in [12, 24, 48]:
                        obs_time = flare_time - timedelta(hours=hours_before)
                        # only add if we have data coverage
                        if obs_time >= min_observation_time and obs_time <= effective_end:
                            if obs_time not in timestamps:
                                timestamps.append(obs_time)

    except Exception as e:
        logger.debug(f"error adding flare-based timestamps: {e}")

    timestamps = sorted(set(timestamps))  # remove duplicates and sort
    logger.info(f"generated {len(timestamps)} observation timestamps (with data coverage)")

    # prepare dataset
    dataset = pipeline.prepare_dataset(timestamps, region_number)

    if len(dataset) == 0:
        raise ValueError("no dataset prepared - check data availability in database")

    logger.info(f"dataset prepared: {len(dataset)} samples")
    logger.info(f"event rate: {dataset['event'].mean():.2%}")

    # train models
    results = pipeline.train_and_evaluate(
        dataset,
        test_size=0.2,
        models=model_types,
    )

    logger.info("training complete:")
    logger.info(f"  train size: {results['train_size']}")
    logger.info(f"  test size: {results['test_size']}")

    if results.get("cox_trained"):
        logger.info(f"  cox c-index (test): {results.get('cox_c_index_test', 'N/A')}")
    else:
        logger.warning("  cox model failed to train")

    if results.get("gb_trained"):
        logger.info(f"  gb c-index (test): {results.get('gb_c_index_test', 'N/A')}")
    else:
        logger.warning("  gb model failed to train")

    if not pipeline.is_fitted:
        error_msg = (
            "training failed - no models could be trained.\n"
            "likely causes:\n"
            "  1. insufficient flare events in database (need at least 5+ for training)\n"
            "  2. all features are constant (no variation in data)\n"
            "  3. insufficient data in date range\n\n"
            "solutions:\n"
            "  - run data ingestion: python scripts/run_ingestion.py\n"
            "  - expand date range: --start-date YYYY-MM-DD\n"
            "  - use different flare class: --target-class C or M (may have more data)\n"
            "  - check database: ensure flare_events table has flare data"
        )
        raise ValueError(error_msg)

    return pipeline


def make_prediction(
    pipeline: SurvivalAnalysisPipeline,
    timestamp: datetime = None,
    region_number: int = None,
    model_type: str = "cox",
) -> dict:
    """
    make prediction for x-class flare timing.

    args:
        pipeline: trained survival pipeline
        timestamp: prediction timestamp (defaults to now)
        region_number: optional region number
        model_type: "cox" or "gb"

    returns:
        prediction dict
    """
    if timestamp is None:
        timestamp = datetime.utcnow()

    logger.info(f"making prediction for {timestamp} (model: {model_type})")

    try:
        prediction = pipeline.predict_survival_probabilities(
            timestamp,
            region_number,
            model_type=model_type,
        )

        return prediction

    except Exception as e:
        logger.error(f"error making prediction: {e}")
        raise


def format_prediction(prediction: dict) -> str:
    """
    format prediction results for display.

    args:
        prediction: prediction dict from pipeline

    returns:
        formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SOLAR FLARE TIME-TO-EVENT PREDICTION")
    lines.append("=" * 60)
    lines.append(f"timestamp: {prediction['timestamp']}")
    lines.append(f"model: {prediction['model_type'].upper()}")
    lines.append(f"hazard score: {prediction['hazard_score']:.4f}")
    lines.append("")
    lines.append("probability distribution (flare in time bucket):")
    lines.append("-" * 60)

    # sort time buckets
    prob_dist = prediction["probability_distribution"]
    sorted_buckets = sorted(prob_dist.items(), key=lambda x: float(x[0].split("-")[0].replace("h", "")))

    for bucket, prob in sorted_buckets:
        # show more precision if probabilities are very small
        if prob > 0 and prob < 0.01:
            lines.append(f"  {bucket:15s} {prob*100:6.4f}%")
        else:
            lines.append(f"  {bucket:15s} {prob*100:6.2f}%")

    lines.append("")
    lines.append("interpretation:")
    lines.append("  - higher hazard score = higher risk of flare")
    lines.append("  - probability shows chance of flare in each time bucket")
    lines.append("  - example: '24h-48h: 15%' means 15% chance between 24-48 hours")

    return "\n".join(lines)


def main():
    """main entry point."""
    parser = argparse.ArgumentParser(description="train survival analysis model and predict x-class flare timing")
    parser.add_argument(
        "--train",
        action="store_true",
        help="train the model (required before prediction)",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="make prediction for current time",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="training start date (YYYY-MM-DD, defaults to 90 days ago)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="training end date (YYYY-MM-DD, defaults to today)",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=12,
        help="hours between observation timestamps (default: 12)",
    )
    parser.add_argument(
        "--region",
        type=int,
        help="optional region number to focus on",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cox", "gb"],
        default="cox",
        help="model type for prediction (default: cox)",
    )
    parser.add_argument(
        "--train-models",
        type=str,
        nargs="+",
        choices=["cox", "gb"],
        help="models to train (default: both)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output file for prediction (json format)",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        help="save trained model to file",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        help="load trained model from file (skips training)",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        choices=["X", "M", "C"],
        default="X",
        help="target flare class for time-to-event prediction (default: X)",
    )
    parser.add_argument(
        "--detect-flares",
        action="store_true",
        help="detect flares from historical flux data before training",
    )

    args = parser.parse_args()

    pipeline = None

    # load existing model if provided
    if args.load_model:
        logger.info(f"loading model from {args.load_model}")
        pipeline = SurvivalAnalysisPipeline.load_model(args.load_model)
        logger.info("model loaded")

    # detect flares from historical data if requested
    if args.detect_flares:
        logger.info("detecting flares from historical flux data...")
        from src.data.flare_detector import FlareDetector
        from src.data.persistence import DataPersister

        detector = FlareDetector()
        persister = DataPersister()

        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        else:
            end_date = datetime.utcnow()

        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        else:
            # default: 90 days ago
            start_date = end_date - timedelta(days=90)

        flares_df = detector.detect_flares_from_database(start_date, end_date, min_class="C")

        if flares_df is not None and len(flares_df) > 0:
            logger.info(f"detected {len(flares_df)} flare events from historical flux data")
            persister.save_flare_events(flares_df, source_name="historical_detection")
            logger.info("flare events saved to database")
        else:
            logger.warning("no flares detected in historical flux data")

    # train if requested and not loaded
    if args.train and pipeline is None:
        # default dates
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        else:
            end_date = datetime.utcnow()

        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        else:
            # default: 90 days ago
            start_date = end_date - timedelta(days=90)

        try:
            pipeline = train_model(
                start_date,
                end_date,
                interval_hours=args.interval_hours,
                region_number=args.region,
                model_types=args.train_models,
                target_flare_class=args.target_class,
            )
            logger.info("model training complete")

            # save model if requested
            if args.save_model and pipeline.is_fitted:
                pipeline.save_model(args.save_model)
            elif args.save_model:
                logger.warning("model not saved - training failed")

        except ValueError as e:
            logger.error(f"training failed: {e}")
            if args.predict:
                logger.error("cannot make prediction without trained model")
            sys.exit(1)

    # predict if requested
    if args.predict:
        if pipeline is None:
            logger.error("no trained model available. use --train or --load-model.")
            sys.exit(1)

        if not pipeline.is_fitted:
            logger.error("model is not fitted. training may have failed. check logs above.")
            sys.exit(1)

        # check if requested model is actually trained
        if args.model == "cox" and not pipeline.cox_model.is_fitted:
            logger.error("cox model not trained. try --model gb or check training logs.")
            sys.exit(1)
        elif args.model == "gb" and not pipeline.gb_model.is_fitted:
            logger.error("gb model not trained. try --model cox or check training logs.")
            sys.exit(1)

        prediction = make_prediction(
            pipeline,
            timestamp=datetime.utcnow(),
            region_number=args.region,
            model_type=args.model,
        )

        # display results
        print(format_prediction(prediction))

        # save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(prediction, f, indent=2, default=str)
            logger.info(f"prediction saved to {args.output}")

    if not args.train and not args.predict:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
