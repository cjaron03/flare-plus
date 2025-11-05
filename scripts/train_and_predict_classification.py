"""train classification model and make predictions for 24-48h flare classes."""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

import joblib

# add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pipeline import ClassificationPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_data_availability(start_date: datetime, end_date: datetime) -> dict:
    """
    check if sufficient data is available for training.

    args:
        start_date: training start date
        end_date: training end date

    returns:
        dict with data availability stats
    """
    pipeline = ClassificationPipeline()

    logger.info(f"checking data availability from {start_date.date()} to {end_date.date()}...")

    try:
        dataset = pipeline.prepare_dataset(
            start_date=start_date,
            end_date=end_date,
            sample_interval_hours=1,
        )

        if len(dataset) == 0:
            return {
                "available": False,
                "reason": "no data found in date range",
                "samples": 0,
            }

        # check for labels
        has_24h = "label_24h" in dataset.columns
        has_48h = "label_48h" in dataset.columns

        if not (has_24h or has_48h):
            return {
                "available": False,
                "reason": "no labels found (need flare events)",
                "samples": len(dataset),
            }

        # check class distribution
        stats = {
            "available": True,
            "samples": len(dataset),
            "has_24h": has_24h,
            "has_48h": has_48h,
        }

        if has_24h:
            label_counts = dataset["label_24h"].value_counts().to_dict()
            stats["24h_distribution"] = label_counts

        if has_48h:
            label_counts = dataset["label_48h"].value_counts().to_dict()
            stats["48h_distribution"] = label_counts

        return stats

    except Exception as e:
        logger.error(f"error checking data: {e}")
        return {
            "available": False,
            "reason": str(e),
            "samples": 0,
        }


def train_model(
    start_date: datetime,
    end_date: datetime,
    sample_interval_hours: int = 1,
    region_number: int = None,
    models: list = None,
    save_path: str = None,
) -> ClassificationPipeline:
    """
    train classification model.

    args:
        start_date: training start date
        end_date: training end date
        sample_interval_hours: hours between samples
        region_number: optional region number
        models: list of model types to train (["logistic", "gradient_boosting"] or None for both)
        save_path: path to save trained model

    returns:
        trained pipeline
    """
    if models is None:
        models = ["logistic", "gradient_boosting"]

    pipeline = ClassificationPipeline()

    logger.info(f"preparing dataset from {start_date.date()} to {end_date.date()}...")
    dataset = pipeline.prepare_dataset(
        start_date=start_date,
        end_date=end_date,
        sample_interval_hours=sample_interval_hours,
        region_number=region_number,
    )

    if len(dataset) == 0:
        raise ValueError("no data available for training")

    logger.info(f"dataset size: {len(dataset)} samples")

    # train models
    logger.info(f"training models: {models}")
    run_name = f"classification_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    pipeline.train_and_evaluate(
        dataset=dataset,
        test_size=0.2,
        models=models,
        run_name=run_name,
    )

    if not pipeline.models:
        error_msg = (
            "training failed - no models could be trained.\n"
            "check logs above for specific errors.\n"
            "common issues:\n"
            "  - non-numeric features in dataset (string/object columns)\n"
            "  - insufficient data for training\n"
            "  - class imbalance too severe"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("training completed successfully!")
    logger.info(f"trained models for windows: {list(pipeline.models.keys())}")

    # save model if requested
    if save_path:
        logger.info(f"saving model to {save_path}...")
        model_data = {
            "models": pipeline.models,
            "evaluation_results": pipeline.evaluation_results,
            "target_windows": [24, 48],
            "trained_at": datetime.now().isoformat(),
        }
        joblib.dump(model_data, save_path)
        logger.info(f"model saved to {save_path}")

    return pipeline


def make_prediction(
    pipeline: ClassificationPipeline,
    timestamp: datetime,
    window: int = 24,
    model_type: str = "gradient_boosting",
    region_number: int = None,
) -> dict:
    """
    make classification prediction.

    args:
        pipeline: trained pipeline
        timestamp: observation timestamp
        window: prediction window (24 or 48 hours)
        model_type: model type ("logistic" or "gradient_boosting")
        region_number: optional region number

    returns:
        prediction dict
    """
    return pipeline.predict(
        timestamp=timestamp,
        window=window,
        model_type=model_type,
        region_number=region_number,
    )


def format_prediction(prediction: dict) -> str:
    """format prediction for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("classification prediction")
    lines.append("=" * 60)
    lines.append(f"timestamp: {prediction['timestamp']}")
    lines.append(f"window: {prediction['window']}h")
    lines.append(f"model: {prediction['model_type'].upper()}")
    lines.append("")
    lines.append("class probabilities:")
    lines.append("-" * 60)

    for class_name, prob in prediction["probabilities"].items():
        lines.append(f"  {class_name:10s} {prob*100:6.2f}%")

    lines.append("")
    lines.append(f"predicted class: {prediction['predicted_class']}")
    lines.append(f"confidence: {prediction['confidence']*100:.2f}%")

    return "\n".join(lines)


def main():
    """main entry point."""
    parser = argparse.ArgumentParser(description="train classification model and predict 24-48h flare classes")
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
        "--sample-interval-hours",
        type=int,
        default=1,
        help="hours between observation timestamps (default: 1)",
    )
    parser.add_argument(
        "--region",
        type=int,
        help="optional region number to focus on",
    )
    parser.add_argument(
        "--window",
        type=int,
        choices=[24, 48],
        default=24,
        help="prediction window in hours (default: 24)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "gradient_boosting"],
        default="gradient_boosting",
        help="model type for prediction (default: gradient_boosting)",
    )
    parser.add_argument(
        "--train-models",
        type=str,
        nargs="+",
        choices=["logistic", "gradient_boosting"],
        help="models to train (default: both)",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        help="save trained model to file (joblib format)",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        help="load trained model from file (joblib format)",
    )

    args = parser.parse_args()

    # default dates
    end_date = datetime.now()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    start_date = end_date - timedelta(days=90)
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    # train model
    if args.train:
        # check data availability
        stats = check_data_availability(start_date, end_date)
        if not stats["available"]:
            logger.error(f"insufficient data: {stats['reason']}")
            logger.error(f"samples: {stats.get('samples', 0)}")
            logger.error("suggestions:")
            logger.error("  - expand date range (--start-date)")
            logger.error("  - ensure data ingestion has run (./flare ingest)")
            logger.error("  - check for flare events in database")
            sys.exit(1)

        logger.info(f"data availability: {stats['samples']} samples")
        if "24h_distribution" in stats:
            logger.info(f"24h class distribution: {stats['24h_distribution']}")
        if "48h_distribution" in stats:
            logger.info(f"48h class distribution: {stats['48h_distribution']}")

        # train
        pipeline = train_model(
            start_date=start_date,
            end_date=end_date,
            sample_interval_hours=args.sample_interval_hours,
            region_number=args.region,
            models=args.train_models,
            save_path=args.save_model,
        )

        logger.info("training completed successfully!")

        # make a test prediction if requested
        if args.predict:
            logger.info("making test prediction...")
            prediction = make_prediction(
                pipeline=pipeline,
                timestamp=end_date,
                window=args.window,
                model_type=args.model,
                region_number=args.region,
            )
            print("\n" + format_prediction(prediction))

    # load model and predict
    elif args.load_model:
        logger.info(f"loading model from {args.load_model}...")
        model_data = joblib.load(args.load_model)

        pipeline = ClassificationPipeline()
        pipeline.models = model_data.get("models", {})
        pipeline.evaluation_results = model_data.get("evaluation_results", {})

        if not pipeline.models:
            logger.error("model file contains no trained models")
            sys.exit(1)

        logger.info("model loaded successfully")

        if args.predict:
            prediction = make_prediction(
                pipeline=pipeline,
                timestamp=end_date,
                window=args.window,
                model_type=args.model,
                region_number=args.region,
            )
            print("\n" + format_prediction(prediction))

    # predict only (requires loaded model)
    elif args.predict:
        logger.error("cannot predict without training or loading a model")
        logger.error("use --train or --load-model first")
        sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
