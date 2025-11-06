#!/usr/bin/env python3
"""setup mlflow tracking server and default experiment."""

import argparse
import logging

import mlflow
from src.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# mlflow config
MLFLOW_CONFIG = CONFIG.get("mlflow", {})
DEFAULT_TRACKING_URI = MLFLOW_CONFIG.get("tracking_uri", "sqlite:///mlruns.db")
DEFAULT_EXPERIMENT_NAME = MLFLOW_CONFIG.get("experiment_name", "flare-plus")


def setup_mlflow(tracking_uri: str = None, experiment_name: str = None):
    """
    initialize mlflow tracking server and create default experiment.

    args:
        tracking_uri: mlflow tracking uri (default: from config)
        experiment_name: experiment name (default: from config)
    """
    tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
    experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME

    logger.info(f"setting up mlflow with tracking uri: {tracking_uri}")

    # set tracking uri
    mlflow.set_tracking_uri(tracking_uri)

    # create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"created experiment: {experiment_name} (id: {experiment_id})")
        else:
            logger.info(f"experiment already exists: {experiment_name} (id: {experiment.experiment_id})")
    except Exception as e:
        logger.error(f"failed to setup experiment: {e}", exc_info=True)
        raise

    logger.info(f"mlflow setup complete. experiment: {experiment_name}")
    logger.info(f"tracking uri: {tracking_uri}")
    logger.info(f"to view ui, run: mlflow ui --backend-store-uri {tracking_uri}")


def main():
    """main entry point."""
    parser = argparse.ArgumentParser(description="setup mlflow tracking")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="mlflow tracking uri (default: from config.yaml)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="experiment name (default: from config.yaml)",
    )

    args = parser.parse_args()

    setup_mlflow(tracking_uri=args.tracking_uri, experiment_name=args.experiment_name)


if __name__ == "__main__":
    main()
