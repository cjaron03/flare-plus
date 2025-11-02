"""run flask api server for model serving."""

import argparse
import logging
import sys
from pathlib import Path

import joblib

# add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import create_app  # noqa: E402
from src.models.pipeline import ClassificationPipeline  # noqa: E402
from src.models.survival_pipeline import SurvivalAnalysisPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_pipeline(model_path: str, pipeline_type: str):
    """load trained pipeline from file."""
    try:
        logger.info(f"loading {pipeline_type} pipeline from {model_path}")
        pipeline_data = joblib.load(model_path)

        if pipeline_type == "classification":
            pipeline = ClassificationPipeline()
            pipeline.models = pipeline_data.get("models", {})
            pipeline.evaluation_results = pipeline_data.get("evaluation_results", {})
        elif pipeline_type == "survival":
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
        else:
            raise ValueError(f"unknown pipeline type: {pipeline_type}")

        logger.info(f"{pipeline_type} pipeline loaded successfully")
        return pipeline

    except Exception as e:
        logger.error(f"failed to load {pipeline_type} pipeline: {e}")
        return None


def main():
    """main entry point for api server."""
    parser = argparse.ArgumentParser(description="run flask api server for model serving")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",  # nosec B104 - default to localhost, can override for production
        help="host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="port to bind to (default: 5000)",
    )
    parser.add_argument(
        "--classification-model",
        type=str,
        help="path to trained classification model (joblib)",
    )
    parser.add_argument(
        "--survival-model",
        type=str,
        help="path to trained survival model (joblib)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of gunicorn workers (default: 2)",
    )

    args = parser.parse_args()

    # load pipelines
    classification_pipeline = None
    survival_pipeline = None

    if args.classification_model:
        classification_pipeline = load_pipeline(args.classification_model, "classification")

    if args.survival_model:
        survival_pipeline = load_pipeline(args.survival_model, "survival")

    if not classification_pipeline and not survival_pipeline:
        logger.warning("no models loaded - api will be in degraded mode")

    # create flask app
    app = create_app(
        classification_pipeline=classification_pipeline,
        survival_pipeline=survival_pipeline,
    )

    logger.info(f"starting api server on {args.host}:{args.port}")
    logger.info("endpoints:")
    logger.info("  GET  /health - health check")
    logger.info("  POST /predict/classification - classification prediction")
    logger.info("  POST /predict/survival - survival prediction")
    logger.info("  POST /predict/all - combined predictions")

    # run with gunicorn if available, otherwise use flask dev server
    try:
        import gunicorn.app.base

        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        options = {
            "bind": f"{args.host}:{args.port}",
            "workers": args.workers,
            "worker_class": "sync",
            "timeout": 120,
            "accesslog": "-",
            "errorlog": "-",
        }

        StandaloneApplication(app, options).run()

    except ImportError:
        logger.warning("gunicorn not available, using flask dev server")
        app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
