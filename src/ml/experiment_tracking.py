"""mlflow experiment tracking wrapper for flare+ models."""

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import CONFIG

logger = logging.getLogger(__name__)

# mlflow config from config.yaml
MLFLOW_CONFIG = CONFIG.get("mlflow", {})
DEFAULT_TRACKING_URI = MLFLOW_CONFIG.get("tracking_uri", "sqlite:///mlruns.db")
DEFAULT_EXPERIMENT_NAME = MLFLOW_CONFIG.get("experiment_name", "flare-plus")
MODEL_REGISTRY_NAME = MLFLOW_CONFIG.get("model_registry_name", "flare-plus-models")


class MLflowTracker:
    """wrapper for mlflow experiment tracking."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        model_registry_name: Optional[str] = None,
    ):
        """
        initialize mlflow tracker.

        args:
            tracking_uri: mlflow tracking uri (default: from config)
            experiment_name: experiment name (default: from config)
            model_registry_name: model registry name (default: from config)
        """
        self.tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
        self.experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
        self.model_registry_name = model_registry_name or MODEL_REGISTRY_NAME

        # set tracking uri
        mlflow.set_tracking_uri(self.tracking_uri)

        # create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"created experiment: {self.experiment_name} (id: {experiment_id})")
            else:
                logger.info(f"using existing experiment: {self.experiment_name} (id: {experiment.experiment_id})")
        except Exception as e:
            logger.warning(f"failed to setup experiment: {e}, using default")
            mlflow.set_experiment(self.experiment_name)

        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> mlflow.ActiveRun:
        """
        start a new mlflow run.

        args:
            run_name: optional name for the run
            tags: optional tags dict

        returns:
            active mlflow run
        """
        mlflow.set_experiment(self.experiment_name)

        tags = tags or {}
        tags.setdefault("training_date", datetime.utcnow().isoformat())

        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Any]):
        """log parameters to current run."""
        try:
            mlflow.log_params(params)
            logger.debug(f"logged {len(params)} parameters")
        except Exception as e:
            logger.warning(f"failed to log parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """log metrics to current run."""
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"logged {len(metrics)} metrics")
        except Exception as e:
            logger.warning(f"failed to log metrics: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        model_type: str = "sklearn",
    ) -> str:
        """
        log model to mlflow.

        args:
            model: trained model object
            artifact_path: path within run artifacts
            signature: optional model signature
            input_example: optional input example
            model_type: model type ('sklearn', 'custom')

        returns:
            model uri
        """
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(
                    model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                )
            else:
                # custom model logging
                mlflow.pyfunc.log_model(
                    artifact_path=artifact_path,
                    python_model=model,
                    signature=signature,
                    input_example=input_example,
                )

            model_uri = mlflow.get_artifact_uri(artifact_path)
            logger.info(f"logged model to: {model_uri}")
            return model_uri
        except Exception as e:
            logger.error(f"failed to log model: {e}", exc_info=True)
            raise

    def log_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        dataset_info: Optional[Dict[str, Any]] = None,
    ):
        """
        log dataset information.

        args:
            dataset_path: path to dataset file
            dataset_name: name of dataset
            dataset_info: optional dataset metadata
        """
        try:
            if os.path.exists(dataset_path):
                mlflow.log_artifact(dataset_path, artifact_path="datasets")
                logger.info(f"logged dataset: {dataset_name}")

            if dataset_info:
                mlflow.log_params({f"dataset_{k}": str(v) for k, v in dataset_info.items()})
        except Exception as e:
            logger.warning(f"failed to log dataset: {e}")

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        register model in mlflow model registry.

        args:
            model_uri: uri of logged model
            model_name: name for registered model
            tags: optional tags

        returns:
            registered model version
        """
        try:
            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"registered model: {model_name} version {result.version}")

            if tags:
                for k, v in tags.items():
                    self.client.set_registered_model_tag(model_name, k, v)

            return result.version
        except Exception as e:
            logger.error(f"failed to register model: {e}", exc_info=True)
            raise

    def load_model(self, model_uri: str) -> Any:
        """
        load model from mlflow.

        args:
            model_uri: model uri or registered model name:version

        returns:
            loaded model
        """
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"loaded model from: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"failed to load model: {e}", exc_info=True)
            raise

    def get_latest_model_version(self, model_name: str) -> Optional[str]:
        """
        get latest version of registered model.

        args:
            model_name: registered model name

        returns:
            latest version string or None
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=["None"])
            if versions:
                return versions[0].version
            return None
        except Exception as e:
            logger.warning(f"failed to get latest model version: {e}")
            return None

    def end_run(self, status: str = "FINISHED"):
        """end current mlflow run."""
        try:
            mlflow.end_run(status=status)
            logger.debug("ended mlflow run")
        except Exception as e:
            logger.warning(f"failed to end run: {e}")
