"""unified prediction service wrapping classification and survival models."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.models.pipeline import ClassificationPipeline
from src.models.survival_pipeline import SurvivalAnalysisPipeline
from src.api.monitoring import InputDriftDetector, OutcomeLogger

logger = logging.getLogger(__name__)


class PredictionService:
    """unified service for both classification and survival predictions."""

    def __init__(
        self,
        classification_pipeline: Optional[ClassificationPipeline] = None,
        survival_pipeline: Optional[SurvivalAnalysisPipeline] = None,
        enable_drift_detection: bool = True,
        enable_outcome_logging: bool = True,
    ):
        """
        initialize prediction service.

        args:
            classification_pipeline: optional pre-trained classification pipeline
            survival_pipeline: optional pre-trained survival analysis pipeline
            enable_drift_detection: enable input drift detection
            enable_outcome_logging: enable outcome logging
        """
        self.classification_pipeline = classification_pipeline
        self.survival_pipeline = survival_pipeline

        # monitoring components
        self.drift_detector = InputDriftDetector() if enable_drift_detection else None
        self.outcome_logger = OutcomeLogger() if enable_outcome_logging else None

    def predict_classification(
        self,
        timestamp: datetime,
        window: int = 24,
        model_type: str = "gradient_boosting",
        region_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        make classification prediction (short-term flare class).

        args:
            timestamp: observation timestamp
            window: prediction window in hours (24 or 48)
            model_type: model type ('logistic' or 'gradient_boosting')
            region_number: optional region number

        returns:
            prediction dict with class probabilities
        """
        if self.classification_pipeline is None:
            raise ValueError("classification pipeline not initialized")

        if not self.classification_pipeline.models:
            raise ValueError("classification models not trained")

        try:
            # detect drift if enabled
            drift_info = None
            if self.drift_detector:
                try:
                    # compute features to check for drift
                    features_df = self.classification_pipeline.feature_engineer.compute_features(
                        timestamp=timestamp,
                        region_number=region_number,
                        normalize=False,
                        standardize=False,
                        handle_missing=True,
                    )
                    if len(features_df) > 0:
                        drift_info = self.drift_detector.detect_drift(features_df, timestamp)
                        # update reference distribution
                        self.drift_detector.update_reference(features_df, timestamp)
                except Exception as e:
                    logger.warning(f"drift detection failed: {e}")

            result = self.classification_pipeline.predict(
                timestamp=timestamp,
                window=window,
                model_type=model_type,
                region_number=region_number,
            )

            # add drift info if available
            if drift_info:
                result["drift_detection"] = drift_info

            # log prediction if enabled
            if self.outcome_logger:
                self.outcome_logger.log_prediction(
                    prediction=result,
                    prediction_type="classification",
                    timestamp=timestamp,
                    region_number=region_number,
                )

            return result
        except Exception as e:
            logger.error(f"classification prediction failed: {e}")
            raise

    def predict_survival(
        self,
        timestamp: datetime,
        model_type: str = "cox",
        region_number: Optional[int] = None,
        time_buckets: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        make survival prediction (time-to-event probabilities).

        args:
            timestamp: observation timestamp
            model_type: model type ('cox' or 'gb')
            region_number: optional region number
            time_buckets: optional custom time buckets

        returns:
            prediction dict with survival probabilities over time buckets
        """
        if self.survival_pipeline is None:
            raise ValueError("survival pipeline not initialized")

        if not self.survival_pipeline.is_fitted:
            raise ValueError("survival model not trained")

        try:
            # detect drift if enabled
            drift_info = None
            if self.drift_detector:
                try:
                    # compute covariates to check for drift
                    covariates_df = self.survival_pipeline.covariate_engineer.compute_time_varying_covariates(
                        timestamp=timestamp,
                        region_number=region_number,
                    )
                    if len(covariates_df) > 0:
                        drift_info = self.drift_detector.detect_drift(covariates_df, timestamp)
                        # update reference distribution
                        self.drift_detector.update_reference(covariates_df, timestamp)
                except Exception as e:
                    logger.warning(f"drift detection failed: {e}")

            result = self.survival_pipeline.predict_survival_probabilities(
                timestamp=timestamp,
                region_number=region_number,
                model_type=model_type,
                time_buckets=time_buckets,
            )

            # add drift info if available
            if drift_info:
                result["drift_detection"] = drift_info

            # log prediction if enabled
            if self.outcome_logger:
                self.outcome_logger.log_prediction(
                    prediction=result,
                    prediction_type="survival",
                    timestamp=timestamp,
                    region_number=region_number,
                )

            return result
        except Exception as e:
            logger.error(f"survival prediction failed: {e}")
            raise

    def predict_all(
        self,
        timestamp: datetime,
        region_number: Optional[int] = None,
        classification_windows: list = [24, 48],
        survival_model_type: str = "cox",
    ) -> Dict[str, Any]:
        """
        make both classification and survival predictions.

        args:
            timestamp: observation timestamp
            region_number: optional region number
            classification_windows: list of windows for classification (default: [24, 48])
            survival_model_type: model type for survival ('cox' or 'gb')

        returns:
            dict with both prediction types
        """
        results = {
            "timestamp": timestamp.isoformat(),
            "region_number": region_number,
            "classifications": {},
            "survival": None,
        }

        # classification predictions
        if self.classification_pipeline and self.classification_pipeline.models:
            for window in classification_windows:
                try:
                    if window in [24, 48]:
                        pred = self.predict_classification(
                            timestamp=timestamp,
                            window=window,
                            region_number=region_number,
                        )
                        results["classifications"][f"{window}h"] = pred
                except Exception as e:
                    logger.warning(f"classification prediction failed for {window}h: {e}")
                    results["classifications"][f"{window}h"] = {"error": str(e)}

        # survival prediction
        if self.survival_pipeline and self.survival_pipeline.is_fitted:
            try:
                pred = self.predict_survival(
                    timestamp=timestamp,
                    model_type=survival_model_type,
                    region_number=region_number,
                )
                results["survival"] = pred
            except Exception as e:
                logger.warning(f"survival prediction failed: {e}")
                results["survival"] = {"error": str(e)}

        return results

    def health_check(self) -> Dict[str, Any]:
        """
        check service health and model availability.

        returns:
            dict with service status
        """
        import shutil
        from src.data.database import get_database
        from src.data.schema import DataIngestionLog, PredictionLog, SystemValidationLog
        from src.config import PROJECT_ROOT
        from sqlalchemy import func, text

        status = {
            "status": "healthy",
            "classification_available": False,
            "survival_available": False,
            "drift_detection_enabled": self.drift_detector is not None,
            "outcome_logging_enabled": self.outcome_logger is not None,
        }

        # check model availability
        if self.classification_pipeline and self.classification_pipeline.models:
            status["classification_available"] = True

        if self.survival_pipeline and self.survival_pipeline.is_fitted:
            status["survival_available"] = True

        if not status["classification_available"] and not status["survival_available"]:
            status["status"] = "degraded"

        # check database connection
        try:
            db = get_database()
            with db.get_session() as session:
                session.execute(text("SELECT 1"))
                status["database_connected"] = True

                # get last ingestion timestamp
                last_ingestion = (
                    session.query(DataIngestionLog)
                    .filter(DataIngestionLog.status == "success")
                    .order_by(DataIngestionLog.run_timestamp.desc())
                    .first()
                )

                if last_ingestion:
                    status["last_ingestion"] = last_ingestion.run_timestamp.isoformat()
                else:
                    status["last_ingestion"] = None

                # get prediction count
                pred_count = session.query(func.count(PredictionLog.id)).scalar()
                status["total_predictions_logged"] = pred_count

                latest_validation = (
                    session.query(SystemValidationLog).order_by(SystemValidationLog.run_timestamp.desc()).first()
                )

                if latest_validation:
                    status["validation"] = {
                        "run_timestamp": latest_validation.run_timestamp.isoformat(),
                        "status": latest_validation.status,
                        "guardrail_triggered": latest_validation.guardrail_triggered,
                        "guardrail_reason": latest_validation.guardrail_reason,
                    }
                else:
                    status["validation"] = None

        except Exception as e:
            logger.error(f"database health check failed: {e}")
            status["database_connected"] = False
            status["last_ingestion"] = None
            status["status"] = "degraded"
            status["validation"] = None

        # check disk space
        try:
            disk_stats = shutil.disk_usage(PROJECT_ROOT)
            free_gb = disk_stats.free / (1024**3)
            total_gb = disk_stats.total / (1024**3)

            status["disk_space"] = {
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "percent_free": round((disk_stats.free / disk_stats.total) * 100, 1),
            }

            if free_gb < 1.0:
                status["status"] = "degraded"
                status["warnings"] = status.get("warnings", [])
                status["warnings"].append("low disk space")

        except Exception as e:
            logger.error(f"disk space check failed: {e}")
            status["disk_space"] = None

        validation_info = status.get("validation")
        if validation_info:
            guardrail_active = validation_info.get("guardrail_triggered", False)
            status["survival_guardrail"] = guardrail_active
            status["confidence_level"] = "low" if guardrail_active else "normal"
            if guardrail_active and status.get("status") == "healthy":
                status["status"] = "degraded"
            if guardrail_active:
                status.setdefault("warnings", [])
                reason = validation_info.get("guardrail_reason") or "survival validation guardrail triggered"
                status["warnings"].append(reason)
        else:
            status["survival_guardrail"] = False
            status["confidence_level"] = "unknown"

        # add performance metrics if outcome logging enabled
        if self.outcome_logger:
            try:
                status["performance_metrics"] = self.outcome_logger.get_performance_metrics()
            except Exception as e:
                logger.error(f"performance metrics retrieval failed: {e}")

        return status
