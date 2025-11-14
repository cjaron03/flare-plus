"""monitoring hooks for input drift detection and outcome logging."""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd

from scipy import stats

from src.data.database import get_database
from src.data.schema import PredictionLog

logger = logging.getLogger(__name__)


class InputDriftDetector:
    """detect input feature drift using statistical tests."""

    def __init__(self, reference_window_days: int = 30, max_reference_samples: int = 1000):
        """
        initialize drift detector.

        args:
            reference_window_days: number of days to use for reference distribution
            max_reference_samples: maximum number of samples to keep in reference
        """
        self.reference_window_days = reference_window_days
        self.max_reference_samples = max_reference_samples
        self.reference_features: Dict[str, deque] = {}
        self.last_update: Optional[datetime] = None

    def update_reference(
        self,
        features: pd.DataFrame,
        timestamp: datetime,
    ):
        """
        update reference distribution with new features.

        args:
            features: dataframe with feature columns
            timestamp: observation timestamp
        """
        # deque with maxlen handles automatic removal of old samples

        for col in features.select_dtypes(include=[np.number]).columns:
            if col not in self.reference_features:
                self.reference_features[col] = deque(maxlen=self.max_reference_samples)

            # add new values
            for val in features[col].dropna().values:
                self.reference_features[col].append((timestamp, val))

            # remove old values (maintain maxlen via deque, but we could also filter by date)
            # deque with maxlen handles this automatically when appending

        self.last_update = timestamp

    def detect_drift(
        self,
        features: pd.DataFrame,
        timestamp: datetime,
        alpha: float = 0.05,
        method: str = "ks",
    ) -> Dict[str, Any]:
        """
        detect drift in feature distributions.

        args:
            features: current features to test
            timestamp: observation timestamp
            alpha: significance level for drift detection
            method: statistical test method ('ks' for kolmogorov-smirnov or 'mannwhitney')

        returns:
            dict with drift detection results
        """
        drift_results = {
            "timestamp": timestamp.isoformat(),
            "drifted_features": [],
            "all_tests": {},
            "overall_drift": False,
        }

        numeric_features = features.select_dtypes(include=[np.number]).columns

        for col in numeric_features:
            current_values = features[col].dropna().values

            if len(current_values) == 0:
                continue

            if col not in self.reference_features or len(self.reference_features[col]) == 0:
                # no reference data yet, skip
                continue

            # get reference values (just the values, ignore timestamps for now)
            reference_values = np.array([val for _, val in self.reference_features[col]])

            if len(reference_values) < 10:
                # not enough reference data
                continue

            try:
                if method == "ks":
                    # kolmogorov-smirnov test
                    statistic, p_value = stats.ks_2samp(reference_values, current_values)
                elif method == "mannwhitney":
                    # mann-whitney u test (non-parametric)
                    statistic, p_value = stats.mannwhitneyu(reference_values, current_values, alternative="two-sided")
                else:
                    raise ValueError(f"unknown method: {method}")

                drift_detected = bool(p_value < alpha)
                drift_results["all_tests"][col] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift_detected": drift_detected,
                }

                if drift_detected:
                    drift_results["drifted_features"].append(col)
                    drift_results["overall_drift"] = True

            except Exception as e:
                logger.warning(f"drift test failed for {col}: {e}")

        return drift_results


class OutcomeLogger:
    """log predictions and actual outcomes for monitoring."""

    def __init__(self, persist_to_db: bool = True):
        """
        initialize outcome logger.

        args:
            persist_to_db: if True, save predictions to database
        """
        self.predictions: List[Dict[str, Any]] = []
        self.max_predictions = 10000  # keep last 10k predictions in memory
        self.persist_to_db = persist_to_db

    def log_prediction(
        self,
        prediction: Dict[str, Any],
        prediction_type: str,
        timestamp: datetime,
        region_number: Optional[int] = None,
    ):
        """
        log a prediction for future outcome tracking.

        args:
            prediction: prediction dict
            prediction_type: 'classification' or 'survival'
            timestamp: observation timestamp
            region_number: optional region number
        """
        # log to in-memory list
        log_entry = {
            "prediction_id": len(self.predictions),
            "timestamp": timestamp.isoformat(),
            "prediction_type": prediction_type,
            "region_number": region_number,
            "prediction": prediction,
            "actual_outcome": None,  # filled later when outcome is known
            "outcome_timestamp": None,
        }

        self.predictions.append(log_entry)

        # maintain max size
        if len(self.predictions) > self.max_predictions:
            self.predictions = self.predictions[-self.max_predictions :]  # noqa: E203

        # persist to database if enabled
        if self.persist_to_db:
            try:
                self._save_to_database(prediction, prediction_type, timestamp, region_number)
            except Exception as e:
                logger.error(f"failed to persist prediction to database: {e}", exc_info=True)

    def update_outcome(
        self,
        prediction_id: int,
        actual_outcome: Dict[str, Any],
        outcome_timestamp: datetime,
    ):
        """
        update prediction with actual outcome.

        args:
            prediction_id: id of prediction to update
            actual_outcome: actual outcome dict
            outcome_timestamp: when outcome was observed
        """
        if prediction_id < len(self.predictions):
            self.predictions[prediction_id]["actual_outcome"] = actual_outcome
            self.predictions[prediction_id]["outcome_timestamp"] = outcome_timestamp.isoformat()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        compute performance metrics from logged predictions with outcomes.

        returns:
            dict with performance metrics
        """
        metrics = {
            "total_predictions": len(self.predictions),
            "predictions_with_outcomes": 0,
            "classification_accuracy": None,
            "survival_c_index": None,
        }

        # filter predictions with outcomes
        completed = [p for p in self.predictions if p["actual_outcome"] is not None]
        metrics["predictions_with_outcomes"] = len(completed)

        if len(completed) == 0:
            return metrics

        # compute classification accuracy
        classification_completed = [p for p in completed if p["prediction_type"] == "classification"]
        if len(classification_completed) > 0:
            correct = 0
            total = 0

            for pred in classification_completed:
                predicted_class = pred["prediction"].get("predicted_class")
                actual_class = pred["actual_outcome"].get("flare_class", "None")

                if predicted_class == actual_class:
                    correct += 1
                total += 1

            if total > 0:
                metrics["classification_accuracy"] = correct / total

        # compute survival c-index (simplified - would need proper survival analysis)
        survival_completed = [p for p in completed if p["prediction_type"] == "survival"]
        if len(survival_completed) > 0:
            # placeholder for c-index computation
            # full implementation would require proper survival analysis metrics
            metrics["survival_predictions_count"] = len(survival_completed)

        return metrics

    def _save_to_database(
        self,
        prediction: Dict[str, Any],
        prediction_type: str,
        timestamp: datetime,
        region_number: Optional[int] = None,
    ):
        """
        save prediction to database.

        args:
            prediction: prediction dict
            prediction_type: 'classification' or 'survival'
            timestamp: observation timestamp
            region_number: optional region number
        """
        db = get_database()

        # extract relevant fields from prediction dict
        model_type = prediction.get("model_type")
        window_hours = prediction.get("window")
        predicted_class = prediction.get("predicted_class")
        class_probs = prediction.get("probabilities") or prediction.get("class_probabilities")
        prob_dist = prediction.get("probability_distribution")
        hazard_score = prediction.get("hazard_score")

        # convert dicts to json strings for storage
        class_probs_json = json.dumps(class_probs) if class_probs else None
        prob_dist_json = json.dumps(prob_dist) if prob_dist else None

        with db.get_session() as session:
            pred_log = PredictionLog(
                prediction_timestamp=datetime.utcnow(),
                observation_timestamp=timestamp,
                prediction_type=prediction_type,
                region_number=region_number,
                model_type=model_type,
                window_hours=window_hours,
                predicted_class=predicted_class,
                class_probabilities=class_probs_json,
                probability_distribution=prob_dist_json,
                hazard_score=hazard_score,
            )
            session.add(pred_log)
            session.commit()
            logger.debug(f"saved prediction to database: id={pred_log.id}")

    def get_predictions_from_db(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        prediction_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        retrieve predictions from database.

        args:
            start_date: filter predictions after this date
            end_date: filter predictions before this date
            prediction_type: filter by prediction type ('classification' or 'survival')
            limit: maximum number of predictions to return

        returns:
            list of prediction dicts
        """
        db = get_database()
        predictions = []

        with db.get_session() as session:
            query = session.query(PredictionLog)

            if start_date:
                query = query.filter(PredictionLog.prediction_timestamp >= start_date)
            if end_date:
                query = query.filter(PredictionLog.prediction_timestamp <= end_date)
            if prediction_type:
                query = query.filter(PredictionLog.prediction_type == prediction_type)

            query = query.order_by(PredictionLog.prediction_timestamp.desc()).limit(limit)

            for pred_log in query.all():
                pred_dict = {
                    "id": pred_log.id,
                    "prediction_timestamp": pred_log.prediction_timestamp.isoformat(),
                    "observation_timestamp": pred_log.observation_timestamp.isoformat(),
                    "prediction_type": pred_log.prediction_type,
                    "region_number": pred_log.region_number,
                    "model_type": pred_log.model_type,
                    "window_hours": pred_log.window_hours,
                    "predicted_class": pred_log.predicted_class,
                    "class_probabilities": (
                        json.loads(pred_log.class_probabilities) if pred_log.class_probabilities else None
                    ),
                    "probability_distribution": (
                        json.loads(pred_log.probability_distribution) if pred_log.probability_distribution else None
                    ),
                    "hazard_score": pred_log.hazard_score,
                    "actual_flare_class": pred_log.actual_flare_class,
                    "actual_flare_time": pred_log.actual_flare_time.isoformat() if pred_log.actual_flare_time else None,
                }
                predictions.append(pred_dict)

        return predictions
