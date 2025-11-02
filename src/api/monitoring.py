"""monitoring hooks for input drift detection and outcome logging."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd

from scipy import stats

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

                drift_results["all_tests"][col] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift_detected": p_value < alpha,
                }

                if p_value < alpha:
                    drift_results["drifted_features"].append(col)
                    drift_results["overall_drift"] = True

            except Exception as e:
                logger.warning(f"drift test failed for {col}: {e}")

        return drift_results


class OutcomeLogger:
    """log predictions and actual outcomes for monitoring."""

    def __init__(self):
        """initialize outcome logger."""
        self.predictions: List[Dict[str, Any]] = []
        self.max_predictions = 10000  # keep last 10k predictions

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
