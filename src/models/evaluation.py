"""model evaluation with calibration, brier score, roc-auc, and reliability diagrams."""

import logging
from typing import Dict, Any, Optional, List, Tuple
import warnings

import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)
if not HAS_MATPLOTLIB:
    logger.warning("matplotlib not available, reliability diagrams will be skipped")


class ModelEvaluator:
    """evaluates models with calibration and comprehensive metrics."""

    def __init__(self, label_encoder: Any = None):
        """
        initialize model evaluator.

        args:
            label_encoder: label encoder used during training
        """
        self.label_encoder = label_encoder

    def calibrate_probabilities(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "isotonic",
        cv: int = 3,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        calibrate model probabilities.

        args:
            model: trained model
            X: feature matrix
            y: true labels
            method: calibration method ('isotonic' or 'sigmoid')
            cv: number of cross-validation folds for calibration

        returns:
            tuple of (calibrated model, calibration info)
        """
        calibrated_model = CalibratedClassifierCV(model, method=method, cv=cv)
        calibrated_model.fit(X, y)

        # get uncalibrated and calibrated probabilities
        uncalibrated_probs = model.predict_proba(X)
        calibrated_probs = calibrated_model.predict_proba(X)

        calibration_info = {
            "method": method,
            "cv_folds": cv,
            "uncalibrated_probs_mean": uncalibrated_probs.mean(axis=0).tolist(),
            "calibrated_probs_mean": calibrated_probs.mean(axis=0).tolist(),
        }

        return calibrated_model, calibration_info

    def compute_brier_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        compute brier score for each class.

        args:
            y_true: true labels
            y_prob: predicted probabilities (shape: [n_samples, n_classes])
            classes: list of class names

        returns:
            dict mapping class names to brier scores
        """
        if classes is None:
            classes = [f"class_{i}" for i in range(y_prob.shape[1])]

        # convert to one-hot encoding
        y_true_binarized = label_binarize(y_true, classes=range(len(classes)))

        brier_scores = {}
        for i, class_name in enumerate(classes):
            score = brier_score_loss(y_true_binarized[:, i], y_prob[:, i])
            brier_scores[class_name] = score

        # overall brier score (macro average)
        brier_scores["macro_avg"] = np.mean(list(brier_scores.values()))

        return brier_scores

    def compute_roc_auc_per_class(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        compute roc-auc score for each class (one-vs-rest).

        args:
            y_true: true labels
            y_prob: predicted probabilities (shape: [n_samples, n_classes])
            classes: list of class names

        returns:
            dict mapping class names to roc-auc scores
        """
        if classes is None:
            classes = [f"class_{i}" for i in range(y_prob.shape[1])]

        # convert to one-hot encoding
        y_true_binarized = label_binarize(y_true, classes=range(len(classes)))

        roc_auc_scores = {}
        for i, class_name in enumerate(classes):
            try:
                score = roc_auc_score(y_true_binarized[:, i], y_prob[:, i])
                roc_auc_scores[class_name] = score
            except ValueError as e:
                # class may not be present in y_true
                logger.warning(f"could not compute roc-auc for {class_name}: {e}")
                roc_auc_scores[class_name] = np.nan

        # macro average
        valid_scores = [s for s in roc_auc_scores.values() if not np.isnan(s)]
        if len(valid_scores) > 0:
            roc_auc_scores["macro_avg"] = np.mean(valid_scores)
        else:
            roc_auc_scores["macro_avg"] = np.nan

        return roc_auc_scores

    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        classes: Optional[List[str]] = None,
        n_bins: int = 10,
        filepath: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        plot reliability diagram (calibration curve) for each class.

        args:
            y_true: true labels
            y_prob: predicted probabilities (shape: [n_samples, n_classes])
            classes: list of class names
            n_bins: number of bins for calibration curve
            filepath: optional path to save figure

        returns:
            matplotlib figure or None if matplotlib not available
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available, skipping reliability diagram")
            return None

        if classes is None:
            classes = [f"class_{i}" for i in range(y_prob.shape[1])]

        # convert to one-hot encoding
        y_true_binarized = label_binarize(y_true, classes=range(len(classes)))

        n_classes = len(classes)
        n_cols = min(2, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_classes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, class_name in enumerate(classes):
            ax = axes[i]

            # compute calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binarized[:, i], y_prob[:, i], n_bins=n_bins, strategy="uniform"
            )

            # plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="calibration curve")
            ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")

            ax.set_xlabel("mean predicted probability")
            ax.set_ylabel("fraction of positives")
            ax.set_title(f"reliability diagram: {class_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # hide unused subplots
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"reliability diagram saved to {filepath}")

        return fig

    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        classes: Optional[List[str]] = None,
        calibrate: bool = True,
        plot_reliability: bool = False,
        reliability_filepath: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        comprehensive model evaluation.

        args:
            model: trained model
            X: feature matrix
            y_true: true labels
            classes: list of class names
            calibrate: whether to calibrate probabilities
            plot_reliability: whether to plot reliability diagram
            reliability_filepath: path to save reliability diagram

        returns:
            dict with evaluation metrics
        """
        # get predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)

        # get class names
        if classes is None:
            if hasattr(model, "classes_"):
                if self.label_encoder is not None:
                    classes = self.label_encoder.inverse_transform(model.classes_).tolist()
                else:
                    classes = model.classes_.tolist()
            else:
                classes = [f"class_{i}" for i in range(y_prob.shape[1])]

        evaluation_results = {
            "predictions": y_pred.tolist(),
            "probabilities": y_prob.tolist(),
            "classes": classes,
        }

        # classification report
        evaluation_results["classification_report"] = classification_report(
            y_true, y_pred, target_names=classes, output_dict=True
        )

        # confusion matrix
        evaluation_results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        # brier score
        brier_scores = self.compute_brier_score(y_true, y_prob, classes)
        evaluation_results["brier_score"] = brier_scores

        # roc-auc per class
        roc_auc_scores = self.compute_roc_auc_per_class(y_true, y_prob, classes)
        evaluation_results["roc_auc"] = roc_auc_scores

        # calibrate if requested
        if calibrate:
            calibrated_model, calibration_info = self.calibrate_probabilities(model, X, y_true)
            evaluation_results["calibration"] = calibration_info

            # recompute metrics with calibrated probabilities
            calibrated_probs = calibrated_model.predict_proba(X)
            calibrated_brier = self.compute_brier_score(y_true, calibrated_probs, classes)
            calibrated_roc_auc = self.compute_roc_auc_per_class(y_true, calibrated_probs, classes)

            evaluation_results["calibrated_brier_score"] = calibrated_brier
            evaluation_results["calibrated_roc_auc"] = calibrated_roc_auc

            # plot reliability diagram with calibrated probabilities
            if plot_reliability:
                self.plot_reliability_diagram(
                    y_true, calibrated_probs, classes, filepath=reliability_filepath
                )

        else:
            # plot reliability diagram with uncalibrated probabilities
            if plot_reliability:
                self.plot_reliability_diagram(y_true, y_prob, classes, filepath=reliability_filepath)

        return evaluation_results


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    classes: Optional[List[str]] = None,
    calibrate: bool = True,
) -> Dict[str, Any]:
    """
    convenience function to evaluate a model.

    args:
        model: trained model
        X: feature matrix
        y_true: true labels
        classes: list of class names
        calibrate: whether to calibrate probabilities

    returns:
        dict with evaluation metrics
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(model, X, y_true, classes, calibrate, plot_reliability=False)


def calibrate_probabilities(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    method: str = "isotonic",
) -> Any:
    """
    convenience function to calibrate probabilities.

    args:
        model: trained model
        X: feature matrix
        y: true labels
        method: calibration method ('isotonic' or 'sigmoid')

    returns:
        calibrated model
    """
    evaluator = ModelEvaluator()
    calibrated_model, _ = evaluator.calibrate_probabilities(model, X, y, method)
    return calibrated_model

