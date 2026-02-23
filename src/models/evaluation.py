# fmt: off
"""model evaluation with calibration, brier score, roc-auc, and reliability diagrams."""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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

        note: calibration should use training data only to avoid data leakage.
        the CalibratedClassifierCV uses internal cross-validation, so passing
        training data is correct - it will split internally for calibration.

        args:
            model: trained model
            X: feature matrix (should be training data, not test data)
            y: true labels (should be training labels, not test labels)
            method: calibration method ('isotonic' or 'sigmoid')
            cv: number of cross-validation folds for calibration

        returns:
            tuple of (calibrated model, calibration info)
        """
        resolved_cv = self._resolve_calibration_cv(y, requested_cv=cv)
        if resolved_cv is None:
            raise ValueError("insufficient class support for calibration cv")

        # CalibratedClassifierCV uses internal CV, so using training data is correct.
        calibrated_model = CalibratedClassifierCV(model, method=method, cv=resolved_cv)
        calibrated_model.fit(X, y)

        # evaluate on same training data for info (calibration already used internal CV)
        uncalibrated_probs = model.predict_proba(X)
        calibrated_probs = calibrated_model.predict_proba(X)

        calibration_info = {
            "method": method,
            "cv_folds": resolved_cv,
            "uncalibrated_probs_mean": uncalibrated_probs.mean(axis=0).tolist(),
            "calibrated_probs_mean": calibrated_probs.mean(axis=0).tolist(),
        }

        return calibrated_model, calibration_info

    @staticmethod
    def _resolve_calibration_cv(
        y: np.ndarray,
        requested_cv: int,
    ) -> Optional[int]:
        """Resolve a safe CV fold count for calibration or return None if impossible."""
        if requested_cv < 2:
            return None

        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            return None

        min_class_count = int(class_counts.min())
        resolved_cv = min(requested_cv, min_class_count)
        if resolved_cv < 2:
            return None
        return resolved_cv

    def select_best_calibration(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        methods: Optional[List[str]] = None,
        cv: int = 3,
        X_eval: Optional[np.ndarray] = None,
        y_eval: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Try multiple calibration methods and select the one with lowest macro brier score.

        Calibration is fitted on (X, y). If X_eval/y_eval are provided, Brier score
        comparison uses held-out data to avoid in-sample bias. Otherwise falls back
        to evaluating on the calibration data (less reliable).

        Returns:
            tuple of (best calibrated model, calibration metadata)
        """
        if methods is None:
            methods = ["sigmoid", "isotonic"]

        if len(y) == 0:
            return None, None

        # use held-out data for evaluation if available, otherwise fall back to training data
        X_score = X_eval if X_eval is not None else X
        y_score = y_eval if y_eval is not None else y

        base_probs = model.predict_proba(X_score)
        baseline_brier = float(self.compute_brier_score(y_score, base_probs)["macro_avg"])

        best_model = None
        best_info = None
        best_brier = np.inf

        for method in methods:
            try:
                calibrated_model, info = self.calibrate_probabilities(
                    model=model,
                    X=X,
                    y=y,
                    method=method,
                    cv=cv,
                )
                calibrated_probs = calibrated_model.predict_proba(X_score)
                calibrated_brier = float(self.compute_brier_score(y_score, calibrated_probs)["macro_avg"])
                info["calibration_brier_macro"] = calibrated_brier
                info["uncalibrated_brier_macro"] = baseline_brier

                if calibrated_brier < best_brier - 1e-9:
                    best_brier = calibrated_brier
                    best_model = calibrated_model
                    best_info = info
            except Exception as exc:
                logger.warning("calibration method %s failed: %s", method, exc)
                continue

        if best_model is None or best_info is None:
            return None, None

        best_info = dict(best_info)
        best_info["selected"] = True
        best_info["improved_brier"] = bool(
            best_info.get("calibration_brier_macro", np.inf)
            <= best_info.get("uncalibrated_brier_macro", np.inf) + 1e-9
        )
        return best_model, best_info

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
                y_true_binarized[:, i],
                y_prob[:, i],
                n_bins=n_bins,
                strategy="uniform",
            )

            # plot calibration curve
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                "s-",
                label="calibration curve",
            )
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
        X_calibration: Optional[np.ndarray] = None,
        y_calibration: Optional[np.ndarray] = None,
        plot_reliability: bool = False,
        reliability_filepath: Optional[str] = None,
        return_calibrated_model: bool = False,
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
                    classes = self.label_encoder.inverse_transform(
                        model.classes_
                    ).tolist()
                else:
                    classes = model.classes_.tolist()
            else:
                classes = [f"class_{i}" for i in range(y_prob.shape[1])]

        # store predictions (needed for confusion matrix, etc.)
        # but store only summary statistics for probabilities to save memory
        prob_summary = {
            "mean": y_prob.mean(axis=0).tolist(),
            "std": y_prob.std(axis=0).tolist(),
            "min": y_prob.min(axis=0).tolist(),
            "max": y_prob.max(axis=0).tolist(),
            "shape": list(y_prob.shape),
        }

        evaluation_results = {
            "predictions": y_pred.tolist(),
            "probabilities_summary": prob_summary,  # summary stats instead of full array
            "classes": classes,
        }

        all_label_ids = list(range(len(classes)))

        # classification report (keep a stable class set even if some classes are absent in y_true)
        evaluation_results["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=all_label_ids,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )

        # confusion matrix with stable class ordering
        evaluation_results["confusion_matrix"] = confusion_matrix(
            y_true,
            y_pred,
            labels=all_label_ids,
        ).tolist()

        # brier score
        brier_scores = self.compute_brier_score(y_true, y_prob, classes)
        evaluation_results["brier_score"] = brier_scores

        # roc-auc per class
        roc_auc_scores = self.compute_roc_auc_per_class(y_true, y_prob, classes)
        evaluation_results["roc_auc"] = roc_auc_scores

        # event/no-event binary metrics (event := any class other than "None")
        evaluation_results["event_metrics"] = self.compute_event_metrics(y_true, y_pred, classes)

        # calibrate if requested
        # use separate calibration set (training data) to avoid data leakage
        if calibrate:
            calibrated_model = None
            if X_calibration is not None and y_calibration is not None:
                calibrated_model, calibration_info = self.select_best_calibration(
                    model, X_calibration, y_calibration,
                    X_eval=X, y_eval=y_true,  # evaluate on held-out test data
                )
            else:
                logger.warning(
                    "calibration requested but no calibration data provided. "
                    "skipping calibration to avoid data leakage."
                )
                calibration_info = None

            if calibrated_model is not None:
                evaluation_results["calibration"] = calibration_info
                evaluation_results["selected_calibration_method"] = calibration_info.get("method")

                # recompute metrics with calibrated probabilities on test data
                calibrated_probs = calibrated_model.predict_proba(X)
                calibrated_pred = np.argmax(calibrated_probs, axis=1)
                calibrated_brier = self.compute_brier_score(
                    y_true, calibrated_probs, classes
                )
                calibrated_roc_auc = self.compute_roc_auc_per_class(
                    y_true, calibrated_probs, classes
                )

                evaluation_results["calibrated_brier_score"] = calibrated_brier
                evaluation_results["calibrated_roc_auc"] = calibrated_roc_auc
                evaluation_results["calibrated_event_metrics"] = self.compute_event_metrics(
                    y_true,
                    calibrated_pred,
                    classes,
                )

                if return_calibrated_model:
                    evaluation_results["calibrated_model"] = calibrated_model

                # plot reliability diagram with calibrated probabilities
                if plot_reliability:
                    self.plot_reliability_diagram(
                        y_true, calibrated_probs, classes, filepath=reliability_filepath
                    )

        else:
            # plot reliability diagram with uncalibrated probabilities
            if plot_reliability:
                self.plot_reliability_diagram(
                    y_true, y_prob, classes, filepath=reliability_filepath
                )

        return evaluation_results

    @staticmethod
    def compute_event_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute binary event/no-event metrics.

        Event is defined as any class except "None".
        """
        if classes is None:
            return {"available": False, "reason": "classes_not_provided"}

        none_idx = next((idx for idx, cls in enumerate(classes) if str(cls).lower() == "none"), None)
        if none_idx is None:
            return {"available": False, "reason": "none_class_not_found"}

        y_true_event = (y_true != none_idx).astype(int)
        y_pred_event = (y_pred != none_idx).astype(int)

        positives = int(y_true_event.sum())
        n_samples = int(len(y_true_event))
        positive_rate = float(positives / n_samples) if n_samples else 0.0

        return {
            "available": True,
            "n_samples": n_samples,
            "positives": positives,
            "positive_rate": positive_rate,
            "accuracy": float(accuracy_score(y_true_event, y_pred_event)),
            "precision": float(precision_score(y_true_event, y_pred_event, zero_division=0)),
            "recall": float(recall_score(y_true_event, y_pred_event, zero_division=0)),
            "f1": float(f1_score(y_true_event, y_pred_event, zero_division=0)),
        }


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
    return evaluator.evaluate_model(
        model, X, y_true, classes, calibrate, plot_reliability=False
    )
# fmt: on


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
