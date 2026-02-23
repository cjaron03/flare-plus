# fmt: off
"""model training with cross-validation and class balancing."""

import logging
from typing import Dict, Any, Optional, List, Tuple
import joblib

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
    # configure tqdm for docker/non-interactive terminals
    TQDM_KWARGS = {
        "file": sys.stderr,  # use stderr to avoid buffering
        "mininterval": 1.0,  # update at least every second
        "miniters": 1,  # update after each iteration
        "disable": False,  # explicitly enable
    }
except ImportError:
    HAS_TQDM = False
    TQDM_KWARGS = {}

    def tqdm(iterable, *args, **kwargs):
        return iterable

logger = logging.getLogger(__name__)


class ModelTrainer:
    """trains baseline models with cross-validation and class balancing."""

    def __init__(
        self,
        use_smote: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
        use_feature_selection: bool = False,
        feature_selection_method: str = "mutual_info",
    ):
        """
        initialize model trainer.

        args:
            use_smote: whether to use smote for oversampling
            cv_folds: number of cross-validation folds
            random_state: random seed
            use_feature_selection: whether to apply feature selection before training
            feature_selection_method: feature selection method ('mutual_info')
        """
        self.use_smote = use_smote
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.use_feature_selection = use_feature_selection
        self.feature_selection_method = feature_selection_method
        self.label_encoders: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.feature_names = None
        self.none_class_id: Optional[int] = None

    def prepare_features_and_labels(
        self,
        features_df: pd.DataFrame,
        label_column: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        prepare features and labels for training.

        args:
            features_df: dataframe with features and labels
            label_column: name of label column (e.g., 'label_24h')

        returns:
            tuple of (X, y, feature_names)
        """
        # get feature columns (exclude timestamp and label columns)
        exclude_cols = ["timestamp", "region_number"] + [
            col
            for col in features_df.columns
            if col.startswith("label_") or col.startswith("num_flares_")
        ]

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        self.feature_names = feature_cols  # type: ignore[assignment]

        # extract features
        X = features_df[feature_cols].values

        # extract labels
        if label_column not in features_df.columns:
            raise ValueError(f"label column '{label_column}' not found in dataframe")

        y = features_df[label_column].values

        # encode labels
        if label_column not in self.label_encoders:
            self.label_encoders[label_column] = LabelEncoder()
            y = self.label_encoders[label_column].fit_transform(y)
        else:
            y = self.label_encoders[label_column].transform(y)

        classes = self.label_encoders[label_column].classes_.tolist()
        if "None" in classes:
            self.none_class_id = int(self.label_encoders[label_column].transform(["None"])[0])
        else:
            self.none_class_id = None

        # handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, feature_cols

    @staticmethod
    def _metric_or_neg_inf(value: Any) -> float:
        """Safely convert possibly missing metric to a sortable float."""
        if value is None:
            return float("-inf")
        try:
            metric = float(value)
        except (TypeError, ValueError):
            return float("-inf")
        if np.isnan(metric):
            return float("-inf")
        return metric

    @staticmethod
    def _safe_nanmean(values: np.ndarray) -> float:
        """Compute nanmean without raising warnings when all values are NaN."""
        if len(values) == 0:
            return np.nan
        if np.isnan(values).all():
            return np.nan
        return float(np.nanmean(values))

    @classmethod
    def _model_selection_sort_key(cls, info: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        """
        Build a deterministic ranking key for event-focused model selection.

        Priority:
        1. event F1
        2. event recall
        3. calibrated brier (lower is better)
        4. weighted F1
        5. accuracy
        """
        event_f1 = cls._metric_or_neg_inf(info.get("test_event_f1"))
        if event_f1 == float("-inf"):
            event_f1 = cls._metric_or_neg_inf(info.get("cv_event_f1_mean"))

        event_recall = cls._metric_or_neg_inf(info.get("test_event_recall"))
        if event_recall == float("-inf"):
            event_recall = cls._metric_or_neg_inf(info.get("cv_event_recall_mean"))

        brier = info.get("test_brier_macro_calibrated")
        if brier is None:
            brier = info.get("test_brier_macro")
        brier_metric = cls._metric_or_neg_inf(brier)
        brier_key = -brier_metric if brier_metric != float("-inf") else float("-inf")

        weighted_f1 = cls._metric_or_neg_inf(info.get("test_weighted_f1"))
        if weighted_f1 == float("-inf"):
            weighted_f1 = cls._metric_or_neg_inf(info.get("cv_f1_mean"))

        accuracy = cls._metric_or_neg_inf(info.get("test_accuracy"))
        if accuracy == float("-inf"):
            accuracy = cls._metric_or_neg_inf(info.get("cv_mean"))

        return (event_f1, event_recall, brier_key, weighted_f1, accuracy)

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.1,
        min_features: int = 5,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        select features using mutual information.

        args:
            X: feature matrix
            y: labels
            feature_names: list of feature names
            threshold: fraction of max score; features above threshold*max are kept
            min_features: minimum number of features to keep

        returns:
            tuple of (filtered X, filtered feature_names)
        """
        scores = mutual_info_classif(X, y, random_state=self.random_state)
        max_score = scores.max() if scores.max() > 0 else 1.0
        cutoff = threshold * max_score

        selected_mask = scores >= cutoff
        # ensure we keep at least min_features
        if selected_mask.sum() < min_features:
            top_indices = np.argsort(scores)[::-1][:min_features]
            selected_mask = np.zeros(len(scores), dtype=bool)
            selected_mask[top_indices] = True

        selected_names = [name for name, keep in zip(feature_names, selected_mask) if keep]
        X_selected = X[:, selected_mask]

        logger.info(
            f"feature selection: kept {len(selected_names)}/{len(feature_names)} features"
        )

        return X_selected, selected_names

    @staticmethod
    def select_best_model(
        trained_models: Dict[str, Tuple[Any, Dict[str, Any]]],
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        select the best model across all types using event-focused ranking.

        args:
            trained_models: dict mapping model names to (model, training_info) tuples

        returns:
            (model, training_info) tuple for the best model, or None if no models
        """
        # skip alias keys
        skip_keys = {"logistic_regression", "best"}
        candidates = {k: v for k, v in trained_models.items() if k not in skip_keys}
        if not candidates:
            return None

        best_key = max(
            candidates,
            key=lambda k: ModelTrainer._model_selection_sort_key(candidates[k][1]),
        )
        best_model, best_info = candidates[best_key]
        best_info = dict(best_info)  # copy to avoid mutating original
        best_info["selected_from"] = best_key
        return best_model, best_info

    def _compute_class_weight(self, y: np.ndarray) -> Optional[Dict[int, float]]:
        """compute balanced class weights as a dict."""
        classes = np.unique(y)
        if len(classes) == 0:
            return None
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    def _build_logistic_pipeline(
        self,
        c: float,
        class_weight: Optional[Dict[int, float]] = None,
    ) -> SkPipeline:
        """build a scaled logistic regression pipeline."""
        return SkPipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        C=c,
                        random_state=self.random_state,
                        class_weight=class_weight,
                        solver="lbfgs",
                        verbose=0,
                    ),
                ),
            ]
        )

    def _build_gradient_boosting_model(self, params: Dict[str, Any]) -> GradientBoostingClassifier:
        """build a gradient boosting classifier from params."""
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            subsample=params["subsample"],
            random_state=self.random_state,
            verbose=0,
        )

    def _gradient_boosting_candidates(self) -> List[Dict[str, Any]]:
        """
        return candidate hyperparameter combinations for gradient boosting.

        This is a balanced shortlist that still spans low/high estimator counts,
        conservative/aggressive learning rates, and regularization settings.
        """
        estimator_lr_pairs = [
            (150, 0.06),
            (150, 0.1),
            (300, 0.03),
            (300, 0.06),
            (500, 0.03),
            (500, 0.06),
        ]
        depth_leaf_map = {
            2: [3, 8],
            3: [8],
        }

        candidates: List[Dict[str, Any]] = []
        for n_estimators, learning_rate in estimator_lr_pairs:
            for max_depth, min_samples_leaf_values in depth_leaf_map.items():
                for min_samples_leaf in min_samples_leaf_values:
                    for subsample in [0.8, 1.0]:
                        candidates.append(
                            {
                                "n_estimators": n_estimators,
                                "learning_rate": learning_rate,
                                "max_depth": max_depth,
                                "min_samples_leaf": min_samples_leaf,
                                "subsample": subsample,
                            }
                        )

        return candidates

    def _build_lightgbm_model(
        self,
        params: Dict[str, Any],
        class_weight: Optional[Dict[int, float]] = None,
    ) -> lgb.LGBMClassifier:
        """build a LightGBM classifier from params."""
        return lgb.LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            class_weight=class_weight,
            random_state=self.random_state,
            verbose=-1,
        )

    def _lightgbm_candidates(self) -> List[Dict[str, Any]]:
        """return candidate hyperparameter combinations for LightGBM."""
        estimator_lr_pairs = [
            (150, 0.06),
            (150, 0.1),
            (300, 0.03),
            (300, 0.06),
            (500, 0.03),
            (500, 0.06),
        ]
        depth_child_map = {
            3: [5, 20],
            5: [10, 20],
        }

        candidates: List[Dict[str, Any]] = []
        for n_estimators, learning_rate in estimator_lr_pairs:
            for max_depth, child_values in depth_child_map.items():
                for min_child_samples in child_values:
                    for subsample in [0.8, 1.0]:
                        for colsample_bytree in [0.8, 1.0]:
                            candidates.append(
                                {
                                    "n_estimators": n_estimators,
                                    "learning_rate": learning_rate,
                                    "max_depth": max_depth,
                                    "min_child_samples": min_child_samples,
                                    "subsample": subsample,
                                    "colsample_bytree": colsample_bytree,
                                }
                            )

        return candidates

    def _build_random_forest_model(
        self,
        params: Dict[str, Any],
        class_weight: Optional[Dict[int, float]] = None,
    ) -> RandomForestClassifier:
        """build a random forest classifier from params."""
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=class_weight,
            random_state=self.random_state,
            verbose=0,
        )

    def _random_forest_candidates(self) -> List[Dict[str, Any]]:
        """return candidate hyperparameter combinations for random forest."""
        candidates: List[Dict[str, Any]] = []
        for n_estimators in [100, 300, 500]:
            for max_depth in [5, 10, None]:
                for min_samples_leaf in [1, 5, 10, 20]:
                    for max_features in ["sqrt", "log2", 0.5]:
                        candidates.append(
                            {
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "min_samples_leaf": min_samples_leaf,
                                "max_features": max_features,
                            }
                        )

        return candidates

    def _cross_validate_estimator(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """run time-series CV and return arrays for accuracy/F1/event-F1/event-recall."""
        cv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_accuracy_scores: List[float] = []
        cv_f1_scores: List[float] = []
        cv_event_f1_scores: List[float] = []
        cv_event_recall_scores: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
            # warn when a fold is missing classes (e.g., rare flare types)
            train_classes_array = np.unique(y[train_idx])
            if len(train_classes_array) < 2:
                logger.warning(
                    f"[{fold_idx}/{self.cv_folds}] skipping cv fold: "
                    f"training fold has only one class {train_classes_array.tolist()}"
                )
                continue

            train_classes = set(train_classes_array)
            val_classes = set(np.unique(y[val_idx]))
            missing = val_classes - train_classes
            if missing:
                logger.warning(
                    f"[{fold_idx}/{self.cv_folds}] validation fold contains classes "
                    f"missing from training fold: {missing}"
                )
            logger.info(f"[{fold_idx}/{self.cv_folds}] training cv fold...")
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            model_cv = clone(estimator)
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight[train_idx]
            model_cv.fit(X_train_cv, y_train_cv, **fit_kwargs)

            y_pred_cv = model_cv.predict(X_val_cv)
            cv_accuracy_scores.append(float(accuracy_score(y_val_cv, y_pred_cv)))
            cv_f1_scores.append(float(f1_score(y_val_cv, y_pred_cv, average="weighted", zero_division=0)))
            if self.none_class_id is not None:
                y_val_event = (y_val_cv != self.none_class_id).astype(int)
                y_pred_event = (y_pred_cv != self.none_class_id).astype(int)
                cv_event_f1_scores.append(float(f1_score(y_val_event, y_pred_event, zero_division=0)))
                cv_event_recall_scores.append(
                    float(recall_score(y_val_event, y_pred_event, zero_division=0))
                )
            logger.info(
                f"[{fold_idx}/{self.cv_folds}] fold accuracy: {cv_accuracy_scores[-1]:.4f}, "
                f"weighted f1: {cv_f1_scores[-1]:.4f}"
            )

        if not cv_accuracy_scores:
            raise ValueError(
                "no valid time-series CV folds available: each training fold had fewer than 2 classes"
            )

        if self.none_class_id is None:
            cv_event_f1_scores = [np.nan] * len(cv_accuracy_scores)
            cv_event_recall_scores = [np.nan] * len(cv_accuracy_scores)

        return (
            np.array(cv_accuracy_scores),
            np.array(cv_f1_scores),
            np.array(cv_event_f1_scores, dtype=float),
            np.array(cv_event_recall_scores, dtype=float),
        )

    @staticmethod
    def _is_better_candidate(
        candidate_accuracy: np.ndarray,
        candidate_f1: np.ndarray,
        candidate_event_f1: np.ndarray,
        candidate_event_recall: np.ndarray,
        best_accuracy: float,
        best_f1: float,
        best_event_f1: float,
        best_event_recall: float,
        tolerance: float = 1e-6,
    ) -> bool:
        """prefer higher event F1/recall; fall back to accuracy and weighted F1."""
        if len(candidate_accuracy) == 0:
            return False

        candidate_accuracy_mean = float(candidate_accuracy.mean())
        candidate_f1_mean = float(candidate_f1.mean()) if len(candidate_f1) else float("-inf")
        candidate_event_f1_mean = ModelTrainer._safe_nanmean(candidate_event_f1)
        candidate_event_recall_mean = ModelTrainer._safe_nanmean(candidate_event_recall)
        has_event_metric = not np.isnan(candidate_event_f1_mean)
        best_has_event_metric = not np.isnan(best_event_f1)

        if has_event_metric:
            if (not best_has_event_metric) or candidate_event_f1_mean > best_event_f1 + tolerance:
                return True
            if abs(candidate_event_f1_mean - best_event_f1) <= tolerance:
                if np.isnan(best_event_recall) or candidate_event_recall_mean > best_event_recall + tolerance:
                    return True

        if candidate_accuracy_mean > best_accuracy + tolerance:
            return True

        if abs(candidate_accuracy_mean - best_accuracy) <= tolerance:
            return candidate_f1_mean > best_f1 + tolerance

        return False

    def train_logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_class_weight: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        train logistic regression model.

        args:
            X: feature matrix
            y: labels
            use_class_weight: whether to use class weights for balancing

        returns:
            tuple of (trained model, training info)
        """
        balanced_class_weight = self._compute_class_weight(y) if use_class_weight else None
        candidate_cs = [0.1, 0.5, 1.0, 2.0, 5.0]
        weighting_options: List[Tuple[str, Optional[Dict[int, float]]]] = [("none", None)]
        if balanced_class_weight is not None:
            weighting_options.append(("balanced", balanced_class_weight))

        best_cv_accuracy = -np.inf
        best_cv_f1 = -np.inf
        best_cv_event_f1 = np.nan
        best_cv_event_recall = np.nan
        best_c = 1.0
        best_weighting = "none"
        selected_class_weight: Optional[Dict[int, float]] = None
        best_accuracy_scores = np.array([])
        best_f1_scores = np.array([])
        best_event_f1_scores = np.array([])
        best_event_recall_scores = np.array([])

        for weighting_name, class_weight in weighting_options:
            for c in candidate_cs:
                logger.info(f"evaluating logistic regression candidate: class_weight={weighting_name}, C={c}")
                candidate_model = self._build_logistic_pipeline(c=c, class_weight=class_weight)
                cv_accuracy, cv_f1, cv_event_f1, cv_event_recall = self._cross_validate_estimator(
                    candidate_model, X, y
                )
                logger.info(
                    "logistic candidate class_weight=%s C=%s: cv accuracy=%.4f, cv weighted f1=%.4f, cv event f1=%.4f",
                    weighting_name,
                    c,
                    cv_accuracy.mean(),
                    cv_f1.mean(),
                    self._safe_nanmean(cv_event_f1),
                )

                if self._is_better_candidate(
                    cv_accuracy,
                    cv_f1,
                    cv_event_f1,
                    cv_event_recall,
                    best_accuracy=best_cv_accuracy,
                    best_f1=best_cv_f1,
                    best_event_f1=best_cv_event_f1,
                    best_event_recall=best_cv_event_recall,
                ):
                    best_cv_accuracy = float(cv_accuracy.mean())
                    best_cv_f1 = float(cv_f1.mean())
                    best_cv_event_f1 = self._safe_nanmean(cv_event_f1)
                    best_cv_event_recall = self._safe_nanmean(cv_event_recall)
                    best_c = c
                    best_weighting = weighting_name
                    selected_class_weight = class_weight
                    best_accuracy_scores = cv_accuracy
                    best_f1_scores = cv_f1
                    best_event_f1_scores = cv_event_f1
                    best_event_recall_scores = cv_event_recall

                if best_cv_accuracy >= 0.999:
                    logger.info("early-stopping logistic search after reaching near-perfect cv accuracy")
                    break

            if best_cv_accuracy >= 0.999:
                break

        model = self._build_logistic_pipeline(c=best_c, class_weight=selected_class_weight)
        logger.info(
            "fitting best logistic model on full data (class_weight=%s, C=%s)...",
            best_weighting,
            best_c,
        )
        model.fit(X, y)

        training_info = {
            "model_type": "logistic_regression",
            "cv_mean": float(best_accuracy_scores.mean()),
            "cv_std": float(best_accuracy_scores.std()),
            "cv_scores": best_accuracy_scores.tolist(),
            "cv_f1_mean": float(best_f1_scores.mean()) if len(best_f1_scores) else None,
            "cv_event_f1_mean": self._safe_nanmean(best_event_f1_scores) if len(best_event_f1_scores) else None,
            "cv_event_recall_mean": (
                self._safe_nanmean(best_event_recall_scores) if len(best_event_recall_scores) else None
            ),
            "selected_c": best_c,
            "selected_weighting": best_weighting,
            "class_weight": selected_class_weight,
        }

        return model, training_info

    def train_gradient_boosting(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_class_weight: bool = True,
    ) -> Tuple[GradientBoostingClassifier, Dict[str, Any]]:
        """
        train gradient boosting model.

        args:
            X: feature matrix
            y: labels
            use_class_weight: whether to use class weights for balancing

        returns:
            tuple of (trained model, training info)
        """
        balanced_class_weight = self._compute_class_weight(y) if use_class_weight else None
        balanced_sample_weight = None
        if balanced_class_weight:
            balanced_sample_weight = np.array([balanced_class_weight[int(label)] for label in y], dtype=float)
        weighting_options: List[Tuple[str, Optional[np.ndarray], Optional[Dict[int, float]]]] = [("none", None, None)]
        if balanced_sample_weight is not None:
            weighting_options.append(("balanced", balanced_sample_weight, balanced_class_weight))

        best_params: Optional[Dict[str, Any]] = None
        best_cv_accuracy = -np.inf
        best_cv_f1 = -np.inf
        best_cv_event_f1 = np.nan
        best_cv_event_recall = np.nan
        best_weighting = "none"
        selected_sample_weight: Optional[np.ndarray] = None
        selected_class_weight: Optional[Dict[int, float]] = None
        best_accuracy_scores = np.array([])
        best_f1_scores = np.array([])
        best_event_f1_scores = np.array([])
        best_event_recall_scores = np.array([])

        for weighting_name, sample_weight, class_weight in weighting_options:
            for params in self._gradient_boosting_candidates():
                logger.info(
                    "evaluating gradient boosting candidate: class_weight=%s, params=%s",
                    weighting_name,
                    params,
                )
                candidate_model = self._build_gradient_boosting_model(params)
                cv_accuracy, cv_f1, cv_event_f1, cv_event_recall = self._cross_validate_estimator(
                    candidate_model,
                    X,
                    y,
                    sample_weight=sample_weight,
                )
                logger.info(
                    "gradient candidate class_weight=%s params=%s: "
                    "cv accuracy=%.4f, cv weighted f1=%.4f, cv event f1=%.4f",
                    weighting_name,
                    params,
                    cv_accuracy.mean(),
                    cv_f1.mean(),
                    self._safe_nanmean(cv_event_f1),
                )

                if self._is_better_candidate(
                    cv_accuracy,
                    cv_f1,
                    cv_event_f1,
                    cv_event_recall,
                    best_accuracy=best_cv_accuracy,
                    best_f1=best_cv_f1,
                    best_event_f1=best_cv_event_f1,
                    best_event_recall=best_cv_event_recall,
                ):
                    best_cv_accuracy = float(cv_accuracy.mean())
                    best_cv_f1 = float(cv_f1.mean())
                    best_cv_event_f1 = self._safe_nanmean(cv_event_f1)
                    best_cv_event_recall = self._safe_nanmean(cv_event_recall)
                    best_weighting = weighting_name
                    best_params = params
                    selected_sample_weight = sample_weight
                    selected_class_weight = class_weight
                    best_accuracy_scores = cv_accuracy
                    best_f1_scores = cv_f1
                    best_event_f1_scores = cv_event_f1
                    best_event_recall_scores = cv_event_recall

                if best_cv_accuracy >= 0.999:
                    logger.info("early-stopping gradient boosting search after reaching near-perfect cv accuracy")
                    break

            if best_cv_accuracy >= 0.999:
                break

        if best_params is None:
            raise ValueError("failed to select gradient boosting hyperparameters")

        model = self._build_gradient_boosting_model(best_params)
        logger.info(
            "fitting best gradient boosting model on full data with class_weight=%s params=%s",
            best_weighting,
            best_params,
        )
        fit_kwargs = {}
        if selected_sample_weight is not None:
            fit_kwargs["sample_weight"] = selected_sample_weight
        model.fit(X, y, **fit_kwargs)

        training_info = {
            "model_type": "gradient_boosting",
            "cv_mean": float(best_accuracy_scores.mean()),
            "cv_std": float(best_accuracy_scores.std()),
            "cv_scores": best_accuracy_scores.tolist(),
            "cv_f1_mean": float(best_f1_scores.mean()) if len(best_f1_scores) else None,
            "cv_event_f1_mean": self._safe_nanmean(best_event_f1_scores) if len(best_event_f1_scores) else None,
            "cv_event_recall_mean": (
                self._safe_nanmean(best_event_recall_scores) if len(best_event_recall_scores) else None
            ),
            "selected_params": best_params,
            "selected_weighting": best_weighting,
            "class_weight": selected_class_weight,
        }

        return model, training_info

    def train_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_class_weight: bool = True,
    ) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
        """
        train LightGBM model.

        args:
            X: feature matrix
            y: labels
            use_class_weight: whether to use class weights for balancing

        returns:
            tuple of (trained model, training info)
        """
        balanced_class_weight = self._compute_class_weight(y) if use_class_weight else None
        weighting_options: List[Tuple[str, Optional[Dict[int, float]]]] = [("none", None)]
        if balanced_class_weight is not None:
            weighting_options.append(("balanced", balanced_class_weight))

        best_params: Optional[Dict[str, Any]] = None
        best_cv_accuracy = -np.inf
        best_cv_f1 = -np.inf
        best_cv_event_f1 = np.nan
        best_cv_event_recall = np.nan
        best_weighting = "none"
        selected_class_weight: Optional[Dict[int, float]] = None
        best_accuracy_scores = np.array([])
        best_f1_scores = np.array([])
        best_event_f1_scores = np.array([])
        best_event_recall_scores = np.array([])

        for weighting_name, class_weight in weighting_options:
            for params in self._lightgbm_candidates():
                logger.info(
                    "evaluating lightgbm candidate: class_weight=%s, params=%s",
                    weighting_name,
                    params,
                )
                candidate_model = self._build_lightgbm_model(params, class_weight=class_weight)
                cv_accuracy, cv_f1, cv_event_f1, cv_event_recall = self._cross_validate_estimator(
                    candidate_model, X, y
                )
                logger.info(
                    "lightgbm candidate class_weight=%s params=%s: "
                    "cv accuracy=%.4f, cv weighted f1=%.4f, cv event f1=%.4f",
                    weighting_name,
                    params,
                    cv_accuracy.mean(),
                    cv_f1.mean(),
                    self._safe_nanmean(cv_event_f1),
                )

                if self._is_better_candidate(
                    cv_accuracy,
                    cv_f1,
                    cv_event_f1,
                    cv_event_recall,
                    best_accuracy=best_cv_accuracy,
                    best_f1=best_cv_f1,
                    best_event_f1=best_cv_event_f1,
                    best_event_recall=best_cv_event_recall,
                ):
                    best_cv_accuracy = float(cv_accuracy.mean())
                    best_cv_f1 = float(cv_f1.mean())
                    best_cv_event_f1 = self._safe_nanmean(cv_event_f1)
                    best_cv_event_recall = self._safe_nanmean(cv_event_recall)
                    best_weighting = weighting_name
                    best_params = params
                    selected_class_weight = class_weight
                    best_accuracy_scores = cv_accuracy
                    best_f1_scores = cv_f1
                    best_event_f1_scores = cv_event_f1
                    best_event_recall_scores = cv_event_recall

                if best_cv_accuracy >= 0.999:
                    logger.info("early-stopping lightgbm search after reaching near-perfect cv accuracy")
                    break

            if best_cv_accuracy >= 0.999:
                break

        if best_params is None:
            raise ValueError("failed to select lightgbm hyperparameters")

        model = self._build_lightgbm_model(best_params, class_weight=selected_class_weight)
        logger.info(
            "fitting best lightgbm model on full data with class_weight=%s params=%s",
            best_weighting,
            best_params,
        )
        model.fit(X, y)

        training_info = {
            "model_type": "lightgbm",
            "cv_mean": float(best_accuracy_scores.mean()),
            "cv_std": float(best_accuracy_scores.std()),
            "cv_scores": best_accuracy_scores.tolist(),
            "cv_f1_mean": float(best_f1_scores.mean()) if len(best_f1_scores) else None,
            "cv_event_f1_mean": self._safe_nanmean(best_event_f1_scores) if len(best_event_f1_scores) else None,
            "cv_event_recall_mean": (
                self._safe_nanmean(best_event_recall_scores) if len(best_event_recall_scores) else None
            ),
            "selected_params": best_params,
            "selected_weighting": best_weighting,
            "class_weight": selected_class_weight,
        }

        return model, training_info

    def train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_class_weight: bool = True,
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        train random forest model.

        args:
            X: feature matrix
            y: labels
            use_class_weight: whether to use class weights for balancing

        returns:
            tuple of (trained model, training info)
        """
        balanced_class_weight = self._compute_class_weight(y) if use_class_weight else None
        weighting_options: List[Tuple[str, Optional[Dict[int, float]]]] = [("none", None)]
        if balanced_class_weight is not None:
            weighting_options.append(("balanced", balanced_class_weight))

        best_params: Optional[Dict[str, Any]] = None
        best_cv_accuracy = -np.inf
        best_cv_f1 = -np.inf
        best_cv_event_f1 = np.nan
        best_cv_event_recall = np.nan
        best_weighting = "none"
        selected_class_weight: Optional[Dict[int, float]] = None
        best_accuracy_scores = np.array([])
        best_f1_scores = np.array([])
        best_event_f1_scores = np.array([])
        best_event_recall_scores = np.array([])

        for weighting_name, class_weight in weighting_options:
            for params in self._random_forest_candidates():
                logger.info(
                    "evaluating random forest candidate: class_weight=%s, params=%s",
                    weighting_name,
                    params,
                )
                candidate_model = self._build_random_forest_model(params, class_weight=class_weight)
                cv_accuracy, cv_f1, cv_event_f1, cv_event_recall = self._cross_validate_estimator(
                    candidate_model, X, y
                )
                logger.info(
                    "random forest candidate class_weight=%s params=%s: "
                    "cv accuracy=%.4f, cv weighted f1=%.4f, cv event f1=%.4f",
                    weighting_name,
                    params,
                    cv_accuracy.mean(),
                    cv_f1.mean(),
                    self._safe_nanmean(cv_event_f1),
                )

                if self._is_better_candidate(
                    cv_accuracy,
                    cv_f1,
                    cv_event_f1,
                    cv_event_recall,
                    best_accuracy=best_cv_accuracy,
                    best_f1=best_cv_f1,
                    best_event_f1=best_cv_event_f1,
                    best_event_recall=best_cv_event_recall,
                ):
                    best_cv_accuracy = float(cv_accuracy.mean())
                    best_cv_f1 = float(cv_f1.mean())
                    best_cv_event_f1 = self._safe_nanmean(cv_event_f1)
                    best_cv_event_recall = self._safe_nanmean(cv_event_recall)
                    best_weighting = weighting_name
                    best_params = params
                    selected_class_weight = class_weight
                    best_accuracy_scores = cv_accuracy
                    best_f1_scores = cv_f1
                    best_event_f1_scores = cv_event_f1
                    best_event_recall_scores = cv_event_recall

                if best_cv_accuracy >= 0.999:
                    logger.info("early-stopping random forest search after reaching near-perfect cv accuracy")
                    break

            if best_cv_accuracy >= 0.999:
                break

        if best_params is None:
            raise ValueError("failed to select random forest hyperparameters")

        model = self._build_random_forest_model(best_params, class_weight=selected_class_weight)
        logger.info(
            "fitting best random forest model on full data with class_weight=%s params=%s",
            best_weighting,
            best_params,
        )
        model.fit(X, y)

        training_info = {
            "model_type": "random_forest",
            "cv_mean": float(best_accuracy_scores.mean()),
            "cv_std": float(best_accuracy_scores.std()),
            "cv_scores": best_accuracy_scores.tolist(),
            "cv_f1_mean": float(best_f1_scores.mean()) if len(best_f1_scores) else None,
            "cv_event_f1_mean": self._safe_nanmean(best_event_f1_scores) if len(best_event_f1_scores) else None,
            "cv_event_recall_mean": (
                self._safe_nanmean(best_event_recall_scores) if len(best_event_recall_scores) else None
            ),
            "selected_params": best_params,
            "selected_weighting": best_weighting,
            "class_weight": selected_class_weight,
        }

        return model, training_info

    def train_with_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "logistic",
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        train model with smote oversampling.

        args:
            X: feature matrix
            y: labels
            model_type: 'logistic', 'gradient_boosting', 'lightgbm', or 'random_forest'

        returns:
            tuple of (trained model, training info)
        """
        if model_type == "logistic":
            base_model = self._build_logistic_pipeline(c=1.0, class_weight=None)
            candidate_pipelines = [
                (
                    c,
                    ImbPipeline(
                        [
                            ("smote", SMOTE(random_state=self.random_state)),
                            ("model", self._build_logistic_pipeline(c=c, class_weight=None)),
                        ]
                    ),
                )
                for c in [0.1, 0.5, 1.0, 2.0, 5.0]
            ]
        elif model_type == "gradient_boosting":
            base_model = self._build_gradient_boosting_model(
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "min_samples_leaf": 5,
                    "subsample": 0.8,
                }
            )
            candidate_pipelines = [
                (
                    params,
                    ImbPipeline(
                        [
                            ("smote", SMOTE(random_state=self.random_state)),
                            ("model", self._build_gradient_boosting_model(params)),
                        ]
                    ),
                )
                for params in self._gradient_boosting_candidates()
            ]
        elif model_type == "lightgbm":
            base_model = self._build_lightgbm_model(
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "min_child_samples": 10,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }
            )
            candidate_pipelines = [
                (
                    params,
                    ImbPipeline(
                        [
                            ("smote", SMOTE(random_state=self.random_state)),
                            ("model", self._build_lightgbm_model(params)),
                        ]
                    ),
                )
                for params in self._lightgbm_candidates()
            ]
        elif model_type == "random_forest":
            base_model = self._build_random_forest_model(
                {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_leaf": 5,
                    "max_features": "sqrt",
                }
            )
            candidate_pipelines = [
                (
                    params,
                    ImbPipeline(
                        [
                            ("smote", SMOTE(random_state=self.random_state)),
                            ("model", self._build_random_forest_model(params)),
                        ]
                    ),
                )
                for params in self._random_forest_candidates()
            ]
        else:
            raise ValueError(f"unknown model type: {model_type}")

        best_key = None
        best_pipeline = None
        best_accuracy_scores = np.array([])
        best_f1_scores = np.array([])
        best_event_f1_scores = np.array([])
        best_event_recall_scores = np.array([])
        best_cv_accuracy = -np.inf
        best_cv_f1 = -np.inf
        best_cv_event_f1 = np.nan
        best_cv_event_recall = np.nan

        for candidate_key, pipeline in candidate_pipelines:
            logger.info(f"evaluating {model_type} + smote candidate: {candidate_key}")
            cv_accuracy, cv_f1, cv_event_f1, cv_event_recall = self._cross_validate_estimator(
                pipeline, X, y
            )
            if self._is_better_candidate(
                cv_accuracy,
                cv_f1,
                cv_event_f1,
                cv_event_recall,
                best_accuracy=best_cv_accuracy,
                best_f1=best_cv_f1,
                best_event_f1=best_cv_event_f1,
                best_event_recall=best_cv_event_recall,
            ):
                best_cv_accuracy = float(cv_accuracy.mean())
                best_cv_f1 = float(cv_f1.mean())
                best_cv_event_f1 = self._safe_nanmean(cv_event_f1)
                best_cv_event_recall = self._safe_nanmean(cv_event_recall)
                best_key = candidate_key
                best_pipeline = pipeline
                best_accuracy_scores = cv_accuracy
                best_f1_scores = cv_f1
                best_event_f1_scores = cv_event_f1
                best_event_recall_scores = cv_event_recall
            if best_cv_accuracy >= 0.999:
                logger.info(f"early-stopping {model_type} + smote search after near-perfect cv accuracy")
                break

        if best_pipeline is None:
            # fallback to default model if candidate search fails unexpectedly
            best_pipeline = ImbPipeline(
                [("smote", SMOTE(random_state=self.random_state)), ("model", base_model)]
            )

        logger.info(f"fitting best {model_type} + smote model on full training data...")
        best_pipeline.fit(X, y)

        training_info = {
            "model_type": f"{model_type}_with_smote",
            "cv_mean": float(best_accuracy_scores.mean()) if len(best_accuracy_scores) else None,
            "cv_std": float(best_accuracy_scores.std()) if len(best_accuracy_scores) else None,
            "cv_scores": best_accuracy_scores.tolist(),
            "cv_f1_mean": float(best_f1_scores.mean()) if len(best_f1_scores) else None,
            "cv_event_f1_mean": self._safe_nanmean(best_event_f1_scores) if len(best_event_f1_scores) else None,
            "cv_event_recall_mean": (
                self._safe_nanmean(best_event_recall_scores) if len(best_event_recall_scores) else None
            ),
            "selected_candidate": best_key,
            "use_smote": True,
        }

        return best_pipeline, training_info

    def train_baseline_models(
        self,
        features_df: pd.DataFrame,
        label_column: str,
        models: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """
        train multiple baseline models.

        args:
            features_df: dataframe with features and labels
            label_column: name of label column (e.g., 'label_24h')
            models: list of model types to train ('logistic', 'gradient_boosting', or both)

        returns:
            dict mapping model names to (model, training_info) tuples
        """
        if models is None:
            models = ["logistic", "gradient_boosting"]

        # prepare data
        X, y, feature_names = self.prepare_features_and_labels(features_df, label_column)

        # NOTE: feature selection is intentionally NOT done here.
        # It must be called from the pipeline AFTER the chronological
        # train/test split to prevent data leakage into the test set.
        # See ClassificationPipeline.train_and_evaluate() for the correct call site.

        logger.info(
            f"training models on {len(X)} samples with {len(feature_names)} features"
        )
        logger.info(
            f"label distribution: {dict(zip(*np.unique(y, return_counts=True)))}"
        )

        trained_models = {}

        # train each model type
        for model_type in models:
            try:
                candidates: List[Tuple[Any, Dict[str, Any]]] = []

                if model_type == "logistic":
                    model, info = self.train_logistic_regression(X, y)
                    candidates.append((model, info))
                    if self.use_smote:
                        smote_model, smote_info = self.train_with_smote(X, y, model_type="logistic")
                        candidates.append((smote_model, smote_info))

                    model, info = max(
                        candidates,
                        key=lambda item: self._model_selection_sort_key(item[1]),
                    )
                    # use consistent key name for lookup
                    trained_models["logistic"] = (model, info)
                    trained_models["logistic_regression"] = (model, info)  # keep backward compatibility

                elif model_type == "gradient_boosting":
                    model, info = self.train_gradient_boosting(X, y)
                    candidates.append((model, info))
                    if self.use_smote:
                        smote_model, smote_info = self.train_with_smote(X, y, model_type="gradient_boosting")
                        candidates.append((smote_model, smote_info))

                    model, info = max(
                        candidates,
                        key=lambda item: self._model_selection_sort_key(item[1]),
                    )
                    trained_models["gradient_boosting"] = (model, info)

                elif model_type == "lightgbm":
                    model, info = self.train_lightgbm(X, y)
                    candidates.append((model, info))
                    if self.use_smote:
                        smote_model, smote_info = self.train_with_smote(X, y, model_type="lightgbm")
                        candidates.append((smote_model, smote_info))

                    model, info = max(
                        candidates,
                        key=lambda item: self._model_selection_sort_key(item[1]),
                    )
                    trained_models["lightgbm"] = (model, info)

                elif model_type == "random_forest":
                    model, info = self.train_random_forest(X, y)
                    candidates.append((model, info))
                    if self.use_smote:
                        smote_model, smote_info = self.train_with_smote(X, y, model_type="random_forest")
                        candidates.append((smote_model, smote_info))

                    model, info = max(
                        candidates,
                        key=lambda item: self._model_selection_sort_key(item[1]),
                    )
                    trained_models["random_forest"] = (model, info)

                else:
                    logger.warning(f"unknown model type: {model_type}, skipping")
                    continue

                logger.info(
                    f"{model_type}: cv accuracy = {info['cv_mean']:.4f} (+/- {info['cv_std']*2:.4f})"
                )

            except Exception as e:
                logger.error(f"error training {model_type}: {e}")
                continue

        # store selected feature names in each training_info for downstream alignment
        for key in trained_models:
            trained_models[key][1]["feature_names"] = list(feature_names)

        # auto-select best model across all types
        best = self.select_best_model(trained_models)
        if best is not None:
            trained_models["best"] = best
            logger.info(
                f"auto-selected best model: {best[1].get('selected_from')} "
                f"(event_f1={best[1].get('test_event_f1', best[1].get('cv_event_f1_mean'))}, "
                f"accuracy={best[1].get('test_accuracy', best[1].get('cv_mean', 0))})"
            )

        self.models[label_column] = trained_models
        return trained_models

    def save_model(self, model: Any, filepath: str):
        """save trained model to file using joblib (safer than pickle)."""
        joblib.dump(model, filepath)

    def load_model(self, filepath: str) -> Any:
        """load trained model from file using joblib."""
        return joblib.load(filepath)


def train_baseline_models(
    features_df: pd.DataFrame,
    label_column: str,
    models: Optional[List[str]] = None,
    use_smote: bool = True,
    cv_folds: int = 5,
) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    convenience function to train baseline models.

    args:
        features_df: dataframe with features and labels
        label_column: name of label column (e.g., 'label_24h')
        models: list of model types to train
        use_smote: whether to use smote for oversampling
        cv_folds: number of cross-validation folds

    returns:
        dict mapping model names to (model, training_info) tuples
    """
    trainer = ModelTrainer(use_smote=use_smote, cv_folds=cv_folds)
    return trainer.train_baseline_models(features_df, label_column, models)
# fmt: on
