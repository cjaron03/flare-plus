# fmt: off
"""model training with cross-validation and class balancing."""

import logging
from typing import Dict, Any, Optional, List, Tuple
import joblib

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
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
    ):
        """
        initialize model trainer.

        args:
            use_smote: whether to use smote for oversampling
            cv_folds: number of cross-validation folds
            random_state: random seed
        """
        self.use_smote = use_smote
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.label_encoders: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.feature_names = None

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

        # handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, feature_cols

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

    def _cross_validate_estimator(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """run stratified CV and return accuracy + weighted F1 arrays."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_accuracy_scores: List[float] = []
        cv_f1_scores: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
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
            logger.info(
                f"[{fold_idx}/{self.cv_folds}] fold accuracy: {cv_accuracy_scores[-1]:.4f}, "
                f"weighted f1: {cv_f1_scores[-1]:.4f}"
            )

        return np.array(cv_accuracy_scores), np.array(cv_f1_scores)

    @staticmethod
    def _is_better_candidate(
        candidate_accuracy: np.ndarray,
        candidate_f1: np.ndarray,
        best_accuracy: float,
        best_f1: float,
        tolerance: float = 1e-6,
    ) -> bool:
        """prefer higher accuracy; break near-ties with weighted F1."""
        if len(candidate_accuracy) == 0:
            return False

        candidate_accuracy_mean = float(candidate_accuracy.mean())
        candidate_f1_mean = float(candidate_f1.mean()) if len(candidate_f1) else float("-inf")

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
        best_c = 1.0
        best_weighting = "none"
        selected_class_weight: Optional[Dict[int, float]] = None
        best_accuracy_scores = np.array([])
        best_f1_scores = np.array([])

        for weighting_name, class_weight in weighting_options:
            for c in candidate_cs:
                logger.info(f"evaluating logistic regression candidate: class_weight={weighting_name}, C={c}")
                candidate_model = self._build_logistic_pipeline(c=c, class_weight=class_weight)
                cv_accuracy, cv_f1 = self._cross_validate_estimator(candidate_model, X, y)
                logger.info(
                    "logistic candidate class_weight=%s C=%s: cv accuracy=%.4f, cv weighted f1=%.4f",
                    weighting_name,
                    c,
                    cv_accuracy.mean(),
                    cv_f1.mean(),
                )

                if self._is_better_candidate(
                    cv_accuracy,
                    cv_f1,
                    best_accuracy=best_cv_accuracy,
                    best_f1=best_cv_f1,
                ):
                    best_cv_accuracy = float(cv_accuracy.mean())
                    best_cv_f1 = float(cv_f1.mean())
                    best_c = c
                    best_weighting = weighting_name
                    selected_class_weight = class_weight
                    best_accuracy_scores = cv_accuracy
                    best_f1_scores = cv_f1

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
        best_weighting = "none"
        selected_sample_weight: Optional[np.ndarray] = None
        selected_class_weight: Optional[Dict[int, float]] = None
        best_accuracy_scores = np.array([])
        best_f1_scores = np.array([])

        for weighting_name, sample_weight, class_weight in weighting_options:
            for params in self._gradient_boosting_candidates():
                logger.info(
                    "evaluating gradient boosting candidate: class_weight=%s, params=%s",
                    weighting_name,
                    params,
                )
                candidate_model = self._build_gradient_boosting_model(params)
                cv_accuracy, cv_f1 = self._cross_validate_estimator(
                    candidate_model,
                    X,
                    y,
                    sample_weight=sample_weight,
                )
                logger.info(
                    "gradient candidate class_weight=%s params=%s: cv accuracy=%.4f, cv weighted f1=%.4f",
                    weighting_name,
                    params,
                    cv_accuracy.mean(),
                    cv_f1.mean(),
                )

                if self._is_better_candidate(
                    cv_accuracy,
                    cv_f1,
                    best_accuracy=best_cv_accuracy,
                    best_f1=best_cv_f1,
                ):
                    best_cv_accuracy = float(cv_accuracy.mean())
                    best_cv_f1 = float(cv_f1.mean())
                    best_weighting = weighting_name
                    best_params = params
                    selected_sample_weight = sample_weight
                    selected_class_weight = class_weight
                    best_accuracy_scores = cv_accuracy
                    best_f1_scores = cv_f1

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
            model_type: 'logistic' or 'gradient_boosting'

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
        else:
            raise ValueError(f"unknown model type: {model_type}")

        best_key = None
        best_pipeline = None
        best_accuracy_scores = np.array([])
        best_f1_scores = np.array([])
        best_cv_accuracy = -np.inf
        best_cv_f1 = -np.inf

        for candidate_key, pipeline in candidate_pipelines:
            logger.info(f"evaluating {model_type} + smote candidate: {candidate_key}")
            cv_accuracy, cv_f1 = self._cross_validate_estimator(pipeline, X, y)
            if self._is_better_candidate(
                cv_accuracy,
                cv_f1,
                best_accuracy=best_cv_accuracy,
                best_f1=best_cv_f1,
            ):
                best_cv_accuracy = float(cv_accuracy.mean())
                best_cv_f1 = float(cv_f1.mean())
                best_key = candidate_key
                best_pipeline = pipeline
                best_accuracy_scores = cv_accuracy
                best_f1_scores = cv_f1
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
                        key=lambda item: item[1].get("cv_mean", float("-inf")) or float("-inf"),
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
                        key=lambda item: item[1].get("cv_mean", float("-inf")) or float("-inf"),
                    )
                    trained_models["gradient_boosting"] = (model, info)

                else:
                    logger.warning(f"unknown model type: {model_type}, skipping")
                    continue

                logger.info(
                    f"{model_type}: cv accuracy = {info['cv_mean']:.4f} (+/- {info['cv_std']*2:.4f})"
                )

            except Exception as e:
                logger.error(f"error training {model_type}: {e}")
                continue

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
