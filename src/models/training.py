"""model training with cross-validation and class balancing."""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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
        self.label_encoders = {}
        self.models = {}
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
            col for col in features_df.columns if col.startswith("label_") or col.startswith("num_flares_")
        ]

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        self.feature_names = feature_cols

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

    def train_logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_class_weight: bool = True,
    ) -> Tuple[LogisticRegression, Dict[str, Any]]:
        """
        train logistic regression model.

        args:
            X: feature matrix
            y: labels
            use_class_weight: whether to use class weights for balancing

        returns:
            tuple of (trained model, training info)
        """
        # compute class weights
        class_weight = None
        if use_class_weight:
            classes = np.unique(y)
            class_weights = compute_class_weight("balanced", classes=classes, y=y)
            class_weight = dict(zip(classes, class_weights))

        # train model
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight=class_weight,
            multi_class="multinomial",
            solver="lbfgs",
        )

        # cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        # fit on full data
        model.fit(X, y)

        training_info = {
            "model_type": "logistic_regression",
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
            "class_weight": class_weight,
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
        # compute class weights
        class_weight = None
        if use_class_weight:
            classes = np.unique(y)
            class_weights = compute_class_weight("balanced", classes=classes, y=y)
            class_weight = dict(zip(classes, class_weights))

        # initial model with default params
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=self.random_state,
            subsample=0.8,
        )

        # cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        # fit on full data
        model.fit(X, y)

        training_info = {
            "model_type": "gradient_boosting",
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
            "class_weight": class_weight,
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
        # create pipeline with smote
        if model_type == "logistic":
            base_model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                multi_class="multinomial",
                solver="lbfgs",
            )
        elif model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state,
                subsample=0.8,
            )
        else:
            raise ValueError(f"unknown model type: {model_type}")

        pipeline = ImbPipeline([("smote", SMOTE(random_state=self.random_state)), ("model", base_model)])

        # cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

        # fit on full data
        pipeline.fit(X, y)

        training_info = {
            "model_type": f"{model_type}_with_smote",
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
            "use_smote": True,
        }

        return pipeline, training_info

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

        logger.info(f"training models on {len(X)} samples with {len(feature_names)} features")
        logger.info(f"label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        trained_models = {}

        # train each model type
        for model_type in models:
            try:
                if model_type == "logistic":
                    if self.use_smote:
                        model, info = self.train_with_smote(X, y, model_type="logistic")
                    else:
                        model, info = self.train_logistic_regression(X, y)
                    trained_models["logistic_regression"] = (model, info)

                elif model_type == "gradient_boosting":
                    if self.use_smote:
                        model, info = self.train_with_smote(X, y, model_type="gradient_boosting")
                    else:
                        model, info = self.train_gradient_boosting(X, y)
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
        """save trained model to file."""
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

    def load_model(self, filepath: str) -> Any:
        """load trained model from file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


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

