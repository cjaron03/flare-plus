# fmt: off
"""main pipeline for short-term classification."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.config import CONFIG
from src.features.pipeline import FeatureEngineer
from src.models.labeling import FlareLabeler
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

# model config
MODEL_CONFIG = CONFIG.get("model", {})
TARGET_WINDOWS = MODEL_CONFIG.get("target_windows", [24, 48])


class ClassificationPipeline:
    """end-to-end pipeline for short-term flare classification."""

    def __init__(
        self,
        use_smote: bool = True,
        cv_folds: int = 5,
        calibrate: bool = True,
        random_state: int = 42,
    ):
        """
        initialize classification pipeline.

        args:
            use_smote: whether to use smote for oversampling
            cv_folds: number of cross-validation folds
            calibrate: whether to calibrate probabilities
            random_state: random seed
        """
        self.feature_engineer = FeatureEngineer()
        self.labeler = FlareLabeler()
        self.trainer = ModelTrainer(
            use_smote=use_smote, cv_folds=cv_folds, random_state=random_state
        )
        self.evaluator = ModelEvaluator()
        self.use_smote = use_smote
        self.cv_folds = cv_folds
        self.calibrate = calibrate
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Any] = {}

    def prepare_dataset(
        self,
        start_date: datetime,
        end_date: datetime,
        sample_interval_hours: int = 1,
        region_number: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        prepare dataset with features and labels.

        args:
            start_date: start date for dataset
            end_date: end date for dataset
            sample_interval_hours: hours between samples
            region_number: optional region number to filter by

        returns:
            dataframe with features and labels
        """
        logger.info(f"preparing dataset from {start_date} to {end_date}")

        # generate timestamps
        timestamps = []
        current = start_date
        while current <= end_date:
            timestamps.append(current)
            current += timedelta(hours=sample_interval_hours)

        logger.info(f"generating features for {len(timestamps)} timestamps")

        # compute features
        features_df = self.feature_engineer.compute_features_batch(
            timestamps,
            region_number=region_number,
            normalize=False,
            standardize=False,
            handle_missing=True,
        )

        if len(features_df) == 0:
            logger.warning("no features generated")
            return pd.DataFrame()

        logger.info(f"created features for {len(features_df)} timestamps")

        # create labels
        logger.info("creating labels")
        labeled_df = self.labeler.create_labels_from_features(
            features_df, windows=TARGET_WINDOWS
        )

        # filter out rows without labels
        for window in TARGET_WINDOWS:
            label_col = f"label_{window}h"
            if label_col in labeled_df.columns:
                initial_count = len(labeled_df)
                labeled_df = labeled_df[labeled_df[label_col].notna()]
                filtered_count = len(labeled_df)
                logger.info(
                    f"filtered {initial_count - filtered_count} rows without {label_col} labels"
                )

        logger.info(f"final dataset size: {len(labeled_df)} samples")

        return labeled_df

    def train_and_evaluate(
        self,
        dataset: pd.DataFrame,
        test_size: float = 0.2,
        models: Optional[List[str]] = None,
        plot_reliability: bool = False,
        reliability_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        train and evaluate models on dataset.

        args:
            dataset: dataframe with features and labels
            test_size: fraction of data to use for testing
            models: list of model types to train
            plot_reliability: whether to plot reliability diagrams
            reliability_dir: directory to save reliability diagrams

        returns:
            dict with training and evaluation results
        """
        if models is None:
            models = ["logistic", "gradient_boosting"]

        results = {}

        for window in TARGET_WINDOWS:
            label_col = f"label_{window}h"
            if label_col not in dataset.columns:
                logger.warning(f"label column {label_col} not found, skipping")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"training models for {window}h prediction window")
            logger.info(f"{'='*60}")

            # split data
            from sklearn.model_selection import train_test_split

            # prepare features and labels
            exclude_cols = ["timestamp", "region_number"] + [
                col
                for col in dataset.columns
                if col.startswith("label_") or col.startswith("num_flares_")
            ]
            feature_cols = [col for col in dataset.columns if col not in exclude_cols]
            X = dataset[feature_cols].values
            y = dataset[label_col].values

            # encode labels
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            classes = label_encoder.classes_.tolist()

            # handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y_encoded,
            )

            logger.info(f"train size: {len(X_train)}, test size: {len(X_test)}")
            # fix: correctly map classes to counts
            unique_labels, counts = np.unique(y_train, return_counts=True)
            class_dist = dict(zip([classes[i] for i in unique_labels], counts))
            logger.info(f"class distribution (train): {class_dist}")

            # train models
            window_results = {}
            for model_type in models:
                logger.info(f"\ntraining {model_type} model...")

                try:
                    # prepare temporary dataframe for training
                    train_df = pd.DataFrame(X_train, columns=feature_cols)
                    train_df[label_col] = label_encoder.inverse_transform(y_train)

                    # train model
                    trained_models = self.trainer.train_baseline_models(
                        train_df, label_col, models=[model_type]
                    )

                    if model_type not in trained_models:
                        logger.warning(f"failed to train {model_type}")
                        continue

                    model, training_info = trained_models[model_type]

                    # evaluate on test set
                    logger.info(f"evaluating {model_type} model...")

                    # update label encoder for evaluator
                    self.evaluator.label_encoder = label_encoder

                    # determine reliability filepath
                    reliability_filepath = None
                    if plot_reliability and reliability_dir:
                        import os
                        os.makedirs(reliability_dir, exist_ok=True)
                        reliability_filepath = os.path.join(
                            reliability_dir, f"reliability_{window}h_{model_type}.png"
                        )

                    # fix: pass training data for calibration, test data for evaluation
                    evaluation_results = self.evaluator.evaluate_model(
                        model,
                        X_test,
                        y_test,
                        classes=classes,
                        calibrate=self.calibrate,
                        X_calibration=X_train,  # use training data for calibration
                        y_calibration=y_train,  # use training labels for calibration
                        plot_reliability=plot_reliability,
                        reliability_filepath=reliability_filepath,
                    )

                    window_results[model_type] = {
                        "model": model,
                        "training_info": training_info,
                        "evaluation_results": evaluation_results,
                        "label_encoder": label_encoder,
                        "feature_names": feature_cols,
                    }

                    # log key metrics
                    logger.info(f"\n{model_type} results:")
                    logger.info(
                        f"  cv accuracy: {training_info['cv_mean']:.4f} (+/- {training_info['cv_std']*2:.4f})"
                    )
                    logger.info(
                        f"  test accuracy: {evaluation_results['classification_report']['accuracy']:.4f}"
                    )
                    logger.info(
                        f"  macro avg brier score: {evaluation_results['brier_score']['macro_avg']:.4f}"
                    )
                    logger.info(
                        f"  macro avg roc-auc: {evaluation_results['roc_auc']['macro_avg']:.4f}"
                    )

                except Exception as e:
                    logger.error(f"error training/evaluating {model_type}: {e}")
                    continue

            results[f"{window}h"] = window_results

        self.models = results
        self.evaluation_results = results

        return results

    def predict(
        self,
        timestamp: datetime,
        window: int,
        model_type: str = "gradient_boosting",
        region_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        make prediction for a given timestamp.

        args:
            timestamp: timestamp to predict for
            window: prediction window in hours (24 or 48)
            model_type: model type to use ('logistic' or 'gradient_boosting')
            region_number: optional region number to filter by

        returns:
            dict with prediction results
        """
        window_key = f"{window}h"
        if window_key not in self.models:
            raise ValueError(f"no models trained for {window}h window")

        if model_type not in self.models[window_key]:
            raise ValueError(f"model type {model_type} not found for {window}h window")

        model_info = self.models[window_key][model_type]
        model = model_info["model"]
        label_encoder = model_info["label_encoder"]
        feature_names = model_info["feature_names"]

        # compute features
        features_df = self.feature_engineer.compute_features(
            timestamp,
            region_number=region_number,
            normalize=False,
            standardize=False,
            handle_missing=True,
        )

        if len(features_df) == 0:
            raise ValueError("could not compute features")

        # extract features in correct order
        X = features_df[feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # predict
        y_pred = model.predict(X)[0]
        y_prob = model.predict_proba(X)[0]

        # decode prediction
        predicted_class = label_encoder.inverse_transform([y_pred])[0]
        class_probs = dict(zip(label_encoder.classes_, y_prob))

        return {
            "timestamp": timestamp,
            "window_hours": window,
            "predicted_class": predicted_class,
            "class_probabilities": class_probs,
            "model_type": model_type,
        }

# fmt: on
