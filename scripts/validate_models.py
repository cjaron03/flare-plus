#!/usr/bin/env python
"""validate trained models for deployment readiness."""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse  # noqa: E402
import joblib  # noqa: E402
import logging  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Dict, Any, Optional  # noqa: E402

import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> Any:
    """
    load model from file.

    args:
        model_path: path to model file

    returns:
        loaded model object
    """
    print(f"Loading model from {model_path}...")

    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    try:
        model = joblib.load(model_path)
        print("[OK] Model loaded successfully")
        return model
    except Exception as e:
        raise RuntimeError(f"failed to load model: {e}")


def check_model_metadata(model: Any) -> Dict[str, Any]:
    """
    extract and validate model metadata.

    args:
        model: loaded model object

    returns:
        dict of metadata
    """
    print("\nChecking model metadata...")
    metadata = {}

    # try to extract common metadata attributes
    if hasattr(model, "trained_at"):
        metadata["trained_at"] = model.trained_at
        print(f"  Training date: {model.trained_at}")

    if hasattr(model, "model_type"):
        metadata["model_type"] = model.model_type
        print(f"  Model type: {model.model_type}")

    if hasattr(model, "feature_names"):
        metadata["feature_names"] = model.feature_names
        print(f"  Features: {len(model.feature_names)} features")
    elif hasattr(model, "feature_names_in_"):
        metadata["feature_names"] = list(model.feature_names_in_)
        print(f"  Features: {len(model.feature_names_in_)} features")

    if hasattr(model, "c_index"):
        metadata["c_index"] = model.c_index
        print(f"  C-index: {model.c_index:.4f}")

    if hasattr(model, "test_score"):
        metadata["test_score"] = model.test_score
        print(f"  Test score: {model.test_score:.4f}")

    # check for survival pipeline attributes
    if hasattr(model, "is_fitted"):
        metadata["is_fitted"] = model.is_fitted
        print(f"  Is fitted: {model.is_fitted}")

    if not metadata:
        print("  [WARNING] No metadata found in model")

    return metadata


def validate_survival_model(model: Any, model_path: Path) -> bool:
    """
    validate survival model with statistical checks.

    args:
        model: loaded survival model
        model_path: path to model file

    returns:
        True if validation passes
    """
    print("\nValidating survival model...")
    passed = True

    # check for required methods
    required_methods = ["predict", "is_fitted"]
    for method in required_methods:
        if not hasattr(model, method):
            print(f"  [FAIL] Missing required method: {method}")
            passed = False
        else:
            print(f"  [OK] Has method: {method}")

    # check if model is fitted
    if hasattr(model, "is_fitted"):
        if not model.is_fitted:
            print("  [FAIL] Model is not fitted")
            passed = False
        else:
            print("  [OK] Model is fitted")

    # check C-index if available
    if hasattr(model, "c_index"):
        c_index = model.c_index
        if c_index <= 0.5:
            print(f"  [FAIL] C-index too low: {c_index:.4f} (should be > 0.5)")
            passed = False
        else:
            print(f"  [OK] C-index acceptable: {c_index:.4f}")

    # try to make a test prediction
    try:
        print("\n  Testing prediction capability...")
        timestamp = datetime.utcnow()

        # try to predict
        if hasattr(model, "predict"):
            result = model.predict(timestamp)

            # check for NaN values
            if "probability_distribution" in result:
                probs = result["probability_distribution"]
                if any(np.isnan(v) for v in probs.values()):
                    print("  [FAIL] Prediction contains NaN values")
                    passed = False
                else:
                    print("  [OK] No NaN values in predictions")

                # check if probabilities are within reasonable bounds
                total = sum(probs.values())
                if total < 0.01:
                    print(f"  [FAIL] Probabilities too low (sum={total:.6f})")
                    passed = False
                elif total > 1.5:
                    print(f"  [FAIL] Probabilities too high (sum={total:.6f})")
                    passed = False
                else:
                    print(f"  [OK] Probabilities within expected range (sum={total:.4f})")

                survival_info = result.get("survival_function", {})
                survival_probs = survival_info.get("probabilities") if isinstance(survival_info, dict) else None
                if survival_probs:
                    survival_range = max(survival_probs) - min(survival_probs)
                    if survival_range < 0.001:
                        print(f"  [FAIL] Survival function nearly flat (range={survival_range:.6f})")
                        passed = False
                    else:
                        min_nonterminal = min(survival_probs[:-1]) if len(survival_probs) > 1 else survival_probs[0]
                        max_drop = (
                            max((survival_probs[i] - survival_probs[i + 1]) for i in range(len(survival_probs) - 1))
                            if len(survival_probs) > 1
                            else 0.0
                        )
                        if min_nonterminal <= 0.01:
                            print("  [FAIL] Survival function collapses to zero inside prediction horizon")
                            passed = False
                        elif max_drop > 0.9:
                            print(f"  [FAIL] Survival function has abrupt drop of {max_drop:.3f}")
                            passed = False
                        else:
                            print(f"  [OK] Survival function shows variation (range={survival_range:.4f})")

            print("  [OK] Model can make predictions")

    except Exception as e:
        print(f"  [FAIL] Prediction test failed: {e}")
        passed = False

    return passed


def validate_classification_model(model: Any, model_path: Path) -> bool:
    """
    validate classification model with statistical checks.

    args:
        model: loaded classification model
        model_path: path to model file

    returns:
        True if validation passes
    """
    print("\nValidating classification model...")
    passed = True

    # check for required attributes
    if not hasattr(model, "models") or not model.models:
        print("  [FAIL] No trained models found in pipeline")
        return False

    print(f"  [OK] Found {len(model.models)} time windows")

    # check each time window's models
    for window, models_dict in model.models.items():
        print(f"\n  Checking {window}h window...")

        for model_type, model_info in models_dict.items():
            print(f"    Model: {model_type}")

            # check for required components
            if "model" not in model_info:
                print("      [FAIL] Missing model object")
                passed = False
                continue

            if "label_encoder" not in model_info:
                print("      [WARNING] Missing label encoder")

            # try to make a prediction
            try:
                timestamp = datetime.utcnow()
                result = model.predict(timestamp, window=window, model_type=model_type)

                # check for NaN values
                if "probabilities" in result:
                    probs = result["probabilities"]
                    if any(np.isnan(v) for v in probs.values()):
                        print("      [FAIL] Contains NaN probabilities")
                        passed = False
                    else:
                        print("      [OK] No NaN values")

                    # check if probabilities sum to approximately 1
                    total = sum(probs.values())
                    if abs(total - 1.0) > 0.1:
                        print(f"      [WARNING] Probabilities don't sum to 1.0 (sum={total:.4f})")
                    else:
                        print("      [OK] Probabilities sum to ~1.0")

                print("      [OK] Can make predictions")

            except Exception as e:
                print(f"      [FAIL] Prediction failed: {e}")
                passed = False

    return passed


def compare_with_previous(model_path: Path, current_metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    compare current model with previous version if it exists.

    args:
        model_path: path to current model
        current_metrics: metrics from current model

    returns:
        comparison dict or None if no previous version
    """
    print("\nComparing with previous model version...")

    # look for versioned models (e.g., model_v1.joblib, model_v2.joblib)
    model_dir = model_path.parent
    model_name = model_path.stem

    # find all versions
    versions = []
    for f in model_dir.glob(f"{model_name}*.joblib"):
        if f != model_path:
            versions.append(f)

    if not versions:
        print("  No previous version found for comparison")
        return None

    # load most recent previous version (by modification time)
    previous_path = max(versions, key=lambda p: p.stat().st_mtime)
    print(f"  Found previous version: {previous_path.name}")

    try:
        previous_model = joblib.load(previous_path)
        previous_metadata = check_model_metadata(previous_model)

        comparison = {
            "previous_path": str(previous_path),
            "previous_metadata": previous_metadata,
            "current_metadata": current_metrics,
        }

        # compare C-index if available
        if "c_index" in current_metrics and "c_index" in previous_metadata:
            current_c = current_metrics["c_index"]
            previous_c = previous_metadata["c_index"]
            improvement = current_c - previous_c

            print("\n  C-index comparison:")
            print(f"    Previous: {previous_c:.4f}")
            print(f"    Current:  {current_c:.4f}")
            print(f"    Change:   {improvement:+.4f}")

            if improvement > 0:
                print("  [OK] Model improved")
            elif improvement < -0.05:
                print("  [WARNING] Model performance decreased significantly")
            else:
                print("  [INFO] Similar performance")

        return comparison

    except Exception as e:
        print(f"  [WARNING] Failed to compare with previous version: {e}")
        return None


def main():
    """main entry point."""
    parser = argparse.ArgumentParser(description="Validate trained model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model file (e.g., models/survival_model.joblib)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["survival", "classification", "auto"],
        default="auto",
        help="Model type (default: auto-detect)",
    )

    args = parser.parse_args()
    model_path = Path(args.model_path)

    print("=" * 70)
    print("FLARE+ MODEL VALIDATOR")
    print("=" * 70)
    print()

    try:
        # load model
        model = load_model(model_path)

        # check metadata
        metadata = check_model_metadata(model)

        # determine model type
        if args.model_type == "auto":
            if hasattr(model, "predict") and hasattr(model, "is_fitted"):
                model_type = "survival"
            elif hasattr(model, "models"):
                model_type = "classification"
            else:
                print("[ERROR] Could not auto-detect model type. Specify --model-type")
                sys.exit(1)
        else:
            model_type = args.model_type

        print(f"\nDetected model type: {model_type}")

        # validate based on type
        if model_type == "survival":
            passed = validate_survival_model(model, model_path)
        else:
            passed = validate_classification_model(model, model_path)

        # compare with previous version (result logged for context)
        compare_with_previous(model_path, metadata)

        # print summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        if passed:
            print("\n[PASS] Model validation successful")
            print(f"Model is ready for deployment: {model_path}")
            sys.exit(0)
        else:
            print("\n[FAIL] Model validation failed")
            print("Please address the issues above before deploying")
            sys.exit(1)

    except Exception as e:
        logger.error(f"validation failed: {e}", exc_info=True)
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
