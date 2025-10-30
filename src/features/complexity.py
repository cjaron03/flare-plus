"""sunspot complexity metrics from mcintosh and mount wilson classifications."""

import logging
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


def compute_mcintosh_complexity(mcintosh_class: Optional[str]) -> Dict[str, Any]:
    """
    compute complexity metrics from mcintosh sunspot classification.

    mcintosh classification format: [size][shape][penumbra]
    size: a, b, c, d, e, f, h, j
    shape: x, r, s, a, h, k
    penumbra: x, r, s, a, h, k

    args:
        mcintosh_class: mcintosh classification string (e.g., "Dkc", "Ekc")

    returns:
        dict with complexity metrics
    """
    if not mcintosh_class or pd.isna(mcintosh_class):
        return {
            "mcintosh_size": None,
            "mcintosh_shape": None,
            "mcintosh_penumbra": None,
            "mcintosh_size_encoded": 0,
            "mcintosh_shape_encoded": 0,
            "mcintosh_penumbra_encoded": 0,
            "mcintosh_complexity_score": 0.0,
        }

    mcintosh_class = str(mcintosh_class).strip().upper()

    # parse mcintosh classification
    if len(mcintosh_class) >= 3:
        size = mcintosh_class[0]
        shape = mcintosh_class[1] if len(mcintosh_class) > 1 else None
        penumbra = mcintosh_class[2] if len(mcintosh_class) > 2 else None
    else:
        size = mcintosh_class[0] if len(mcintosh_class) > 0 else None
        shape = None
        penumbra = None

    # encode size (a=1, b=2, c=3, d=4, e=5, f=6, h=7, j=8)
    size_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "H": 7, "J": 8}
    size_encoded = size_map.get(size, 0)

    # encode shape (x=0, r=1, s=2, a=3, h=4, k=5)
    # higher complexity: k > h > a > s > r > x
    shape_map = {"X": 0, "R": 1, "S": 2, "A": 3, "H": 4, "K": 5}
    shape_encoded = shape_map.get(shape, 0) if shape else 0

    # encode penumbra (x=0, r=1, s=2, a=3, h=4, k=5)
    penumbra_map = {"X": 0, "R": 1, "S": 2, "A": 3, "H": 4, "K": 5}
    penumbra_encoded = penumbra_map.get(penumbra, 0) if penumbra else 0

    # compute overall complexity score (normalized 0-1)
    # weight: size (40%), shape (30%), penumbra (30%)
    max_size = 8
    max_shape_penumbra = 5
    complexity_score = (
        (size_encoded / max_size) * 0.4
        + (shape_encoded / max_shape_penumbra) * 0.3
        + (penumbra_encoded / max_shape_penumbra) * 0.3
    )

    return {
        "mcintosh_size": size,
        "mcintosh_shape": shape,
        "mcintosh_penumbra": penumbra,
        "mcintosh_size_encoded": size_encoded,
        "mcintosh_shape_encoded": shape_encoded,
        "mcintosh_penumbra_encoded": penumbra_encoded,
        "mcintosh_complexity_score": complexity_score,
    }


def compute_mount_wilson_complexity(mount_wilson_class: Optional[str]) -> Dict[str, Any]:
    """
    compute complexity metrics from mount wilson magnetic classification.

    mount wilson classes: alpha, beta, gamma, delta, beta-gamma, beta-gamma-delta, etc.
    complexity increases: alpha < beta < gamma < delta

    args:
        mount_wilson_class: mount wilson classification string

    returns:
        dict with complexity metrics
    """
    if not mount_wilson_class or pd.isna(mount_wilson_class):
        return {
            "mount_wilson_class": None,
            "mount_wilson_has_alpha": 0,
            "mount_wilson_has_beta": 0,
            "mount_wilson_has_gamma": 0,
            "mount_wilson_has_delta": 0,
            "mount_wilson_complexity_score": 0.0,
        }

    mount_wilson_class = str(mount_wilson_class).strip().lower()

    # check for presence of each component
    has_alpha = 1 if "alpha" in mount_wilson_class else 0
    has_beta = 1 if "beta" in mount_wilson_class else 0
    has_gamma = 1 if "gamma" in mount_wilson_class else 0
    has_delta = 1 if "delta" in mount_wilson_class else 0

    # compute complexity score
    # base scores: alpha=1, beta=2, gamma=3, delta=4
    # combinations add complexity
    score = 0.0
    if has_alpha:
        score += 1.0
    if has_beta:
        score += 2.0
    if has_gamma:
        score += 3.0
    if has_delta:
        score += 4.0

    # normalize to 0-1 range (max possible: 4.0 for delta-only or combination)
    complexity_score = min(score / 4.0, 1.0)

    return {
        "mount_wilson_class": mount_wilson_class,
        "mount_wilson_has_alpha": has_alpha,
        "mount_wilson_has_beta": has_beta,
        "mount_wilson_has_gamma": has_gamma,
        "mount_wilson_has_delta": has_delta,
        "mount_wilson_complexity_score": complexity_score,
    }


def compute_magnetic_complexity_score(
    magnetic_type: Optional[str], magnetic_complexity: Optional[str] = None
) -> Dict[str, Any]:
    """
    compute magnetic complexity score from magnetogram data.

    combines magnetic type and complexity information to create a unified score.

    args:
        magnetic_type: magnetic type string (e.g., "beta-gamma")
        magnetic_complexity: magnetic complexity string (e.g., "beta-gamma-delta")

    returns:
        dict with magnetic complexity metrics
    """
    # use magnetic_complexity if available, otherwise fall back to magnetic_type
    complexity_str = magnetic_complexity if magnetic_complexity else magnetic_type

    if not complexity_str or pd.isna(complexity_str):
        return {
            "magnetic_complexity": None,
            "magnetic_has_alpha": 0,
            "magnetic_has_beta": 0,
            "magnetic_has_gamma": 0,
            "magnetic_has_delta": 0,
            "magnetic_complexity_score": 0.0,
        }

    complexity_str = str(complexity_str).strip().lower()

    # check for presence of each component
    has_alpha = 1 if "alpha" in complexity_str else 0
    has_beta = 1 if "beta" in complexity_str else 0
    has_gamma = 1 if "gamma" in complexity_str else 0
    has_delta = 1 if "delta" in complexity_str else 0

    # compute complexity score (same as mount wilson)
    score = 0.0
    if has_alpha:
        score += 1.0
    if has_beta:
        score += 2.0
    if has_gamma:
        score += 3.0
    if has_delta:
        score += 4.0

    # normalize to 0-1 range
    complexity_score = min(score / 4.0, 1.0)

    return {
        "magnetic_complexity": complexity_str,
        "magnetic_has_alpha": has_alpha,
        "magnetic_has_beta": has_beta,
        "magnetic_has_gamma": has_gamma,
        "magnetic_has_delta": has_delta,
        "magnetic_complexity_score": complexity_score,
    }
