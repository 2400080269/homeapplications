"""Data loading, quality inspection, and leak-free dataset splitting."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    RANDOM_STATE,
    RAW_FEATURES,
    TARGET_CAP_VALUE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_california_housing_data(
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Load the California Housing dataset into a Pandas DataFrame.

    Args:
        data_dir: Local directory to store/read the cached scikit-learn archive.

    Returns:
        Tuple of (full DataFrame, feature column names, target column name).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    housing = fetch_california_housing(data_home=str(data_dir), as_frame=True)
    df = housing.frame.copy()
    return df, list(RAW_FEATURES), TARGET_COLUMN


def inspect_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Audit dataset integrity, censoring boundaries, and structural outliers.

    Args:
        df: Full California Housing DataFrame including target.

    Returns:
        Structured dictionary summarizing missing values, duplicates, censoring, and extremes.
    """
    missing_by_col = df.isnull().sum().to_dict()
    total_missing = int(sum(missing_by_col.values()))
    duplicates = int(df.duplicated().sum())

    target_capped_mask = df[TARGET_COLUMN] >= TARGET_CAP_VALUE
    age_capped_mask = df["HouseAge"] >= 52.0

    extreme_rooms = int((df["AveRooms"] > 20.0).sum())
    extreme_bedrooms = int((df["AveBedrms"] > 5.0).sum())
    extreme_occup = int((df["AveOccup"] > 15.0).sum())

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "total_missing": total_missing,
        "missing_by_column": missing_by_col,
        "duplicate_rows": duplicates,
        "target_capped_count": int(target_capped_mask.sum()),
        "target_capped_pct": float(target_capped_mask.mean() * 100.0),
        "age_capped_count": int(age_capped_mask.sum()),
        "age_capped_pct": float(age_capped_mask.mean() * 100.0),
        "extreme_ave_rooms_gt_20": extreme_rooms,
        "extreme_ave_bedrms_gt_5": extreme_bedrooms,
        "extreme_ave_occup_gt_15": extreme_occup,
        "target_stats": {
            "mean": float(df[TARGET_COLUMN].mean()),
            "std": float(df[TARGET_COLUMN].std()),
            "min": float(df[TARGET_COLUMN].min()),
            "median": float(df[TARGET_COLUMN].median()),
            "max": float(df[TARGET_COLUMN].max()),
        },
    }


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify_by_target_decile: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into train and test partitions before any feature fitting.

    Uses target decile stratification by default so skewed and top-coded values
    are evenly balanced across train and holdout sets.

    Args:
        df: Complete DataFrame.
        test_size: Fraction allocated to the holdout test split.
        random_state: Seed for deterministic splitting.
        stratify_by_target_decile: Whether to stratify across 10 target quantiles.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[RAW_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    stratify_bins = None
    if stratify_by_target_decile:
        stratify_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_bins,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )
