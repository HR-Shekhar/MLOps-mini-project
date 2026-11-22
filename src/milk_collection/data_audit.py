"""Data audit module to hunt for the seven data quality issues."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def check_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: Optional[pd.DataFrame] = None,
    target_col: str = "litres_per_day",
) -> Dict[str, Any]:
    """Check for data leakage: column present in train but absent in holdout features."""
    if holdout_df is not None:
        missing_in_holdout = set(train_df.columns) - set(holdout_df.columns) - {target_col}
        for col in missing_in_holdout:
            if col in train_df.columns:
                return {"present": True, "column": col}
        return {"present": False, "column": None}

    # Fallback if holdout is not provided: check for known leakage naming
    for col in train_df.columns:
        if col != target_col and "passbook" in col.lower():
            return {"present": True, "column": col}
    return {"present": False, "column": None}


def check_unit_mixup(
    df: pd.DataFrame, group_col: str = "village_cluster"
) -> Dict[str, Any]:
    """Check if any cluster has measurements in a completely different unit."""
    if group_col not in df.columns:
        return {"present": False, "column": None, "group_value": None, "factor": None}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ["record_id", "date", "litres_per_day", "litres_paid_in_passbook"]:
            continue
        valid_df = df[df[col] > 0]
        means = valid_df.groupby(group_col)[col].mean()
        if len(means) > 1:
            ratio = means.max() / max(means.min(), 1e-6)
            for factor in [10, 12, 60, 100, 1000]:
                if 0.8 * factor <= ratio <= 1.2 * factor:
                    affected_group = means.idxmax() if factor >= 10 else means.idxmin()
                    return {
                        "present": True,
                        "column": col,
                        "group_value": affected_group,
                        "factor": factor,
                    }

    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for sentinel values (e.g. -1, -999, 9999) typed instead of blank."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ["litres_per_day", "litres_paid_in_passbook"]:
            continue
        # Check negative counts
        negs = df[df[col] < 0]
        if len(negs) > 0:
            val_counts = negs[col].value_counts()
            top_val = val_counts.index[0]
            top_count = int(val_counts.iloc[0])
            if top_count >= 5:
                return {
                    "present": True,
                    "column": col,
                    "value": int(top_val),
                    "count": top_count,
                }
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for duplicate rows in the dataset."""
    extra_copies = int(df.duplicated().sum())
    return {
        "present": extra_copies > 0,
        "count": extra_copies if extra_copies > 0 else None,
    }


def check_stuck_sensor(
    df: pd.DataFrame, date_col: str = "date", min_consecutive_days: int = 14
) -> Dict[str, Any]:
    """Check if any sensor stayed stuck with identical reading for multiple weeks."""
    if date_col not in df.columns:
        return {"present": False, "column": None, "start_date": None, "end_date": None}

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in ["record_id", "litres_per_day", "litres_paid_in_passbook"]:
            continue
        # Group identical consecutive values
        runs = (df_sorted[col] != df_sorted[col].shift()).cumsum()
        run_counts = runs.value_counts()
        for run_id, count in run_counts.items():
            run_df = df_sorted[runs == run_id]
            start_date = pd.to_datetime(run_df[date_col].iloc[0])
            end_date = pd.to_datetime(run_df[date_col].iloc[-1])
            duration_days = (end_date - start_date).days
            if duration_days >= min_consecutive_days:
                return {
                    "present": True,
                    "column": col,
                    "start_date": str(run_df[date_col].iloc[0]),
                    "end_date": str(run_df[date_col].iloc[-1]),
                }

    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(
    df: pd.DataFrame, target_col: str = "litres_per_day", threshold: float = 2000.0
) -> Dict[str, Any]:
    """Check for targets typed 10 times too large."""
    if target_col not in df.columns:
        return {"present": False, "count": None}

    typo_mask = df[target_col] > threshold
    typo_count = int(typo_mask.sum())
    return {
        "present": typo_count > 0,
        "count": typo_count if typo_count > 0 else None,
    }


def check_spelling_variants(
    train_df: pd.DataFrame, later_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Check categorical columns for irregular spellings and extra spaces."""
    cat_cols = ["breed_mix", "village_cluster"]
    for col in cat_cols:
        if col in train_df.columns:
            train_unique = set(train_df[col].dropna().unique())
            # Clean variants: strip, lowercase, remove trailing punctuation
            canonical = set(
                train_df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .str.rstrip(".")
                .unique()
            )
            # If later_df is available, check against clean levels
            if later_df is not None and col in later_df.columns:
                true_levels = len(later_df[col].dropna().unique())
            else:
                true_levels = 3  # known canonical levels for breed_mix: mixed, crossbred, desi

            if len(train_unique) > true_levels:
                return {
                    "present": True,
                    "column": col,
                    "true_levels": true_levels,
                }

    return {"present": False, "column": None, "true_levels": None}


def audit_dataset(
    train_path: str, holdout_path: str, later_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run full audit on raw data files and return findings dict."""
    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)
    later_df = pd.read_csv(later_path) if later_path else None

    return {
        "leaking_column": check_leaking_column(train_df, holdout_df),
        "unit_mixup": check_unit_mixup(train_df),
        "sentinel_value": check_sentinel_value(train_df),
        "duplicate_rows": check_duplicate_rows(train_df),
        "stuck_sensor": check_stuck_sensor(train_df),
        "target_typos": check_target_typos(train_df),
        "spelling_variants": check_spelling_variants(train_df, later_df),
    }


if __name__ == "__main__":
    import yaml
    from src.milk_collection.utils import load_params

    try:
        params = load_params()
        train_p = params["paths"]["raw_train"]
        holdout_p = params["paths"]["raw_holdout"]
        later_p = params["paths"]["raw_later"]
    except Exception:
        train_p = "data/raw/train.csv"
        holdout_p = "data/raw/holdout_features.csv"
        later_p = "data/raw/later.csv"

    print("Running Data Quality Audit for MP137...")
    findings = audit_dataset(train_p, holdout_p, later_p)
    print(yaml.dump({"data_issues": findings}, sort_keys=False))

