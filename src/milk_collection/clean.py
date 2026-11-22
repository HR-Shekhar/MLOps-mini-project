"""Data cleaning pipeline to clean raw records."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def standardize_breed(val: str) -> str:
    """Standardize breed_mix string into canonical category ('mixed', 'crossbred', 'desi')."""
    if pd.isna(val):
        return "mixed"
    s = str(val).strip().lower().rstrip(".")
    if "mix" in s:
        return "mixed"
    if "cro" in s:
        return "crossbred"
    if "des" in s:
        return "desi"
    return "mixed"


def clean_dataframe(
    df: pd.DataFrame,
    is_train: bool = True,
    buffalo_median: Optional[float] = None,
    drop_leak: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """Clean the raw milk collection dataframe.

    Operations performed:
    1. Removes duplicate rows (if train).
    2. Drops leaking column 'litres_paid_in_passbook' if drop_leak is True.
    3. Standardizes 'breed_mix' categories to canonical levels.
    4. Handles sentinel value in 'buffaloes' (-1) by replacing with median.
    5. Fixes target typos in 'litres_per_day' (values > 2000 divided by 10).
    """
    cleaned = df.copy()

    # 1. Remove duplicate rows
    if is_train:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2. Drop leaking column
    if drop_leak and "litres_paid_in_passbook" in cleaned.columns:
        cleaned = cleaned.drop(columns=["litres_paid_in_passbook"])

    # 3. Standardize breed_mix
    if "breed_mix" in cleaned.columns:
        cleaned["breed_mix"] = cleaned["breed_mix"].apply(standardize_breed)

    # 4. Handle sentinel values in buffaloes (-1)
    if "buffaloes" in cleaned.columns:
        valid_buffaloes = cleaned.loc[cleaned["buffaloes"] >= 0, "buffaloes"]
        if buffalo_median is None:
            computed_median = float(valid_buffaloes.median()) if len(valid_buffaloes) > 0 else 70.0
        else:
            computed_median = buffalo_median
        cleaned["buffaloes"] = cleaned["buffaloes"].replace(-1, computed_median)
    else:
        computed_median = 70.0

    # 5. Fix target typos if present in target column
    if "litres_per_day" in cleaned.columns:
        typo_mask = cleaned["litres_per_day"] > 2000.0
        cleaned.loc[typo_mask, "litres_per_day"] = cleaned.loc[typo_mask, "litres_per_day"] / 10.0

    return cleaned, computed_median


if __name__ == "__main__":
    from src.milk_collection.utils import load_params

    try:
        params = load_params()
        in_p = params["paths"]["raw_train"]
        out_p = params["paths"]["cleaned_train"]
    except Exception:
        in_p = "data/raw/train.csv"
        out_p = "data/cleaned_train.csv"

    print(f"Cleaning raw dataset from {in_p}...")
    raw = pd.read_csv(in_p)
    cleaned, med = clean_dataframe(raw, is_train=True)
    cleaned.to_csv(out_p, index=False)
    print(f"Successfully cleaned: {len(raw)} -> {len(cleaned)} rows saved to {out_p} (imputed buffalo median: {med})")

