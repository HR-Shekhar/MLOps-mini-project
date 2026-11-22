"""Tests for data cleaning operations using synthetic DataFrames."""

import pandas as pd
import pytest
from src.milk_collection.clean import clean_dataframe, standardize_breed


def test_standardize_breed():
    assert standardize_breed("MIXED") == "mixed"
    assert standardize_breed(" mixed ") == "mixed"
    assert standardize_breed("mix.") == "mixed"
    assert standardize_breed("crossbred") == "crossbred"
    assert standardize_breed("cro.") == "crossbred"
    assert standardize_breed("desi") == "desi"
    assert standardize_breed("des.") == "desi"
    assert standardize_breed("Desi ") == "desi"


def test_clean_dataframe():
    raw_sample = pd.DataFrame({
        "record_id": ["R1", "R1", "R2", "R3"],
        "breed_mix": ["MIXED", "MIXED", "cro.", "desi "],
        "buffaloes": [50, 50, -1, 70],
        "litres_paid_in_passbook": [505.0, 505.0, 400.0, 600.0],
        "litres_per_day": [5000.0, 5000.0, 420.0, 610.0],
    })

    cleaned, median_buffaloes = clean_dataframe(raw_sample, is_train=True)

    # 1. Duplicate row removed: length should be 3
    assert len(cleaned) == 3

    # 2. Leaking column dropped
    assert "litres_paid_in_passbook" not in cleaned.columns

    # 3. Target typo fixed (5000 -> 500)
    assert cleaned.loc[cleaned["record_id"] == "R1", "litres_per_day"].iloc[0] == 500.0

    # 4. Sentinel value -1 replaced with median
    assert -1 not in cleaned["buffaloes"].values

    # 5. Breed mix standardized
    assert set(cleaned["breed_mix"].unique()) == {"mixed", "crossbred", "desi"}
