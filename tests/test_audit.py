"""Tests for data auditing logic using synthetic in-memory DataFrames."""

import pandas as pd
import pytest
from src.milk_collection.data_audit import (
    check_duplicate_rows,
    check_leaking_column,
    check_sentinel_value,
    check_spelling_variants,
    check_stuck_sensor,
    check_target_typos,
    check_unit_mixup,
)


@pytest.fixture
def sample_clean_df():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "cows": [50, 60, 55, 65],
        "buffaloes": [30, 40, 35, 45],
        "fodder_price_rs": [25.0, 26.0, 24.5, 25.5],
        "temp_c": [22.0, 23.0, 21.5, 24.0],
        "vet_visits": [2, 3, 1, 4],
        "feed_kg": [1500, 1600, 1550, 1650],
        "breed_mix": ["mixed", "crossbred", "desi", "mixed"],
        "village_cluster": ["Cluster A", "Cluster B", "Cluster A", "Cluster C"],
        "litres_per_day": [500.0, 600.0, 550.0, 650.0],
    })


def test_check_duplicate_rows(sample_clean_df):
    assert check_duplicate_rows(sample_clean_df)["present"] is False

    # Introduce duplicate
    df_dup = pd.concat([sample_clean_df, sample_clean_df.iloc[[0]]], ignore_index=True)
    res = check_duplicate_rows(df_dup)
    assert res["present"] is True
    assert res["count"] == 1


def test_check_leaking_column(sample_clean_df):
    holdout_df = sample_clean_df.drop(columns=["litres_per_day"])
    assert check_leaking_column(sample_clean_df, holdout_df)["present"] is False

    # Add leak column
    df_leak = sample_clean_df.copy()
    df_leak["litres_paid_in_passbook"] = df_leak["litres_per_day"] + 2.0
    res = check_leaking_column(df_leak, holdout_df)
    assert res["present"] is True
    assert res["column"] == "litres_paid_in_passbook"


def test_check_sentinel_value(sample_clean_df):
    assert check_sentinel_value(sample_clean_df)["present"] is False

    df_sentinel = sample_clean_df.copy()
    # add 5 negative sentinel rows
    for i in range(5):
        df_sentinel.loc[len(df_sentinel)] = [f"R_neg_{i}", "2026-01-05", 50, -1, 25.0, 22.0, 2, 1500, "mixed", "Cluster A", 500.0]
    res = check_sentinel_value(df_sentinel)
    assert res["present"] is True
    assert res["column"] == "buffaloes"
    assert res["value"] == -1
    assert res["count"] == 5


def test_check_target_typos(sample_clean_df):
    assert check_target_typos(sample_clean_df)["present"] is False

    df_typo = sample_clean_df.copy()
    df_typo.loc[0, "litres_per_day"] = 5000.0
    res = check_target_typos(df_typo)
    assert res["present"] is True
    assert res["count"] == 1


def test_check_spelling_variants():
    df_clean = pd.DataFrame({"breed_mix": ["mixed", "crossbred", "desi"]})
    assert check_spelling_variants(df_clean)["present"] is False

    df_dirty = pd.DataFrame({"breed_mix": ["mixed", "crossbred", "desi", "MIXED", "des.", "cro."]})
    res = check_spelling_variants(df_dirty)
    assert res["present"] is True
    assert res["column"] == "breed_mix"
    assert res["true_levels"] == 3
