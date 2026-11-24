"""Tests for champion model loading and inference using synthetic test data."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest


def test_champion_model_exists():
    model_path = Path("models/model.joblib")
    assert model_path.is_file(), "models/model.joblib does not exist"


def test_champion_model_inference():
    model_path = Path("models/model.joblib")
    model = joblib.load(model_path)

    # Synthetic sample request with exact 8 inference features
    sample_df = pd.DataFrame({
        "cows": [100, 50],
        "buffaloes": [70, 40],
        "fodder_price_rs": [25.0, 20.0],
        "temp_c": [28.0, 32.0],
        "vet_visits": [5, 2],
        "feed_kg": [1500, 1200],
        "breed_mix": ["mixed", "crossbred"],
        "village_cluster": ["Cluster A", "Cluster B"],
    })

    preds = model.predict(sample_df)
    assert len(preds) == 2
    assert isinstance(preds, np.ndarray)
    assert np.all(preds > 0), "Predicted litres must be positive"
    assert np.all(preds < 2000), "Predicted litres must be realistic daily volumes"
