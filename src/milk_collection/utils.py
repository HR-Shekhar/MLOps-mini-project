"""Utility helper functions."""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict
import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration parameters from YAML file safely."""
    path = Path(params_path)
    if not path.is_file():
        # Fallback to search in parent directories
        for parent in path.resolve().parents:
            candidate = parent / "params.yaml"
            if candidate.is_file():
                path = candidate
                break
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sha256(file_path: str) -> str:
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
