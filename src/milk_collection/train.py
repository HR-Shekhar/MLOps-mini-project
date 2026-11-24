"""Model training pipeline with MLflow tracking and champion model registration."""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from src.milk_collection.clean import clean_dataframe
from src.milk_collection.utils import compute_sha256, load_params


def build_pipeline(regressor: Any, numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """Construct an end-to-end preprocessing and regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def train_and_track_models(
    repo_root: str = ".",
    experiment_name: str = "mp137-milk-collection",
    registered_model_name: str = "mp137-milk-predictor",
) -> Dict[str, Any]:
    """Train candidate models, log to MLflow SQLite, register the champion, and save model.joblib."""
    root_path = Path(repo_root).resolve()
    params_file = root_path / "params.yaml"
    params = load_params(str(params_file))

    raw_train_path = root_path / params["paths"]["raw_train"]
    model_save_path = root_path / params["paths"]["model_path"]
    findings_path = root_path / params["paths"]["findings"]

    # 1. Compute original raw train.csv SHA-256
    data_sha256 = compute_sha256(str(raw_train_path))
    print(f"Dataset SHA-256: {data_sha256}")

    # 2. Configure MLflow tracking with SQLite backend in repository root
    db_path = (root_path / "mlflow.db").as_posix()
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {tracking_uri}")

    # 3. Load & clean training data
    raw_df = pd.read_csv(raw_train_path)
    clean_df, median_buffaloes = clean_dataframe(raw_df, is_train=True)
    print(f"Loaded and cleaned dataset: {len(clean_df)} records")

    numeric_features = params["features"]["numeric"]
    categorical_features = params["features"]["categorical"]
    target_col = params["features"]["target"]

    feature_cols = numeric_features + categorical_features
    X = clean_df[feature_cols]
    y = clean_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=params["base"]["random_state"]
    )

    # 4. Define candidate models (3+ distinct configurations)
    candidates = [
        {
            "name": "RandomForest_Setting1",
            "model_type": "RandomForestRegressor",
            "params": {"n_estimators": 50, "max_depth": 6, "random_state": 42},
            "regressor": RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42),
        },
        {
            "name": "RandomForest_Setting2",
            "model_type": "RandomForestRegressor",
            "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 4, "random_state": 42},
            "regressor": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=42),
        },
        {
            "name": "GradientBoosting_Setting3",
            "model_type": "GradientBoostingRegressor",
            "params": {"n_estimators": 100, "learning_rate": 0.08, "max_depth": 4, "random_state": 42},
            "regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42),
        },
        {
            "name": "Ridge_Baseline",
            "model_type": "Ridge",
            "params": {"alpha": 1.0, "random_state": 42},
            "regressor": Ridge(alpha=1.0, random_state=42),
        },
    ]

    runs_info = []

    # 5. Train each model in an MLflow run
    for cand in candidates:
        with mlflow.start_run(run_name=cand["name"]) as run:
            run_id = run.info.run_id

            # Log required data_sha256 and hyperparameters
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", cand["model_type"])
            for p_name, p_val in cand["params"].items():
                mlflow.log_param(p_name, p_val)

            # Fit pipeline
            pipe = build_pipeline(cand["regressor"], numeric_features, categorical_features)
            pipe.fit(X_train, y_train)

            # Evaluate MAE
            y_train_pred = pipe.predict(X_train)
            y_val_pred = pipe.predict(X_val)

            train_mae = float(mean_absolute_error(y_train, y_train_pred))
            val_mae = float(mean_absolute_error(y_val, y_val_pred))

            # Metric name MUST contain 'mae'
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("mae", val_mae)

            # Log model artifact with cloudpickle format for safety
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            runs_info.append({
                "run_id": run_id,
                "name": cand["name"],
                "val_mae": val_mae,
                "pipeline": pipe,
            })
            print(f"Run {cand['name']} ({run_id}): val_mae = {val_mae:.4f}")

    # 6. Select the best run
    runs_info.sort(key=lambda r: r["val_mae"])
    best_run = runs_info[0]
    best_run_id = best_run["run_id"]
    best_pipeline = best_run["pipeline"]
    print(f"\nBest run: {best_run['name']} (ID: {best_run_id}) with val_mae: {best_run['val_mae']:.4f}")

    # 7. Register model in MLflow model registry and assign @champion alias
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    reg_model = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    champion_version = str(reg_model.version)

    # Set alias 'champion' to version 1
    client.set_registered_model_alias(name=registered_model_name, alias="champion", version=champion_version)
    print(f"Registered model '{registered_model_name}' version {champion_version} with alias '@champion'")

    # 8. Save best model to models/model.joblib for production service
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved production champion model to {model_save_path}")

    # 9. Update tracking section in findings.yaml
    for f_path in [findings_path, root_path / "findings.yaml"]:
        if f_path.is_file():
            with open(f_path, "r", encoding="utf-8") as f:
                findings_data = yaml.safe_load(f)

            findings_data["tracking"] = {
                "experiment_name": experiment_name,
                "best_run_id": best_run_id,
                "registered_model_name": registered_model_name,
                "champion_version": int(champion_version),
            }

            with open(f_path, "w", encoding="utf-8") as f:
                yaml.dump(findings_data, f, sort_keys=False)
            print(f"Updated tracking section in {f_path}")

    return {
        "best_run_id": best_run_id,
        "champion_version": champion_version,
        "val_mae": best_run["val_mae"],
        "experiment_name": experiment_name,
        "registered_model_name": registered_model_name,
    }


if __name__ == "__main__":
    train_and_track_models()
