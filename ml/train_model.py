"""
CrowdCast AI - Synthetic Dataset Generator & Random Forest Training Pipeline.

NOTE ON DATA INTEGRITY:
The dataset generated and used by this module is a SYNTHETIC DEMONSTRATION DATASET
engineered to reflect realistic diurnal university/urban traffic and event patterns.
Metrics reported during training reflect generalization on synthetic holdout splits,
NOT real-world field validation.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config

EVENT_TYPES = [
    "None",
    "College symposiums",
    "Technical festivals",
    "Sports events",
    "Public gatherings",
    "College admissions",
    "Seminars",
    "Cultural events",
]

EVENT_MULTIPLIERS = {
    "None": 0.0,
    "College symposiums": 0.85,
    "Technical festivals": 1.15,
    "Sports events": 1.25,
    "Public gatherings": 1.30,
    "College admissions": 0.90,
    "Seminars": 0.55,
    "Cultural events": 1.20,
}


def generate_synthetic_dataset(n_samples: int = 1800, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset for demonstration and baseline ML training."""
    rng = np.random.default_rng(random_state)

    hours = rng.integers(0, 24, size=n_samples)
    days = rng.integers(0, 7, size=n_samples)  # 0=Monday ... 6=Sunday
    event_types = rng.choice(
        EVENT_TYPES,
        size=n_samples,
        p=[0.36, 0.11, 0.10, 0.10, 0.08, 0.09, 0.08, 0.08],
    )

    records = []
    for i in range(n_samples):
        hr = int(hours[i])
        day = int(days[i])
        ev_type = str(event_types[i])
        is_weekend = 1 if day >= 5 else 0

        # Base diurnal pattern (morning rush 8-10, afternoon 13-14, evening rush 16-19)
        morning_peak = np.exp(-0.5 * ((hr - 9.0) / 1.6) ** 2)
        afternoon_wave = np.exp(-0.5 * ((hr - 13.5) / 2.0) ** 2) * 0.65
        evening_peak = np.exp(-0.5 * ((hr - 17.5) / 1.8) ** 2) * 0.95
        diurnal_factor = max(0.08, morning_peak + afternoon_wave + evening_peak)

        if is_weekend:
            diurnal_factor *= 0.62

        # Historical baseline crowd for this slot
        historical_crowd = int(
            np.clip(
                180 + diurnal_factor * 950 + rng.normal(0, 65),
                40,
                3500,
            )
        )

        # Event parameters
        if ev_type == "None":
            ev_start_hr = 0
            ev_end_hr = 0
            ev_active = 0
            expected_visitors = 0
        else:
            ev_start_hr = int(rng.choice([8, 9, 10, 14, 15, 16]))
            ev_duration = int(rng.choice([3, 4, 6, 8]))
            ev_end_hr = min(23, ev_start_hr + ev_duration)
            ev_active = 1 if (ev_start_hr - 1 <= hr <= ev_end_hr + 1) else 0
            base_vis = {
                "College symposiums": 900,
                "Technical festivals": 1800,
                "Sports events": 2100,
                "Public gatherings": 2400,
                "College admissions": 1100,
                "Seminars": 400,
                "Cultural events": 1900,
            }[ev_type]
            expected_visitors = int(max(100, rng.normal(base_vis, base_vis * 0.22)))

        # Historical & current nearby traffic congestion ratios (0.05 to 0.98)
        historical_traffic_ratio = float(
            np.clip(0.15 + 0.48 * (diurnal_factor / 1.5) + rng.normal(0, 0.06), 0.05, 0.95)
        )
        event_traffic_boost = (
            (expected_visitors / 3000.0) * EVENT_MULTIPLIERS[ev_type] * (0.85 if ev_active else 0.2)
        )
        nearby_traffic_ratio = float(
            np.clip(
                historical_traffic_ratio + event_traffic_boost + rng.normal(0, 0.07),
                0.05,
                0.99,
            )
        )

        # Optional weather factor (0 = clear, 0.5 = rain, 1.0 = storm)
        weather_severity = float(rng.choice([0.0, 0.0, 0.0, 0.2, 0.5, 0.8]))

        # Active event surge calculation
        if ev_active and expected_visitors > 0:
            ev_mid = (ev_start_hr + ev_end_hr) / 2.0
            proximity = np.exp(-0.5 * ((hr - ev_mid) / 2.2) ** 2)
            event_crowd_contribution = expected_visitors * (0.45 + 0.55 * proximity)
        else:
            event_crowd_contribution = expected_visitors * 0.08

        # Target 1: Current / Immediate Estimated Crowd Count
        target_crowd = int(
            np.clip(
                historical_crowd * 0.78
                + event_crowd_contribution * EVENT_MULTIPLIERS.get(ev_type, 1.0)
                + nearby_traffic_ratio * 340
                - weather_severity * 90
                + rng.normal(0, 45),
                35,
                6500,
            )
        )

        # Trend targets (+15m, +30m, +60m)
        next_hr_slope = 1.08 if hr in (7, 8, 15, 16) or (ev_active and hr <= ev_start_hr + 1) else (
            0.91 if hr in (11, 19, 20, 21, 22) else 1.02
        )
        crowd_15m = int(np.clip(target_crowd * (1.0 + (next_hr_slope - 1.0) * 0.35) + rng.normal(0, 20), 30, 6800))
        crowd_30m = int(np.clip(target_crowd * (1.0 + (next_hr_slope - 1.0) * 0.68) + rng.normal(0, 30), 30, 7000))
        crowd_60m = int(np.clip(target_crowd * next_hr_slope + rng.normal(0, 45), 30, 7200))

        # Congestion level classification target
        composite_load = (target_crowd / 2200.0) * 0.5 + nearby_traffic_ratio * 0.5
        if composite_load < 0.35:
            congestion_class = "Low"
        elif composite_load < 0.58:
            congestion_class = "Moderate"
        elif composite_load < 0.78:
            congestion_class = "Heavy"
        else:
            congestion_class = "Severe"

        records.append(
            {
                "hour": hr,
                "day_of_week": day,
                "historical_crowd_count": historical_crowd,
                "event_type": ev_type,
                "event_start_hour": ev_start_hr,
                "event_end_hour": ev_end_hr,
                "event_active": ev_active,
                "expected_visitors": expected_visitors,
                "nearby_traffic_ratio": round(nearby_traffic_ratio, 4),
                "historical_traffic_ratio": round(historical_traffic_ratio, 4),
                "weather_severity": round(weather_severity, 2),
                "target_crowd_count": target_crowd,
                "target_crowd_15m": crowd_15m,
                "target_crowd_30m": crowd_30m,
                "target_crowd_60m": crowd_60m,
                "target_congestion_level": congestion_class,
                "is_synthetic": 1,
            }
        )

    return pd.DataFrame(records)


def train_and_save_models() -> dict:
    """
    Generate synthetic dataset (if needed), perform strict train/test split BEFORE
    fitting preprocessing pipelines, train Random Forest models, and save artifacts.
    """
    os.makedirs(Config.ML_DIR, exist_ok=True)
    os.makedirs(Config.SAVED_MODEL_DIR, exist_ok=True)

    df = generate_synthetic_dataset(n_samples=1800, random_state=42)
    df.to_csv(Config.DATASET_PATH, index=False)

    # Check and handle missing/NULL values explicitly per ML best practices
    null_counts = df.isnull().sum().to_dict()
    df = df.dropna(subset=["target_crowd_count", "target_congestion_level"])

    feature_cols = [
        "hour",
        "day_of_week",
        "historical_crowd_count",
        "event_type",
        "event_start_hour",
        "event_end_hour",
        "event_active",
        "expected_visitors",
        "nearby_traffic_ratio",
        "historical_traffic_ratio",
        "weather_severity",
    ]
    numeric_features = [
        "hour",
        "day_of_week",
        "historical_crowd_count",
        "event_start_hour",
        "event_end_hour",
        "event_active",
        "expected_visitors",
        "nearby_traffic_ratio",
        "historical_traffic_ratio",
        "weather_severity",
    ]
    categorical_features = ["event_type"]

    X = df[feature_cols]
    y_reg = df[["target_crowd_count", "target_crowd_15m", "target_crowd_30m", "target_crowd_60m"]]
    y_clf = df["target_congestion_level"]

    # Strict Featurization Ordering: Split BEFORE fitting preprocessing pipelines
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor_reg = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    preprocessor_clf = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_reg),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=14,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_clf),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Fit strictly on training split
    reg_pipeline.fit(X_train, y_reg_train)
    clf_pipeline.fit(X_train, y_clf_train)

    # Evaluate on held-out synthetic test split
    reg_preds = reg_pipeline.predict(X_test)
    clf_preds = clf_pipeline.predict(X_test)

    mae_current = float(mean_absolute_error(y_reg_test["target_crowd_count"], reg_preds[:, 0]))
    rmse_current = float(np.sqrt(mean_squared_error(y_reg_test["target_crowd_count"], reg_preds[:, 0])))
    r2_current = float(r2_score(y_reg_test["target_crowd_count"], reg_preds[:, 0]))
    clf_acc = float(accuracy_score(y_clf_test, clf_preds))
    clf_f1 = float(f1_score(y_clf_test, clf_preds, average="weighted"))

    metadata = {
        "model_name": "CrowdCast-RF-Ensemble-v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "SYNTHETIC_DEMONSTRATION_DATASET",
        "dataset_disclaimer": (
            "Trained on synthetic demonstration data (ml/dataset.csv). Metrics reflect "
            "synthetic holdout split validation, NOT real-world field accuracy."
        ),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "null_values_checked": null_counts,
        "feature_columns": feature_cols,
        "synthetic_test_metrics": {
            "crowd_mae": round(mae_current, 2),
            "crowd_rmse": round(rmse_current, 2),
            "crowd_r2": round(r2_current, 4),
            "congestion_accuracy": round(clf_acc, 4),
            "congestion_f1_weighted": round(clf_f1, 4),
        },
    }

    artifact_bundle = {
        "reg_pipeline": reg_pipeline,
        "clf_pipeline": clf_pipeline,
        "metadata": metadata,
    }
    joblib.dump(artifact_bundle, Config.CROWD_MODEL_PATH)

    with open(Config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    info = train_and_save_models()
    print("Trained CrowdCast AI ML Models Successfully:")
    print(json.dumps(info["synthetic_test_metrics"], indent=2))

