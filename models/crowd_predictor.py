"""
Crowd Prediction Engine (Scikit-learn Random Forest Regressor + Classifier).

Clearly distinguishes synthetic ML model estimates from real-world sensor counts,
and computes ensemble tree agreement for prediction confidence.
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from config import Config
from ml.train_model import EVENT_TYPES, train_and_save_models


class CrowdPredictor:
    """Wrapper around the trained Scikit-learn Random Forest pipeline."""

    def __init__(self) -> None:
        self.bundle: Optional[Dict[str, Any]] = None
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        if not os.path.exists(Config.CROWD_MODEL_PATH):
            train_and_save_models()
        self.bundle = joblib.load(Config.CROWD_MODEL_PATH)

    def reload_model(self) -> Dict[str, Any]:
        meta = train_and_save_models()
        self.bundle = joblib.load(Config.CROWD_MODEL_PATH)
        return meta

    @staticmethod
    def _parse_hour(time_str: Optional[str], default_hr: int = 9) -> int:
        if not time_str:
            return default_hr
        try:
            parts = str(time_str).strip().split(":")
            return max(0, min(23, int(parts[0])))
        except (ValueError, IndexError):
            return default_hr

    @staticmethod
    def _classify_crowd_level(count: int) -> str:
        if count < 400:
            return "Low"
        if count < 950:
            return "Moderate"
        if count < 1800:
            return "High"
        return "Critical"

    @staticmethod
    def _congestion_color(level: str) -> str:
        mapping = {
            "Low": "Green",
            "Moderate": "Yellow",
            "Heavy": "Orange",
            "Severe": "Red",
        }
        return mapping.get(level, "Yellow")

    def predict(
        self,
        hour: Optional[int] = None,
        day_of_week: Optional[int] = None,
        historical_crowd_count: int = 550,
        event_type: str = "None",
        event_start_time: Optional[str] = None,
        event_end_time: Optional[str] = None,
        expected_visitors: int = 0,
        nearby_traffic_level: float = 0.38,
        historical_traffic_conditions: float = 0.32,
        weather_severity: Optional[float] = None,
        weather_configured: bool = False,
        nearby_roads: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run ML prediction and return structured crowd + horizon trend outputs."""
        self._ensure_model_loaded()
        now = datetime.now()
        cur_hour = int(hour) if hour is not None else now.hour
        cur_hour = max(0, min(23, cur_hour))
        cur_day = int(day_of_week) if day_of_week is not None else now.weekday()
        cur_day = max(0, min(6, cur_day))

        clean_event_type = event_type if event_type in EVENT_TYPES else "None"
        if clean_event_type == "None" or expected_visitors <= 0:
            ev_start_hr = 0
            ev_end_hr = 0
            ev_active = 0
            expected_visitors = 0
        else:
            ev_start_hr = self._parse_hour(event_start_time, 9)
            ev_end_hr = self._parse_hour(event_end_time, 17)
            if ev_end_hr < ev_start_hr:
                ev_end_hr = min(23, ev_start_hr + 4)
            ev_active = 1 if (ev_start_hr - 1 <= cur_hour <= ev_end_hr + 1) else 0

        # Weather input is only used as an active live signal when a weather provider is configured
        effective_weather = float(weather_severity) if (weather_configured and weather_severity is not None) else 0.0

        input_row = pd.DataFrame(
            [
                {
                    "hour": cur_hour,
                    "day_of_week": cur_day,
                    "historical_crowd_count": max(20, int(historical_crowd_count)),
                    "event_type": clean_event_type,
                    "event_start_hour": ev_start_hr,
                    "event_end_hour": ev_end_hr,
                    "event_active": ev_active,
                    "expected_visitors": max(0, int(expected_visitors)),
                    "nearby_traffic_ratio": float(np.clip(nearby_traffic_level, 0.0, 1.0)),
                    "historical_traffic_ratio": float(np.clip(historical_traffic_conditions, 0.0, 1.0)),
                    "weather_severity": float(np.clip(effective_weather, 0.0, 1.0)),
                }
            ]
        )

        reg_pipeline = self.bundle["reg_pipeline"]
        clf_pipeline = self.bundle["clf_pipeline"]

        reg_out = reg_pipeline.predict(input_row)[0]
        est_current = max(25, int(round(reg_out[0])))
        est_15m = max(25, int(round(reg_out[1])))
        est_30m = max(25, int(round(reg_out[2])))
        est_60m = max(25, int(round(reg_out[3])))

        pred_congestion = str(clf_pipeline.predict(input_row)[0])
        clf_proba = clf_pipeline.predict_proba(input_row)[0]

        # Calculate ensemble tree variance / class probability agreement for confidence
        transformed_x = reg_pipeline.named_steps["preprocessor"].transform(input_row)
        rf_reg = reg_pipeline.named_steps["regressor"]
        tree_preds = np.array([tree.predict(transformed_x)[0, 0] for tree in rf_reg.estimators_])
        coeff_var = float(np.std(tree_preds) / max(1.0, np.mean(tree_preds)))
        reg_stability = float(np.clip(1.0 - coeff_var, 0.35, 0.96))
        clf_conf = float(np.max(clf_proba))
        ensemble_confidence = round((reg_stability * 0.55 + clf_conf * 0.45) * 100.0, 1)

        # Simulate 12-hour diurnal curve around current day to identify predicted peak time
        hourly_curve = []
        peak_hour = cur_hour
        peak_crowd = est_current
        for h in range(6, 23):
            sim_active = (
                1
                if (clean_event_type != "None" and expected_visitors > 0 and (ev_start_hr - 1 <= h <= ev_end_hr + 1))
                else 0
            )
            sim_df = input_row.copy()
            sim_df["hour"] = h
            sim_df["event_active"] = sim_active
            # Adjust diurnal historical baseline proportionally
            diurnal_scale = (
                np.exp(-0.5 * ((h - 9.0) / 1.8) ** 2)
                + np.exp(-0.5 * ((h - 13.5) / 2.2) ** 2) * 0.65
                + np.exp(-0.5 * ((h - 17.5) / 1.9) ** 2) * 0.95
            )
            sim_df["historical_crowd_count"] = int(max(80, 180 + diurnal_scale * 880))
            h_pred = int(round(reg_pipeline.predict(sim_df)[0][0]))
            hourly_curve.append({"hour": f"{h:02d}:00", "predicted_crowd": max(30, h_pred)})
            if h_pred > peak_crowd:
                peak_crowd = h_pred
                peak_hour = h

        predicted_peak_window = f"{peak_hour:02d}:00 - {(peak_hour + 1) % 24:02d}:00"

        # Identify potential high-risk roads from nearby road network
        high_risk_roads = []
        if nearby_roads:
            for rd in nearby_roads[:8]:
                rd_name = rd.get("road_name") or rd.get("name") or "Unnamed Corridor"
                rd_type = rd.get("road_type", "secondary")
                base_pot = float(rd.get("congestion_potential", 0.55))
                if est_current > 1200 or pred_congestion in ("Heavy", "Severe"):
                    risk_tag = "High Risk" if base_pot >= 0.55 else "Moderate Risk"
                else:
                    risk_tag = "Moderate Risk" if base_pot >= 0.65 else "Low Risk"
                high_risk_roads.append(
                    {
                        "road_name": rd_name,
                        "road_type": rd_type,
                        "risk_level": risk_tag,
                        "estimated_load_pct": int(min(99, round((base_pot * 65) + (est_current / 60.0)))),
                    }
                )
        else:
            high_risk_roads = [
                {
                    "road_name": "Primary Approach Corridor",
                    "road_type": "primary",
                    "risk_level": "High Risk" if pred_congestion in ("Heavy", "Severe") else "Moderate Risk",
                    "estimated_load_pct": min(95, int(nearby_traffic_level * 100)),
                }
            ]

        crowd_level = self._classify_crowd_level(est_current)

        return {
            "estimated_crowd_count": est_current,
            "estimated_crowd_level": crowd_level,
            "predictions_horizon": {
                "current": est_current,
                "next_15_min": est_15m,
                "next_30_min": est_30m,
                "next_60_min": est_60m,
            },
            "trend_direction": (
                "Increasing"
                if est_60m > est_current * 1.05
                else ("Decreasing" if est_60m < est_current * 0.95 else "Stable")
            ),
            "predicted_peak_time": predicted_peak_window,
            "predicted_peak_crowd": peak_crowd,
            "expected_congestion_level": pred_congestion,
            "congestion_color": self._congestion_color(pred_congestion),
            "potential_high_risk_roads": high_risk_roads,
            "hourly_forecast_curve": hourly_curve,
            "confidence_metrics": {
                "ensemble_agreement_pct": ensemble_confidence,
                "tree_std_dev": round(float(np.std(tree_preds)), 1),
                "validation_note": (
                    "Confidence reflects Random Forest ensemble tree agreement (120 trees) "
                    "trained on the synthetic demonstration dataset, not field sensor validation."
                ),
            },
            "weather_status": (
                "Active (Live Weather Provider Configured)"
                if weather_configured
                else "Not Configured (Neutral Baseline Used)"
            ),
            "model_metadata": self.bundle.get("metadata", {}),
            "data_source": "ml_prediction_synthetic_trained",
            "data_label": "Estimated Data (ML Model trained on Synthetic Dataset)",
        }


crowd_predictor = CrowdPredictor()

