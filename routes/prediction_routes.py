"""
ML Crowd Prediction, Crowd-Traffic Fusion Risk Engine, and Smart Routing Routes.
Implements:
- POST /api/prediction/crowd
- POST /api/prediction/congestion
- POST /api/prediction/routes
"""
import json
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request

from config import Config
from database.db import get_db_connection
from models.congestion_engine import congestion_engine
from models.crowd_predictor import crowd_predictor
from services.location_service import validate_coordinates
from services.routing_service import routing_service
from services.traffic_service import traffic_service

prediction_bp = Blueprint("prediction_bp", __name__)


@prediction_bp.route("/api/prediction/crowd", methods=["POST"])
def predict_crowd():
    """
    Run Scikit-learn Random Forest crowd prediction for current, +15m, +30m, and +60m horizons,
    log prediction to SQLite, and return structured forecast with synthetic dataset disclaimer.
    """
    payload = request.get_json(silent=True) or {}
    lat_raw = payload.get("latitude", 13.0108)
    lon_raw = payload.get("longitude", 80.2354)
    is_valid, err_msg, lat, lon = validate_coordinates(lat_raw, lon_raw)
    if not is_valid:
        return jsonify({"status": "error", "error": err_msg}), 400

    location_label = str(payload.get("location_label", f"Area ({lat:.4f}, {lon:.4f})"))
    hour = payload.get("hour")
    day_of_week = payload.get("day_of_week")
    historical_crowd = int(payload.get("historical_crowd_count", 600))
    event_type = str(payload.get("event_type", "None"))
    event_start_time = payload.get("event_start_time")
    event_end_time = payload.get("event_end_time")
    expected_visitors = int(payload.get("expected_visitors", 0))
    nearby_traffic_level = float(payload.get("nearby_traffic_level", 0.42))
    historical_traffic = float(payload.get("historical_traffic_conditions", 0.35))
    weather_severity = payload.get("weather_severity")
    nearby_roads = payload.get("nearby_roads")

    # If user didn't pass explicit event parameters, check if there's an active event in DB
    if event_type == "None" and expected_visitors == 0 and payload.get("use_scheduled_events", True):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events ORDER BY expected_participants DESC LIMIT 1")
        ev = cursor.fetchone()
        conn.close()
        if ev:
            event_type = ev["event_category"]
            event_start_time = ev["start_time"]
            event_end_time = ev["end_time"]
            expected_visitors = int(ev["expected_participants"])

    weather_configured = bool(Config.OPENWEATHER_API_KEY)

    try:
        prediction = crowd_predictor.predict(
            hour=hour,
            day_of_week=day_of_week,
            historical_crowd_count=historical_crowd,
            event_type=event_type,
            event_start_time=event_start_time,
            event_end_time=event_end_time,
            expected_visitors=expected_visitors,
            nearby_traffic_level=nearby_traffic_level,
            historical_traffic_conditions=historical_traffic,
            weather_severity=weather_severity,
            weather_configured=weather_configured,
            nearby_roads=nearby_roads,
        )

        # Persist prediction in SQLite
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO crowd_predictions (
                location_label, latitude, longitude, estimated_crowd_count, crowd_level,
                pred_15m, pred_30m, pred_60m, predicted_peak_time, expected_congestion_level,
                confidence_score, model_name, data_source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location_label,
                lat,
                lon,
                prediction["estimated_crowd_count"],
                prediction["estimated_crowd_level"],
                prediction["predictions_horizon"]["next_15_min"],
                prediction["predictions_horizon"]["next_30_min"],
                prediction["predictions_horizon"]["next_60_min"],
                prediction["predicted_peak_time"],
                prediction["expected_congestion_level"],
                prediction["confidence_metrics"]["ensemble_agreement_pct"],
                prediction["model_metadata"].get("model_name", "CrowdCast-RF-Ensemble-v1"),
                prediction["data_source"],
                now_iso,
            ),
        )
        conn.commit()
        conn.close()

        return jsonify({"status": "ok", "prediction": prediction}), 200
    except Exception as exc:
        return jsonify({"status": "error", "error": f"Crowd prediction error: {exc}"}), 500


@prediction_bp.route("/api/prediction/congestion", methods=["POST"])
def predict_congestion_fusion():
    """
    Execute the Crowd-Traffic Fusion Engine (0-100 risk score) combining traffic flow,
    estimated crowd, historical diurnal patterns, event impact, and road capacity indicators.
    """
    payload = request.get_json(silent=True) or {}
    lat_raw = payload.get("latitude", 13.0108)
    lon_raw = payload.get("longitude", 80.2354)
    is_valid, err_msg, lat, lon = validate_coordinates(lat_raw, lon_raw)
    if not is_valid:
        return jsonify({"status": "error", "error": err_msg}), 400

    location_label = str(payload.get("location_label", f"Area ({lat:.4f}, {lon:.4f})"))
    hour = payload.get("hour")
    day_of_week = payload.get("day_of_week")
    custom_weights = payload.get("weights")

    # Fetch stored road segments, events, and campus gates to enrich fusion
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM road_segments ORDER BY congestion_potential DESC LIMIT 12")
    db_roads = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM events ORDER BY event_date DESC, start_time ASC")
    db_events = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM campus_gates ORDER BY id ASC")
    db_gates = [dict(r) for r in cursor.fetchall()]
    conn.close()

    traffic_ratio = payload.get("traffic_congestion_ratio")
    traffic_source = payload.get("traffic_data_source", "demonstration_sample_traffic")
    if traffic_ratio is None:
        t_flow = traffic_service.fetch_current_traffic(lat, lon, nearby_roads=db_roads)
        traffic_ratio = float(t_flow["summary"]["congestion_ratio"])
        traffic_source = t_flow["data_source"]
        # Attach segment-level congestion ratios
        road_segments = t_flow.get("segments", db_roads)
    else:
        traffic_ratio = float(traffic_ratio)
        road_segments = payload.get("road_segments") or db_roads

    total_event_visitors = int(
        payload.get(
            "expected_event_visitors",
            sum(int(e.get("expected_participants", 0)) for e in db_events),
        )
    )
    total_event_vehicles = int(
        payload.get(
            "expected_event_vehicles",
            sum(int(e.get("expected_vehicles", 0)) for e in db_events),
        )
    )
    active_events_count = len(db_events) if total_event_visitors > 0 else 0

    estimated_crowd = payload.get("estimated_crowd_count")
    if estimated_crowd is None:
        c_pred = crowd_predictor.predict(
            hour=hour,
            day_of_week=day_of_week,
            expected_visitors=total_event_visitors,
            event_type=db_events[0]["event_category"] if db_events else "None",
            nearby_traffic_level=traffic_ratio,
        )
        estimated_crowd = int(c_pred["estimated_crowd_count"])
    else:
        estimated_crowd = int(estimated_crowd)

    fusion_result = congestion_engine.calculate_fusion_risk(
        traffic_congestion_ratio=traffic_ratio,
        estimated_crowd_count=estimated_crowd,
        expected_event_visitors=total_event_visitors,
        expected_event_vehicles=total_event_vehicles,
        active_events_count=active_events_count,
        hour=hour,
        day_of_week=day_of_week,
        road_segments=road_segments,
        campus_gates=db_gates,
        custom_weights=custom_weights,
        traffic_data_source=traffic_source,
    )

    # Also compute smart route & gate recommendations
    routing_recommendations = routing_service.recommend_routes_and_gates(
        origin_lat=lat,
        origin_lon=lon,
        dest_lat=payload.get("dest_lat"),
        dest_lon=payload.get("dest_lon"),
        traffic_segments=road_segments,
        estimated_crowd_count=estimated_crowd,
        expected_event_visitors=total_event_visitors,
        traffic_is_live=(traffic_source == "live_tomtom_api"),
    )
    fusion_result["routing_and_gates"] = routing_recommendations

    # Persist fusion prediction in SQLite
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO congestion_predictions (
            location_label, latitude, longitude, risk_score, risk_level, color_code,
            traffic_component, crowd_component, historical_component, event_component,
            expected_peak_period, contributing_factors_json, recommended_actions_json,
            weights_used_json, data_source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            location_label,
            lat,
            lon,
            fusion_result["congestion_risk_score"],
            fusion_result["risk_level"],
            fusion_result["color_code"],
            fusion_result["sub_scores"]["traffic_congestion_score"],
            fusion_result["sub_scores"]["expected_crowd_score"],
            fusion_result["sub_scores"]["historical_pattern_score"],
            fusion_result["sub_scores"]["event_impact_score"],
            fusion_result["expected_peak_period"],
            json.dumps(fusion_result["main_contributing_factors"]),
            json.dumps(fusion_result["recommended_actions"]),
            json.dumps(fusion_result["weights_used"]),
            fusion_result["data_source"],
            now_iso,
        ),
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "ok", "fusion": fusion_result}), 200

