"""
Event-Aware Prediction & Event Management Routes (Section 7).
Implements:
- GET    /api/events
- POST   /api/events
- PUT    /api/events/<int:event_id>
- DELETE /api/events/<int:event_id>
"""
from datetime import datetime, timezone
from typing import Any, Dict
from flask import Blueprint, jsonify, request

from database.db import get_db_connection
from models.crowd_predictor import crowd_predictor

event_bp = Blueprint("event_bp", __name__)

SUPPORTED_EVENT_CATEGORIES = [
    "College symposiums",
    "Technical festivals",
    "Sports events",
    "Public gatherings",
    "College admissions",
    "Seminars",
    "Cultural events",
]


def compute_event_impact_comparison(event_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare normal baseline traffic/crowd patterns against event-day predictions
    across the event's active window.
    """
    participants = max(0, int(event_row.get("expected_participants", 200)))
    vehicles = max(0, int(event_row.get("expected_vehicles", 60)))
    gates = max(1, int(event_row.get("entry_gates_count", 2)))
    category = event_row.get("event_category", "College symposiums")
    start_time = str(event_row.get("start_time", "09:00"))
    end_time = str(event_row.get("end_time", "17:00"))

    try:
        start_hr = max(0, min(23, int(start_time.split(":")[0])))
    except (ValueError, IndexError):
        start_hr = 9

    # Baseline (No Event) prediction at event start hour
    normal_pred = crowd_predictor.predict(
        hour=start_hr,
        historical_crowd_count=520,
        event_type="None",
        expected_visitors=0,
        nearby_traffic_level=0.32,
        historical_traffic_conditions=0.30,
    )

    # Event-Day prediction at event start hour
    event_traffic_ratio = min(0.96, 0.32 + (vehicles / 900.0) + (participants / 4500.0))
    event_pred = crowd_predictor.predict(
        hour=start_hr,
        historical_crowd_count=520,
        event_type=category,
        event_start_time=start_time,
        event_end_time=end_time,
        expected_visitors=participants,
        nearby_traffic_level=event_traffic_ratio,
        historical_traffic_conditions=0.30,
    )

    normal_crowd = normal_pred["estimated_crowd_count"]
    event_crowd = event_pred["estimated_crowd_count"]
    crowd_surge_pct = round(((event_crowd - normal_crowd) / max(1, normal_crowd)) * 100.0, 1)

    persons_per_gate_hr = int(round(participants / float(gates)))
    vehicles_per_gate_hr = int(round(vehicles / float(gates)))

    # Hourly comparison curve (08:00 to 20:00) for Normal vs Event Day chart
    normal_curve = {pt["hour"]: pt["predicted_crowd"] for pt in normal_pred["hourly_forecast_curve"]}
    event_curve = {pt["hour"]: pt["predicted_crowd"] for pt in event_pred["hourly_forecast_curve"]}
    comparison_series = []
    for h in range(7, 21):
        h_key = f"{h:02d}:00"
        comparison_series.append(
            {
                "hour": h_key,
                "normal_day_crowd": normal_curve.get(h_key, 420),
                "event_day_crowd": event_curve.get(h_key, 680),
            }
        )

    return {
        "normal_baseline_crowd": normal_crowd,
        "event_day_predicted_crowd": event_crowd,
        "crowd_surge_percentage": crowd_surge_pct,
        "normal_congestion_level": normal_pred["expected_congestion_level"],
        "event_congestion_level": event_pred["expected_congestion_level"],
        "event_congestion_color": event_pred["congestion_color"],
        "estimated_additional_delay_min": round((vehicles / float(gates)) * 0.065, 1),
        "per_gate_load": {
            "entry_gates_count": gates,
            "participants_per_gate": persons_per_gate_hr,
            "vehicles_per_gate": vehicles_per_gate_hr,
            "gate_pressure_status": (
                "High Gate Queue Risk — Open Overflow Gate"
                if persons_per_gate_hr > 550 or vehicles_per_gate_hr > 160
                else "Manageable Gate Load"
            ),
        },
        "hourly_comparison": comparison_series,
        "data_label": "Estimated Event Impact (ML Comparison vs. Normal Baseline)",
    }


def _validate_event_payload(data: Dict[str, Any]) -> tuple:
    name = str(data.get("event_name", "")).strip()
    location = str(data.get("event_location", "")).strip()
    category = str(data.get("event_category", "College symposiums")).strip()
    event_date = str(data.get("event_date", datetime.now().strftime("%Y-%m-%d"))).strip()
    start_time = str(data.get("start_time", "09:00")).strip()
    end_time = str(data.get("end_time", "17:00")).strip()

    if not name:
        return False, "Event name is required.", None
    if not location:
        return False, "Event location is required.", None
    if category not in SUPPORTED_EVENT_CATEGORIES:
        category = "Public gatherings"

    try:
        participants = max(1, int(data.get("expected_participants", 200)))
        gates = max(1, int(data.get("entry_gates_count", 2)))
        vehicles = max(0, int(data.get("expected_vehicles", 50)))
    except (TypeError, ValueError):
        return False, "Participants, entry gates, and expected vehicles must be valid integers.", None

    lat = data.get("latitude")
    lon = data.get("longitude")
    try:
        lat_f = float(lat) if lat not in (None, "") else 13.0108
        lon_f = float(lon) if lon not in (None, "") else 80.2354
    except (TypeError, ValueError):
        lat_f, lon_f = 13.0108, 80.2354

    cleaned = {
        "event_name": name,
        "event_location": location,
        "event_category": category,
        "event_date": event_date,
        "start_time": start_time,
        "end_time": end_time,
        "expected_participants": participants,
        "entry_gates_count": gates,
        "expected_vehicles": vehicles,
        "latitude": lat_f,
        "longitude": lon_f,
        "notes": str(data.get("notes", "")).strip(),
    }
    return True, None, cleaned


@event_bp.route("/api/events", methods=["GET"])
def list_events():
    """Return all scheduled events along with their Normal vs. Event-Day impact analysis."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events ORDER BY event_date DESC, start_time ASC, id DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    enriched = []
    for r in rows:
        r["impact_analysis"] = compute_event_impact_comparison(r)
        enriched.append(r)

    return (
        jsonify(
            {
                "status": "ok",
                "supported_categories": SUPPORTED_EVENT_CATEGORIES,
                "count": len(enriched),
                "events": enriched,
            }
        ),
        200,
    )


@event_bp.route("/api/events", methods=["POST"])
def create_event():
    """Create a new campus or public event and return its predicted traffic & crowd impact."""
    payload = request.get_json(silent=True) or {}
    ok, err, cleaned = _validate_event_payload(payload)
    if not ok:
        return jsonify({"status": "error", "error": err}), 400

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO events (
            event_name, event_location, event_category, event_date,
            start_time, end_time, expected_participants, entry_gates_count,
            expected_vehicles, latitude, longitude, notes, data_source, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'user_input', ?, ?)
        """,
        (
            cleaned["event_name"],
            cleaned["event_location"],
            cleaned["event_category"],
            cleaned["event_date"],
            cleaned["start_time"],
            cleaned["end_time"],
            cleaned["expected_participants"],
            cleaned["entry_gates_count"],
            cleaned["expected_vehicles"],
            cleaned["latitude"],
            cleaned["longitude"],
            cleaned["notes"],
            now_iso,
            now_iso,
        ),
    )
    event_id = cursor.lastrowid
    conn.commit()
    cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    created_row = dict(cursor.fetchone())
    conn.close()

    created_row["impact_analysis"] = compute_event_impact_comparison(created_row)
    return jsonify({"status": "ok", "event": created_row}), 201


@event_bp.route("/api/events/<int:event_id>", methods=["PUT"])
def update_event(event_id: int):
    """Update an existing event and recompute its impact analysis."""
    payload = request.get_json(silent=True) or {}
    ok, err, cleaned = _validate_event_payload(payload)
    if not ok:
        return jsonify({"status": "error", "error": err}), 400

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM events WHERE id = ?", (event_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({"status": "error", "error": f"Event ID {event_id} not found."}), 404

    cursor.execute(
        """
        UPDATE events
        SET event_name = ?, event_location = ?, event_category = ?, event_date = ?,
            start_time = ?, end_time = ?, expected_participants = ?, entry_gates_count = ?,
            expected_vehicles = ?, latitude = ?, longitude = ?, notes = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            cleaned["event_name"],
            cleaned["event_location"],
            cleaned["event_category"],
            cleaned["event_date"],
            cleaned["start_time"],
            cleaned["end_time"],
            cleaned["expected_participants"],
            cleaned["entry_gates_count"],
            cleaned["expected_vehicles"],
            cleaned["latitude"],
            cleaned["longitude"],
            cleaned["notes"],
            now_iso,
            event_id,
        ),
    )
    conn.commit()
    cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    updated_row = dict(cursor.fetchone())
    conn.close()

    updated_row["impact_analysis"] = compute_event_impact_comparison(updated_row)
    return jsonify({"status": "ok", "event": updated_row}), 200


@event_bp.route("/api/events/<int:event_id>", methods=["DELETE"])
def delete_event(event_id: int):
    """Delete an event by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM events WHERE id = ?", (event_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({"status": "error", "error": f"Event ID {event_id} not found."}), 404

    cursor.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "deleted_id": event_id}), 200

