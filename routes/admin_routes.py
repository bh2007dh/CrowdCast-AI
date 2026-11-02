"""
Administration Module Routes (Section 10).
Supports:
- Admin authentication with securely hashed passwords (Werkzeug PBKDF2/Scrypt)
- Adding campus locations
- Registering and managing campus entry/exit gates
- Viewing previous predictions and managing demonstration datasets (retraining ML model, updating weights)
"""
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash

from database.db import get_db_connection, init_db
from models.congestion_engine import congestion_engine
from models.crowd_predictor import crowd_predictor
from services.location_service import validate_coordinates

admin_bp = Blueprint("admin_bp", __name__)


@admin_bp.route("/api/admin/login", methods=["POST"])
def admin_login():
    """Authenticate administrator against hashed password in SQLite users table."""
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))

    if not username or not password:
        return jsonify({"status": "error", "error": "Username and password are required."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"status": "error", "error": "Invalid administrator credentials."}), 401

    session["admin_logged_in"] = True
    session["admin_username"] = user["username"]
    return (
        jsonify(
            {
                "status": "ok",
                "authenticated": True,
                "username": user["username"],
                "role": user["role"],
            }
        ),
        200,
    )


@admin_bp.route("/api/admin/logout", methods=["POST"])
def admin_logout():
    session.pop("admin_logged_in", None)
    session.pop("admin_username", None)
    return jsonify({"status": "ok", "authenticated": False}), 200


@admin_bp.route("/api/admin/gates", methods=["GET"])
def list_campus_gates():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM campus_gates ORDER BY id ASC")
    gates = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify({"status": "ok", "gates": gates}), 200


@admin_bp.route("/api/admin/gates", methods=["POST"])
def register_campus_gate():
    """Register a new campus entry/exit gate for smart gate allocation."""
    payload = request.get_json(silent=True) or {}
    campus_name = str(payload.get("campus_name", "Main University Campus")).strip()
    gate_name = str(payload.get("gate_name", "")).strip()
    gate_type = str(payload.get("gate_type", "Entry/Exit")).strip()
    connected_road = str(payload.get("connected_road", "Perimeter Road")).strip()
    status = str(payload.get("status", "Open")).strip()

    if not gate_name:
        return jsonify({"status": "error", "error": "Gate name is required."}), 400

    ok, err, lat, lon = validate_coordinates(
        payload.get("latitude", 13.0108), payload.get("longitude", 80.2354)
    )
    if not ok:
        return jsonify({"status": "error", "error": err}), 400

    try:
        max_cap = max(50, int(payload.get("max_capacity_per_hour", 800)))
        lanes = max(1, int(payload.get("vehicle_lanes", 2)))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "error": "Capacity and lanes must be valid integers."}), 400

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO campus_gates (
            campus_name, gate_name, gate_type, latitude, longitude,
            max_capacity_per_hour, vehicle_lanes, connected_road, status, data_source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'admin_registered', ?)
        """,
        (campus_name, gate_name, gate_type, lat, lon, max_cap, lanes, connected_road, status, now_iso),
    )
    gate_id = cursor.lastrowid
    conn.commit()
    cursor.execute("SELECT * FROM campus_gates WHERE id = ?", (gate_id,))
    gate_row = dict(cursor.fetchone())
    conn.close()
    return jsonify({"status": "ok", "gate": gate_row}), 201


@admin_bp.route("/api/admin/gates/<int:gate_id>", methods=["DELETE"])
def delete_campus_gate(gate_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM campus_gates WHERE id = ?", (gate_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "deleted_gate_id": gate_id}), 200


@admin_bp.route("/api/admin/campuses", methods=["POST"])
def add_campus_location():
    """Register a new campus reference location."""
    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", "")).strip()
    city = str(payload.get("city", "")).strip()
    locality = str(payload.get("locality", "")).strip()
    address = str(payload.get("formatted_address", "")).strip()

    if not label:
        return jsonify({"status": "error", "error": "Campus label is required."}), 400

    ok, err, lat, lon = validate_coordinates(
        payload.get("latitude", 13.0108), payload.get("longitude", 80.2354)
    )
    if not ok:
        return jsonify({"status": "error", "error": err}), 400

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO locations (
            label, city, locality, area_name, formatted_address,
            latitude, longitude, radius_m, is_campus, data_source, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 2000, 1, 'admin_registered', ?)
        """,
        (label, city or "Urban Campus", locality or label, label, address or label, lat, lon, now_iso),
    )
    loc_id = cursor.lastrowid
    conn.commit()
    cursor.execute("SELECT * FROM locations WHERE id = ?", (loc_id,))
    loc_row = dict(cursor.fetchone())
    conn.close()
    return jsonify({"status": "ok", "campus": loc_row}), 201


@admin_bp.route("/api/admin/weights", methods=["POST"])
def update_fusion_weights():
    """Update the Crowd-Traffic Fusion Engine weights at runtime."""
    payload = request.get_json(silent=True) or {}
    try:
        new_weights = {
            "traffic_congestion": float(payload.get("traffic_congestion", 0.35)),
            "expected_crowd": float(payload.get("expected_crowd", 0.30)),
            "historical_patterns": float(payload.get("historical_patterns", 0.20)),
            "event_impact": float(payload.get("event_impact", 0.15)),
        }
    except (TypeError, ValueError):
        return jsonify({"status": "error", "error": "Weights must be valid numbers."}), 400

    if any(v < 0 for v in new_weights.values()) or sum(new_weights.values()) <= 0:
        return jsonify({"status": "error", "error": "Weights must be non-negative and sum to > 0."}), 400

    congestion_engine.weights = new_weights
    congestion_engine._normalize_weights()
    return jsonify({"status": "ok", "weights": congestion_engine.weights}), 200


@admin_bp.route("/api/admin/dataset/retrain", methods=["POST"])
def retrain_ml_dataset():
    """Regenerate the synthetic demonstration dataset and retrain the Random Forest model."""
    meta = crowd_predictor.reload_model()
    init_db()
    return jsonify({"status": "ok", "message": "Synthetic dataset regenerated and Random Forest model retrained.", "metadata": meta}), 200

