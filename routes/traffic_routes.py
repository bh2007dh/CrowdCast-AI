"""
Real-Time Traffic & Incident Routes.
Implements:
- GET /api/traffic/current
- GET /api/traffic/incidents
"""
from flask import Blueprint, jsonify, request
from services.location_service import validate_coordinates, validate_radius
from services.nearby_places_service import nearby_places_service
from services.traffic_service import traffic_service

traffic_bp = Blueprint("traffic_bp", __name__)


@traffic_bp.route("/api/traffic/current", methods=["GET"])
def get_current_traffic():
    """
    Retrieve current traffic flow speed, free-flow speed, delay, road congestion classification
    (Green, Yellow, Orange, Red), last updated time, and provider availability status.
    """
    lat_raw = request.args.get("lat") or request.args.get("latitude")
    lon_raw = request.args.get("lon") or request.args.get("longitude")
    radius_raw = request.args.get("radius_m") or request.args.get("radius", 1000)

    is_valid, err_msg, lat, lon = validate_coordinates(lat_raw, lon_raw)
    if not is_valid:
        return jsonify({"status": "error", "error": err_msg}), 400

    radius_m = validate_radius(radius_raw)
    try:
        surrounding = nearby_places_service.fetch_surrounding_area(lat, lon, radius_m=radius_m)
        roads = surrounding.get("roads", [])
        traffic_data = traffic_service.fetch_current_traffic(lat, lon, nearby_roads=roads)
        return (
            jsonify(
                {
                    "status": "ok",
                    "coordinates": {"latitude": lat, "longitude": longitude if (longitude := lon) else lon},
                    "traffic": traffic_data,
                }
            ),
            200,
        )
    except Exception as exc:
        return jsonify({"status": "error", "error": f"Traffic retrieval error: {exc}"}), 500


@traffic_bp.route("/api/traffic/incidents", methods=["GET"])
def get_traffic_incidents():
    """Retrieve active traffic incidents or clearly labelled demonstration sample incidents."""
    lat_raw = request.args.get("lat") or request.args.get("latitude")
    lon_raw = request.args.get("lon") or request.args.get("longitude")
    radius_raw = request.args.get("radius_m") or request.args.get("radius", 2000)

    is_valid, err_msg, lat, lon = validate_coordinates(lat_raw, lon_raw)
    if not is_valid:
        return jsonify({"status": "error", "error": err_msg}), 400

    radius_m = validate_radius(radius_raw)
    try:
        incidents_data = traffic_service.fetch_incidents(lat, lon, radius_m=radius_m)
        return jsonify({"status": "ok", "incidents_data": incidents_data}), 200
    except Exception as exc:
        return jsonify({"status": "error", "error": f"Traffic incident retrieval error: {exc}"}), 500

