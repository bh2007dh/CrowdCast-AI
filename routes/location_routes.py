"""
Location & Surrounding Area Recognition Routes.
Implements:
- POST /api/location/detect
- GET  /api/location/nearby
- GET  /api/location/search
- GET  /api/location/campuses
"""
from flask import Blueprint, jsonify, request
from database.db import get_db_connection
from services.geocoding_service import geocoding_service
from services.location_service import location_service, validate_coordinates, validate_radius
from services.nearby_places_service import nearby_places_service

location_bp = Blueprint("location_bp", __name__)


@location_bp.route("/api/location/detect", methods=["POST"])
def detect_location():
    """
    Validate coordinates obtained from browser Geolocation API or manual selection,
    reverse-geocode into a readable address, city, locality, and area, and return context.
    """
    payload = request.get_json(silent=True) or {}
    lat_raw = payload.get("latitude")
    lon_raw = payload.get("longitude")
    radius_raw = payload.get("radius_m", 1000)
    source_mode = str(payload.get("source_mode", "browser_geolocation")).strip()

    is_valid, err_msg, lat, lon = validate_coordinates(lat_raw, lon_raw)
    if not is_valid:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": err_msg,
                    "hint": "Provide valid WGS84 latitude (-90..90) and longitude (-180..180) or use manual search.",
                }
            ),
            400,
        )

    radius_m = validate_radius(radius_raw)
    try:
        result = location_service.process_location(
            latitude=lat,
            longitude=lon,
            radius_m=radius_m,
            source_mode=source_mode,
        )
        return jsonify({"status": "ok", "location": result}), 200
    except Exception as exc:
        return jsonify({"status": "error", "error": f"Location resolution failed: {exc}"}), 500


@location_bp.route("/api/location/nearby", methods=["GET"])
def get_nearby_area():
    """
    Identify nearby main/secondary roads, intersections, traffic junctions, colleges,
    bus stops, railway stations, hospitals, commercial areas, event venues, and parking.
    """
    lat_raw = request.args.get("lat") or request.args.get("latitude")
    lon_raw = request.args.get("lon") or request.args.get("longitude")
    radius_raw = request.args.get("radius_m") or request.args.get("radius", 1000)

    is_valid, err_msg, lat, lon = validate_coordinates(lat_raw, lon_raw)
    if not is_valid:
        return jsonify({"status": "error", "error": err_msg}), 400

    radius_m = validate_radius(radius_raw)
    try:
        surrounding = nearby_places_service.fetch_surrounding_area(
            latitude=lat,
            longitude=lon,
            radius_m=radius_m,
        )
        return jsonify({"status": "ok", "surrounding": surrounding}), 200
    except Exception as exc:
        return jsonify({"status": "error", "error": f"Failed to inspect surrounding area: {exc}"}), 500


@location_bp.route("/api/location/search", methods=["GET"])
def manual_location_search():
    """Manual location search fallback when browser GPS permission is denied or unavailable."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"status": "error", "error": "Search query parameter 'q' is required."}), 400

    results = geocoding_service.search_location(query)
    return (
        jsonify(
            {
                "status": "ok",
                "query": query,
                "count": len(results),
                "results": results,
            }
        ),
        200,
    )


@location_bp.route("/api/location/campuses", methods=["GET"])
def list_campuses():
    """Return registered campus reference locations and gates."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM locations WHERE is_campus = 1 ORDER BY id DESC")
    campuses = [dict(r) for r in cursor.fetchall()]
    cursor.execute("SELECT * FROM campus_gates ORDER BY id ASC")
    gates = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify({"status": "ok", "campuses": campuses, "gates": gates}), 200

