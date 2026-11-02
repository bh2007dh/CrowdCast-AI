"""
Location Validation, Distance Calculation, and Privacy-Conscious Area Persistence Service.
"""
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from config import Config
from database.db import get_db_connection
from services.geocoding_service import geocoding_service


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in metres between two WGS84 coordinates."""
    r_earth = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(r_earth * c, 1)


def validate_coordinates(lat: Any, lon: Any) -> Tuple[bool, Optional[str], Optional[float], Optional[float]]:
    """Validate latitude and longitude values."""
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return False, "Coordinates must be valid numeric values.", None, None

    if math.isnan(lat_f) or math.isnan(lon_f) or math.isinf(lat_f) or math.isinf(lon_f):
        return False, "Coordinates cannot be NaN or infinite.", None, None

    if not (-90.0 <= lat_f <= 90.0):
        return False, "Latitude must be between -90 and 90 degrees.", None, None

    if not (-180.0 <= lon_f <= 180.0):
        return False, "Longitude must be between -180 and 180 degrees.", None, None

    if abs(lat_f) < 1e-6 and abs(lon_f) < 1e-6:
        return False, "Null Island (0.0, 0.0) is not a valid user land location.", None, None

    return True, None, round(lat_f, 6), round(lon_f, 6)


def validate_radius(radius: Any) -> int:
    """Validate and normalize search radius in metres (500, 1000, 2000, 5000, 10000)."""
    try:
        r_int = int(float(radius))
    except (TypeError, ValueError):
        return Config.DEFAULT_RADIUS

    if r_int in Config.ALLOWED_RADII:
        return r_int
    # Snap to nearest allowed radius
    return min(Config.ALLOWED_RADII, key=lambda allowed: abs(allowed - r_int))


class LocationService:
    """Manages active location recognition, reverse geocoding, and privacy-safe persistence."""

    def process_location(
        self,
        latitude: float,
        longitude: float,
        radius_m: int = 1000,
        source_mode: str = "browser_geolocation",
    ) -> Dict[str, Any]:
        """
        Reverse-geocode coordinates and update the current active area context in SQLite
        without accumulating unnecessary raw user tracking history.
        """
        geo = geocoding_service.reverse_geocode(latitude, longitude)
        valid_r = validate_radius(radius_m)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Privacy rule: round persisted coordinates to ~110m grid (3 decimal places) for non-campus records
        # and update existing session record rather than storing an endless trail of user movements.
        stored_lat = round(latitude, 3)
        stored_lon = round(longitude, 3)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM locations WHERE is_campus = 0 ORDER BY id DESC LIMIT 1"
        )
        existing = cursor.fetchone()
        if existing:
            loc_id = existing["id"]
            cursor.execute(
                """
                UPDATE locations
                SET label = ?, city = ?, locality = ?, area_name = ?,
                    formatted_address = ?, latitude = ?, longitude = ?,
                    radius_m = ?, data_source = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    f"{geo['locality']}, {geo['city']}",
                    geo["city"],
                    geo["locality"],
                    geo["area_name"],
                    geo["formatted_address"],
                    stored_lat,
                    stored_lon,
                    valid_r,
                    f"{source_mode}:{geo['data_source']}",
                    now_iso,
                    loc_id,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO locations (
                    label, city, locality, area_name, formatted_address,
                    latitude, longitude, radius_m, is_campus, data_source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    f"{geo['locality']}, {geo['city']}",
                    geo["city"],
                    geo["locality"],
                    geo["area_name"],
                    geo["formatted_address"],
                    stored_lat,
                    stored_lon,
                    valid_r,
                    f"{source_mode}:{geo['data_source']}",
                    now_iso,
                ),
            )
            loc_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return {
            "location_id": loc_id,
            "latitude": latitude,
            "longitude": longitude,
            "radius_m": valid_r,
            "allowed_radii": Config.ALLOWED_RADII,
            "city": geo["city"],
            "locality": geo["locality"],
            "area_name": geo["area_name"],
            "formatted_address": geo["formatted_address"],
            "acquisition_mode": source_mode,
            "geocoding_source": geo["data_source"],
            "data_label": geo["data_label"],
            "privacy_note": "Precise GPS coordinates are used in-memory for map centering; database stores only coarse area context.",
            "timestamp": now_iso,
        }


location_service = LocationService()

