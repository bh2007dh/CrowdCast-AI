"""
Geocoding Service (Reverse Geocoding & Manual Location Search via OpenStreetMap Nominatim).
Clearly reports data_source ('live_openstreetmap_nominatim' vs 'fallback_catalog_resolver').
"""
from datetime import datetime, timezone
from typing import Any, Dict, List
import requests

from config import Config

# Curated fallback reference locations for offline/sandboxed resilience
KNOWN_REFERENCE_LOCATIONS = [
    {
        "name": "Guindy Institutional & University Hub, Chennai",
        "city": "Chennai",
        "locality": "Guindy / Adyar",
        "area_name": "Anna University & IIT Madras Corridor",
        "formatted_address": "Sardar Patel Road, Guindy, Chennai, Tamil Nadu 600025, India",
        "latitude": 13.0108,
        "longitude": 80.2354,
    },
    {
        "name": "Bengaluru Electronic City & Tech Corridor",
        "city": "Bengaluru",
        "locality": "Electronic City Phase 1",
        "area_name": "Hosur Road IT & Campus Zone",
        "formatted_address": "Hosur Road, Electronic City, Bengaluru, Karnataka 560100, India",
        "latitude": 12.8452,
        "longitude": 77.6602,
    },
    {
        "name": "Coimbatore Peelamedu Educational Corridor",
        "city": "Coimbatore",
        "locality": "Peelamedu",
        "area_name": "Avinashi Road College & Hospital Belt",
        "formatted_address": "Avinashi Road, Peelamedu, Coimbatore, Tamil Nadu 641004, India",
        "latitude": 11.0247,
        "longitude": 77.0028,
    },
    {
        "name": "Hyderabad Gachibowli University & Financial District",
        "city": "Hyderabad",
        "locality": "Gachibowli",
        "area_name": "ISB & University of Hyderabad Approach",
        "formatted_address": "Old Mumbai Highway, Gachibowli, Hyderabad, Telangana 500032, India",
        "latitude": 17.4401,
        "longitude": 78.3489,
    },
    {
        "name": "New Delhi Hauz Khas Institutional Zone",
        "city": "New Delhi",
        "locality": "Hauz Khas",
        "area_name": "Outer Ring Road & IIT Delhi Corridor",
        "formatted_address": "Hauz Khas, South Delhi, New Delhi 110016, India",
        "latitude": 28.5459,
        "longitude": 77.1926,
    },
]


class GeocodingService:
    """Handles reverse-geocoding of coordinates and forward manual search."""

    def __init__(self) -> None:
        self.base_url = Config.NOMINATIM_BASE_URL.rstrip("/")
        self.headers = {
            "User-Agent": Config.NOMINATIM_USER_AGENT,
            "Accept-Language": "en",
        }

    def reverse_geocode(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Reverse-geocode coordinates into readable address, city, locality, and area."""
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            resp = requests.get(
                f"{self.base_url}/reverse",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "zoom": 17,
                },
                headers=self.headers,
                timeout=6.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                addr = data.get("address", {})
                city = (
                    addr.get("city")
                    or addr.get("town")
                    or addr.get("municipality")
                    or addr.get("county")
                    or addr.get("state_district")
                    or "Detected Urban Municipality"
                )
                locality = (
                    addr.get("suburb")
                    or addr.get("neighbourhood")
                    or addr.get("quarter")
                    or addr.get("village")
                    or addr.get("city_district")
                    or city
                )
                area_name = (
                    addr.get("amenity")
                    or addr.get("road")
                    or addr.get("commercial")
                    or addr.get("industrial")
                    or locality
                )
                formatted_address = data.get("display_name") or f"{area_name}, {locality}, {city}"
                return {
                    "latitude": round(float(latitude), 6),
                    "longitude": round(float(longitude), 6),
                    "city": city,
                    "locality": locality,
                    "area_name": area_name,
                    "formatted_address": formatted_address,
                    "osm_place_id": data.get("place_id"),
                    "data_source": "live_openstreetmap_nominatim",
                    "data_label": "Real Data (OpenStreetMap Nominatim Reverse Geocoding)",
                    "resolved_at": now_iso,
                }
        except requests.RequestException:
            pass

        # Fallback: match closest reference location or provide coordinate-labelled descriptor
        closest = min(
            KNOWN_REFERENCE_LOCATIONS,
            key=lambda loc: (loc["latitude"] - latitude) ** 2 + (loc["longitude"] - longitude) ** 2,
        )
        dist_deg = ((closest["latitude"] - latitude) ** 2 + (closest["longitude"] - longitude) ** 2) ** 0.5
        if dist_deg < 0.08:
            return {
                "latitude": round(float(latitude), 6),
                "longitude": round(float(longitude), 6),
                "city": closest["city"],
                "locality": closest["locality"],
                "area_name": closest["area_name"],
                "formatted_address": closest["formatted_address"],
                "data_source": "fallback_reference_geocoder",
                "data_label": "Fallback Reference Data (Offline Geocoder Match)",
                "resolved_at": now_iso,
            }

        return {
            "latitude": round(float(latitude), 6),
            "longitude": round(float(longitude), 6),
            "city": f"Zone ({latitude:.3f}, {longitude:.3f})",
            "locality": f"Sector {latitude:.3f}N, {longitude:.3f}E",
            "area_name": "Coordinate-Anchored Urban Sector",
            "formatted_address": f"Coordinates {latitude:.5f}, {longitude:.5f} (Reverse Geocoder Offline)",
            "data_source": "fallback_coordinate_resolver",
            "data_label": "Estimated Coordinate Sector (Nominatim Network Unreachable)",
            "resolved_at": now_iso,
        }

    def search_location(self, query: str) -> List[Dict[str, Any]]:
        """Forward geocode a user's manual text search query."""
        clean_q = (query or "").strip()
        if not clean_q:
            return []

        results: List[Dict[str, Any]] = []
        try:
            resp = requests.get(
                f"{self.base_url}/search",
                params={
                    "q": clean_q,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "limit": 5,
                },
                headers=self.headers,
                timeout=6.0,
            )
            if resp.status_code == 200:
                items = resp.json()
                for item in items:
                    addr = item.get("address", {})
                    lat = float(item.get("lat"))
                    lon = float(item.get("lon"))
                    city = (
                        addr.get("city")
                        or addr.get("town")
                        or addr.get("county")
                        or addr.get("state")
                        or "Urban Area"
                    )
                    locality = (
                        addr.get("suburb")
                        or addr.get("neighbourhood")
                        or addr.get("city_district")
                        or city
                    )
                    results.append(
                        {
                            "name": item.get("name") or item.get("display_name", "").split(",")[0],
                            "city": city,
                            "locality": locality,
                            "area_name": addr.get("road") or locality,
                            "formatted_address": item.get("display_name"),
                            "latitude": round(lat, 6),
                            "longitude": round(lon, 6),
                            "data_source": "live_openstreetmap_nominatim",
                        }
                    )
                if results:
                    return results
        except requests.RequestException:
            pass

        # Offline / fallback search across curated reference locations
        q_lower = clean_q.lower()
        for loc in KNOWN_REFERENCE_LOCATIONS:
            if (
                q_lower in loc["name"].lower()
                or q_lower in loc["city"].lower()
                or q_lower in loc["locality"].lower()
                or q_lower in loc["formatted_address"].lower()
            ):
                results.append({**loc, "data_source": "fallback_reference_catalog"})

        if not results:
            # Return the full reference catalog so manual selection always works offline
            results = [{**loc, "data_source": "fallback_reference_catalog"} for loc in KNOWN_REFERENCE_LOCATIONS]

        return results


geocoding_service = GeocodingService()

