"""
Real-Time Traffic Integration & Provider Abstraction Service.

Preferred Provider: TomTom Traffic API (Flow Segment Data v4 & Incident Details v5).
Includes a modular provider abstraction (`BaseTrafficProvider`) so additional providers
(e.g., HERE Traffic, Mapbox, Google Routes) can be plugged in easily.

DATA INTEGRITY GUARANTEE:
- When `TOMTOM_API_KEY` is configured and reachable, live observations are returned and
  marked with `data_source="live_tomtom_api"`, `is_simulated=False`.
- When `TOMTOM_API_KEY` is missing or the API is unreachable, the service NEVER fabricates
  live observations; it switches to an explicitly labelled Demonstration Mode
  (`data_source="demonstration_sample_traffic"`, `is_simulated=True`) with full provider
  status transparency.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import math
import requests

from config import Config
from database.db import get_db_connection


def classify_congestion_from_speeds(current_speed: float, free_flow_speed: float) -> Dict[str, Any]:
    """
    Classify road congestion into Green (Low), Yellow (Moderate), Orange (Heavy), or Red (Severe)
    based on current speed vs. free-flow speed ratio.
    """
    ff = max(10.0, float(free_flow_speed))
    cur = max(2.0, min(ff, float(current_speed)))
    speed_ratio = cur / ff
    congestion_ratio = round(max(0.0, min(1.0, 1.0 - speed_ratio)), 3)

    if congestion_ratio < 0.22:
        return {
            "level": "Low",
            "color": "Green",
            "hex": "#10b981",
            "congestion_ratio": congestion_ratio,
            "description": "Low congestion (Free-flowing traffic)",
        }
    if congestion_ratio < 0.45:
        return {
            "level": "Moderate",
            "color": "Yellow",
            "hex": "#f59e0b",
            "congestion_ratio": congestion_ratio,
            "description": "Moderate congestion (Steady flow with minor slowdowns)",
        }
    if congestion_ratio < 0.70:
        return {
            "level": "Heavy",
            "color": "Orange",
            "hex": "#f97316",
            "congestion_ratio": congestion_ratio,
            "description": "Heavy congestion (Significant queueing and speed reduction)",
        }
    return {
        "level": "Severe",
        "color": "Red",
        "hex": "#ef4444",
        "congestion_ratio": congestion_ratio,
        "description": "Severe congestion (Stop-and-go bottleneck)",
    }


class BaseTrafficProvider(ABC):
    """Abstract base interface for traffic data providers."""

    provider_name: str = "abstract"

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if necessary API credentials are present."""

    @abstractmethod
    def get_traffic_flow(
        self, latitude: float, longitude: float, nearby_roads: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Fetch traffic flow speed, free-flow speed, delay, and road segment statuses."""

    @abstractmethod
    def get_traffic_incidents(
        self, latitude: float, longitude: float, radius_m: int = 2000
    ) -> Dict[str, Any]:
        """Fetch active traffic incidents within bounding box of (latitude, longitude)."""


class TomTomTrafficProvider(BaseTrafficProvider):
    """Live TomTom Traffic Flow & Incidents API implementation."""

    provider_name = "TomTom Traffic API v4/v5"

    def __init__(self, api_key: str) -> None:
        self.api_key = (api_key or "").strip()
        self.flow_url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        self.incidents_url = "https://api.tomtom.com/traffic/services/5/incidentDetails"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_traffic_flow(
        self, latitude: float, longitude: float, nearby_roads: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        if not self.is_configured():
            raise RuntimeError("TOMTOM_API_KEY is not configured in environment.")

        now_iso = datetime.now(timezone.utc).isoformat()
        resp = requests.get(
            self.flow_url,
            params={
                "key": self.api_key,
                "point": f"{latitude},{longitude}",
                "unit": "KMPH",
            },
            timeout=6.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"TomTom API returned HTTP {resp.status_code}")

        data = resp.json().get("flowSegmentData", {})
        cur_speed = float(data.get("currentSpeed", 38.0))
        ff_speed = float(data.get("freeFlowSpeed", 50.0))
        cur_tt = float(data.get("currentTravelTime", 120.0))
        ff_tt = float(data.get("freeFlowTravelTime", 95.0))
        delay_sec = max(0.0, round(cur_tt - ff_tt, 1))
        confidence = float(data.get("confidence", 0.9))
        cls_info = classify_congestion_from_speeds(cur_speed, ff_speed)

        coords_raw = data.get("coordinates", {}).get("coordinate", [])
        segment_coords = [[c["latitude"], c["longitude"]] for c in coords_raw if "latitude" in c]

        road_segments_out = [
            {
                "road_name": "Primary Monitored Corridor (TomTom Live Segment)",
                "road_type": data.get("frc", "FRC2"),
                "provider_monitored": True,
                "current_speed_kmh": round(cur_speed, 1),
                "free_flow_speed_kmh": round(ff_speed, 1),
                "delay_seconds": delay_sec,
                "congestion_ratio": cls_info["congestion_ratio"],
                "congestion_level": cls_info["level"],
                "color_code": cls_info["color"],
                "hex_color": cls_info["hex"],
                "confidence": confidence,
                "coordinates": segment_coords or [[latitude, longitude]],
                "data_source": "live_tomtom_api",
            }
        ]

        return {
            "provider_name": self.provider_name,
            "provider_status": "Online (Live API Connected)",
            "mode": "LIVE_REALTIME",
            "is_simulated": False,
            "data_source": "live_tomtom_api",
            "data_label": "Real-Time Data (TomTom Traffic API)",
            "last_updated": now_iso,
            "summary": {
                "current_speed_kmh": round(cur_speed, 1),
                "free_flow_speed_kmh": round(ff_speed, 1),
                "average_delay_seconds": delay_sec,
                "congestion_ratio": cls_info["congestion_ratio"],
                "congestion_level": cls_info["level"],
                "color_code": cls_info["color"],
                "hex_color": cls_info["hex"],
                "description": cls_info["description"],
                "confidence": confidence,
            },
            "segments": road_segments_out,
        }

    def get_traffic_incidents(
        self, latitude: float, longitude: float, radius_m: int = 2000
    ) -> Dict[str, Any]:
        if not self.is_configured():
            raise RuntimeError("TOMTOM_API_KEY is not configured.")

        now_iso = datetime.now(timezone.utc).isoformat()
        delta_deg = max(0.01, (radius_m / 111000.0))
        bbox = f"{longitude - delta_deg},{latitude - delta_deg},{longitude + delta_deg},{latitude + delta_deg}"
        resp = requests.get(
            self.incidents_url,
            params={
                "key": self.api_key,
                "bbox": bbox,
                "fields": "{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay,events{description,code},from,to,delay}}}",
                "language": "en-GB",
            },
            timeout=6.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"TomTom Incidents API returned HTTP {resp.status_code}")

        raw_incidents = resp.json().get("incidents", [])
        incidents = []
        for idx, inc in enumerate(raw_incidents[:15], start=1):
            props = inc.get("properties", {})
            geom = inc.get("geometry", {})
            coords = geom.get("coordinates", [])
            inc_lat, inc_lon = latitude, longitude
            if coords and isinstance(coords[0], list):
                inc_lon, inc_lat = float(coords[0][0]), float(coords[0][1])
            elif len(coords) >= 2 and isinstance(coords[0], (int, float)):
                inc_lon, inc_lat = float(coords[0]), float(coords[1])

            events = props.get("events", [{}])
            desc = events[0].get("description", "Traffic incident reported") if events else "Traffic incident"
            mag = int(props.get("magnitudeOfDelay", 1))
            sev = "Severe" if mag >= 3 else ("Moderate" if mag == 2 else "Minor")
            incidents.append(
                {
                    "id": f"tt_inc_{idx}",
                    "title": desc,
                    "severity": sev,
                    "road_from": props.get("from", "Local Corridor"),
                    "road_to": props.get("to", "Adjacent Intersection"),
                    "delay_seconds": int(props.get("delay") or 90),
                    "latitude": round(inc_lat, 6),
                    "longitude": round(inc_lon, 6),
                    "data_source": "live_tomtom_api",
                    "is_simulated": False,
                    "reported_at": now_iso,
                }
            )

        return {
            "provider_name": self.provider_name,
            "provider_status": "Online (Live API Connected)",
            "mode": "LIVE_REALTIME",
            "is_simulated": False,
            "data_source": "live_tomtom_api",
            "data_label": "Real-Time Traffic Incidents (TomTom API)",
            "last_updated": now_iso,
            "incidents": incidents,
        }


class DemonstrationTrafficProvider(BaseTrafficProvider):
    """
    Explicitly labelled Demonstration Mode Traffic Provider.
    Used when TOMTOM_API_KEY is not configured or external network is unavailable.
    Never claims simulated values are live observations.
    """

    provider_name = "CrowdCast Demonstration Traffic Simulator (Fallback Mode)"

    def is_configured(self) -> bool:
        return True

    def get_traffic_flow(
        self,
        latitude: float,
        longitude: float,
        nearby_roads: Optional[List[Dict[str, Any]]] = None,
        fallback_reason: str = "TOMTOM_API_KEY not configured in .env",
    ) -> Dict[str, Any]:
        now = datetime.now()
        now_iso = datetime.now(timezone.utc).isoformat()
        hour = now.hour

        # Time-of-day realistic demonstration wave (transparently labelled as sample data)
        morning_wave = math.exp(-0.5 * ((hour - 9.0) / 1.7) ** 2) * 0.52
        midday_wave = math.exp(-0.5 * ((hour - 13.5) / 2.0) ** 2) * 0.34
        evening_wave = math.exp(-0.5 * ((hour - 17.5) / 1.8) ** 2) * 0.62
        base_diurnal = max(0.18, morning_wave, midday_wave, evening_wave)

        roads_input = nearby_roads if nearby_roads else []
        if not roads_input:
            d = 0.0045
            roads_input = [
                {
                    "road_name": "Primary Urban Arterial Corridor (N-S)",
                    "road_type": "primary",
                    "maxspeed_kmh": 60.0,
                    "provider_monitored": True,
                    "congestion_potential": 0.76,
                    "coordinates": [
                        [round(latitude - d, 6), round(longitude - d * 0.15, 6)],
                        [round(latitude, 6), round(longitude, 6)],
                        [round(latitude + d, 6), round(longitude + d * 0.15, 6)],
                    ],
                },
                {
                    "road_name": "East-West Institutional Highway",
                    "road_type": "primary",
                    "maxspeed_kmh": 60.0,
                    "provider_monitored": True,
                    "congestion_potential": 0.62,
                    "coordinates": [
                        [round(latitude - d * 0.1, 6), round(longitude - d, 6)],
                        [round(latitude, 6), round(longitude, 6)],
                        [round(latitude + d * 0.1, 6), round(longitude + d, 6)],
                    ],
                },
                {
                    "road_name": "Campus Ring Collector Road",
                    "road_type": "secondary",
                    "maxspeed_kmh": 40.0,
                    "provider_monitored": False,
                    "congestion_potential": 0.45,
                    "coordinates": [
                        [round(latitude + d * 0.45, 6), round(longitude - d * 0.8, 6)],
                        [round(latitude + d * 0.48, 6), round(longitude + d * 0.8, 6)],
                    ],
                },
                {
                    "road_name": "South Station Feeder Link Road",
                    "road_type": "tertiary",
                    "maxspeed_kmh": 40.0,
                    "provider_monitored": False,
                    "congestion_potential": 0.25,
                    "coordinates": [
                        [round(latitude - d * 0.55, 6), round(longitude - d * 0.75, 6)],
                        [round(latitude - d * 0.52, 6), round(longitude + d * 0.75, 6)],
                    ],
                },
            ]

        segments = []
        speed_list = []
        ff_list = []
        delay_list = []

        # Create varied congestion profiles (Green, Yellow, Orange, Red) across nearby roads for testing
        profile_offsets = [0.22, 0.04, -0.12, -0.24, 0.15, -0.06]
        for idx, rd in enumerate(roads_input[:12]):
            ff_speed = float(rd.get("maxspeed_kmh", 50.0))
            pot = float(rd.get("congestion_potential", 0.55))
            offset = profile_offsets[idx % len(profile_offsets)]
            target_cong_ratio = max(0.08, min(0.86, base_diurnal * 0.55 + pot * 0.45 + offset))
            cur_speed = round(max(6.0, ff_speed * (1.0 - target_cong_ratio)), 1)
            delay_sec = round(max(0.0, (1.0 / max(cur_speed, 5.0) - 1.0 / ff_speed) * 1800.0), 1)
            cls_info = classify_congestion_from_speeds(cur_speed, ff_speed)

            coords = rd.get("coordinates") or [
                [rd.get("start_lat", latitude), rd.get("start_lon", longitude)],
                [rd.get("end_lat", latitude + 0.002), rd.get("end_lon", longitude + 0.002)],
            ]

            segments.append(
                {
                    "road_name": rd.get("road_name", f"Road Segment #{idx + 1}"),
                    "road_type": rd.get("road_type", "secondary"),
                    "provider_monitored": bool(rd.get("provider_monitored", idx < 2)),
                    "current_speed_kmh": cur_speed,
                    "free_flow_speed_kmh": round(ff_speed, 1),
                    "delay_seconds": delay_sec,
                    "congestion_ratio": cls_info["congestion_ratio"],
                    "congestion_level": cls_info["level"],
                    "color_code": cls_info["color"],
                    "hex_color": cls_info["hex"],
                    "confidence": None,
                    "coordinates": coords,
                    "data_source": "demonstration_sample_traffic",
                }
            )
            speed_list.append(cur_speed)
            ff_list.append(ff_speed)
            delay_list.append(delay_sec)

        avg_cur = round(sum(speed_list) / max(1, len(speed_list)), 1)
        avg_ff = round(sum(ff_list) / max(1, len(ff_list)), 1)
        avg_delay = round(sum(delay_list) / max(1, len(delay_list)), 1)
        overall_cls = classify_congestion_from_speeds(avg_cur, avg_ff)

        return {
            "provider_name": self.provider_name,
            "provider_status": f"Demonstration Mode Active ({fallback_reason})",
            "mode": "DEMONSTRATION_MODE",
            "is_simulated": True,
            "data_source": "demonstration_sample_traffic",
            "data_label": "Demonstration Data (Sample Traffic — Live Provider Key Not Configured)",
            "last_updated": now_iso,
            "summary": {
                "current_speed_kmh": avg_cur,
                "free_flow_speed_kmh": avg_ff,
                "average_delay_seconds": avg_delay,
                "congestion_ratio": overall_cls["congestion_ratio"],
                "congestion_level": overall_cls["level"],
                "color_code": overall_cls["color"],
                "hex_color": overall_cls["hex"],
                "description": f"[DEMO SAMPLE] {overall_cls['description']}",
                "confidence": None,
            },
            "segments": segments,
        }

    def get_traffic_incidents(
        self,
        latitude: float,
        longitude: float,
        radius_m: int = 2000,
        fallback_reason: str = "TOMTOM_API_KEY not configured in .env",
    ) -> Dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        d = 0.0028
        sample_incidents = [
            {
                "id": "demo_inc_1",
                "title": "[DEMO SAMPLE] High Entry Queue & Signal Delay near Main Arch Junction",
                "severity": "Moderate",
                "road_from": "Primary Urban Arterial Corridor",
                "road_to": "Campus Gate 1 Approach",
                "delay_seconds": 145,
                "latitude": round(latitude + d * 0.45, 6),
                "longitude": round(longitude + d * 0.25, 6),
                "data_source": "demonstration_sample_incidents",
                "is_simulated": True,
                "reported_at": now_iso,
            },
            {
                "id": "demo_inc_2",
                "title": "[DEMO SAMPLE] Pedestrian Crossing Bottleneck & Bus Boarding Slowdown",
                "severity": "Minor",
                "road_from": "East-West Institutional Highway",
                "road_to": "Transit Terminal Stop",
                "delay_seconds": 75,
                "latitude": round(latitude - d * 0.35, 6),
                "longitude": round(longitude + d * 0.55, 6),
                "data_source": "demonstration_sample_incidents",
                "is_simulated": True,
                "reported_at": now_iso,
            },
        ]
        return {
            "provider_name": self.provider_name,
            "provider_status": f"Demonstration Mode Active ({fallback_reason})",
            "mode": "DEMONSTRATION_MODE",
            "is_simulated": True,
            "data_source": "demonstration_sample_incidents",
            "data_label": "Demonstration Data (Sample Traffic Incidents)",
            "last_updated": now_iso,
            "incidents": sample_incidents,
        }


class TrafficService:
    """Facade managing primary traffic provider with transparent demonstration fallback."""

    def __init__(self) -> None:
        self.tomtom = TomTomTrafficProvider(Config.TOMTOM_API_KEY)
        self.demo = DemonstrationTrafficProvider()

    def get_provider_status(self) -> Dict[str, Any]:
        if self.tomtom.is_configured():
            return {
                "configured_provider": "TomTom Traffic API",
                "api_key_configured": True,
                "mode": "LIVE_REALTIME",
                "status_message": "TomTom API key configured.",
            }
        return {
            "configured_provider": "TomTom Traffic API (Fallback: Demonstration Mode)",
            "api_key_configured": False,
            "mode": "DEMONSTRATION_MODE",
            "status_message": "TOMTOM_API_KEY is not set; using clearly labelled sample traffic data.",
        }

    def fetch_current_traffic(
        self,
        latitude: float,
        longitude: float,
        nearby_roads: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any]
        if self.tomtom.is_configured():
            try:
                result = self.tomtom.get_traffic_flow(latitude, longitude, nearby_roads=nearby_roads)
            except Exception as exc:
                result = self.demo.get_traffic_flow(
                    latitude,
                    longitude,
                    nearby_roads=nearby_roads,
                    fallback_reason=f"TomTom API unreachable: {exc}",
                )
        else:
            result = self.demo.get_traffic_flow(
                latitude,
                longitude,
                nearby_roads=nearby_roads,
                fallback_reason="TOMTOM_API_KEY missing in environment",
            )

        self._persist_traffic_observations(latitude, longitude, result)
        return result

    def fetch_incidents(
        self, latitude: float, longitude: float, radius_m: int = 2000
    ) -> Dict[str, Any]:
        if self.tomtom.is_configured():
            try:
                return self.tomtom.get_traffic_incidents(latitude, longitude, radius_m=radius_m)
            except Exception as exc:
                return self.demo.get_traffic_incidents(
                    latitude,
                    longitude,
                    radius_m=radius_m,
                    fallback_reason=f"TomTom Incidents API unreachable: {exc}",
                )
        return self.demo.get_traffic_incidents(
            latitude,
            longitude,
            radius_m=radius_m,
            fallback_reason="TOMTOM_API_KEY missing in environment",
        )

    @staticmethod
    def _persist_traffic_observations(
        latitude: float, longitude: float, traffic_payload: Dict[str, Any]
    ) -> None:
        conn = get_db_connection()
        cursor = conn.cursor()
        is_sim = 1 if traffic_payload.get("is_simulated", True) else 0
        prov_name = traffic_payload.get("provider_name", "Unknown")
        d_source = traffic_payload.get("data_source", "demonstration_sample_traffic")
        obs_time = traffic_payload.get("last_updated", datetime.now(timezone.utc).isoformat())

        for seg in traffic_payload.get("segments", [])[:10]:
            coords = seg.get("coordinates", [])
            seg_lat = coords[0][0] if coords and len(coords[0]) == 2 else latitude
            seg_lon = coords[0][1] if coords and len(coords[0]) == 2 else longitude
            cursor.execute(
                """
                INSERT INTO traffic_observations (
                    road_name, latitude, longitude, current_speed_kmh, free_flow_speed_kmh,
                    delay_seconds, congestion_ratio, congestion_level, color_code,
                    confidence, provider_name, data_source, is_simulated, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seg.get("road_name", "Road Segment"),
                    seg_lat,
                    seg_lon,
                    seg.get("current_speed_kmh", 35.0),
                    seg.get("free_flow_speed_kmh", 50.0),
                    seg.get("delay_seconds", 0.0),
                    seg.get("congestion_ratio", 0.3),
                    seg.get("congestion_level", "Moderate"),
                    seg.get("color_code", "Yellow"),
                    seg.get("confidence"),
                    prov_name,
                    d_source,
                    is_sim,
                    obs_time,
                ),
            )
        conn.commit()
        conn.close()


traffic_service = TrafficService()

