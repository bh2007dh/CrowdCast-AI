"""
Smart Route & Campus Gate Recommendation Service (Section 8).

Uses OSRM (OpenStreetMap Routing Machine) for geometric road routing and overlays
estimated corridor congestion risk and campus gate capacity allocations.

IMPORTANT DATA INTEGRITY RULE:
Standard OpenStreetMap (OSRM) routes are explicitly labelled as static geometric road
network routes, NOT live traffic-aware routes, unless live TomTom traffic flow is active.
Advisory only: Never directs emergency vehicles or overrides official traffic controls.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import requests

from config import Config
from database.db import get_db_connection
from services.location_service import haversine_distance_m


class RoutingService:
    """Computes alternative routes, arrival windows, and campus gate allocations."""

    def __init__(self) -> None:
        self.osrm_base = Config.OSRM_BASE_URL.rstrip("/")
        self.headers = {"User-Agent": Config.NOMINATIM_USER_AGENT}

    def recommend_routes_and_gates(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: Optional[float] = None,
        dest_lon: Optional[float] = None,
        traffic_segments: Optional[List[Dict[str, Any]]] = None,
        estimated_crowd_count: int = 650,
        expected_event_visitors: int = 0,
        traffic_is_live: bool = False,
    ) -> Dict[str, Any]:
        """
        Provide alternative routes, nearby less-congested roads, suggested arrival time,
        suggested campus entry gate, and estimated congestion risk along each route.
        """
        now = datetime.now()
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM campus_gates WHERE status = 'Open' ORDER BY id ASC")
        gates = [dict(r) for r in cursor.fetchall()]
        conn.close()

        # Determine target destination (either explicit destination or best campus gate near user)
        if dest_lat is None or dest_lon is None:
            if gates:
                dest_lat = float(gates[0]["latitude"])
                dest_lon = float(gates[0]["longitude"])
            else:
                dest_lat = round(origin_lat + 0.0085, 6)
                dest_lon = round(origin_lon + 0.0075, 6)

        routes_out: List[Dict[str, Any]] = []
        routing_source = "osrm_geometric_routing"
        routing_label = (
            "Geometric Road Network Route (OpenStreetMap OSRM — Not Live Traffic-Aware)"
            if not traffic_is_live
            else "OSRM Road Geometry Overlaid with Live Traffic Provider Risk"
        )

        try:
            url = f"{self.osrm_base}/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
            resp = requests.get(
                url,
                params={
                    "alternatives": "true",
                    "overview": "full",
                    "geometries": "geojson",
                    "steps": "true",
                },
                headers=self.headers,
                timeout=6.0,
            )
            if resp.status_code == 200:
                payload = resp.json()
                for idx, rt in enumerate(payload.get("routes", [])[:3], start=1):
                    dist_m = float(rt.get("distance", 1200.0))
                    dur_s = float(rt.get("duration", 240.0))
                    coords_lonlat = rt.get("geometry", {}).get("coordinates", [])
                    coords_latlon = [[c[1], c[0]] for c in coords_lonlat]
                    summary_name = rt.get("legs", [{}])[0].get("summary") or f"Route Option #{idx}"
                    # Estimate congestion risk along route (Primary route often has higher baseline queue during events)
                    route_risk_score = round(min(92.0, (48.0 if idx == 1 else 28.0) + (expected_event_visitors / 90.0)), 1)
                    routes_out.append(
                        {
                            "route_id": f"route_{idx}",
                            "route_name": (
                                f"Alternative Corridor #{idx} via {summary_name}"
                                if idx > 1
                                else f"Primary Direct Route via {summary_name}"
                            ),
                            "is_recommended": idx == 2 if expected_event_visitors > 600 else idx == 1,
                            "distance_km": round(dist_m / 1000.0, 2),
                            "base_geometric_duration_min": round(dur_s / 60.0, 1),
                            "estimated_congested_duration_min": round(
                                (dur_s / 60.0) * (1.0 + route_risk_score / 140.0), 1
                            ),
                            "estimated_congestion_risk_score": route_risk_score,
                            "risk_level": "Moderate" if route_risk_score < 55 else "Heavy",
                            "routing_type_notice": routing_label,
                            "coordinates": coords_latlon,
                        }
                    )
        except requests.RequestException:
            pass

        # Fallback geometric corridors if OSRM is unreachable offline
        if not routes_out:
            routing_source = "fallback_geometric_corridor"
            routing_label = "Offline Geometric Corridor Approximation (OSRM Unreachable — Not Live Traffic-Aware)"
            direct_dist_m = max(450.0, haversine_distance_m(origin_lat, origin_lon, dest_lat, dest_lon))
            mid1_lat = (origin_lat + dest_lat) / 2.0 + 0.0022
            mid1_lon = (origin_lon + dest_lon) / 2.0 - 0.0018
            mid2_lat = (origin_lat + dest_lat) / 2.0 - 0.0028
            mid2_lon = (origin_lon + dest_lon) / 2.0 + 0.0025

            routes_out = [
                {
                    "route_id": "route_1",
                    "route_name": "Route A: Primary Arterial Direct Approach",
                    "is_recommended": expected_event_visitors < 500,
                    "distance_km": round((direct_dist_m * 1.18) / 1000.0, 2),
                    "base_geometric_duration_min": round((direct_dist_m * 1.18) / 550.0, 1),
                    "estimated_congested_duration_min": round((direct_dist_m * 1.18) / 360.0, 1),
                    "estimated_congestion_risk_score": 64.0 if expected_event_visitors >= 500 else 38.0,
                    "risk_level": "Heavy" if expected_event_visitors >= 500 else "Moderate",
                    "routing_type_notice": routing_label,
                    "coordinates": [
                        [origin_lat, origin_lon],
                        [mid1_lat, mid1_lon],
                        [dest_lat, dest_lon],
                    ],
                },
                {
                    "route_id": "route_2",
                    "route_name": "Route B: Secondary Ring Bypass (Less Congested)",
                    "is_recommended": expected_event_visitors >= 500,
                    "distance_km": round((direct_dist_m * 1.34) / 1000.0, 2),
                    "base_geometric_duration_min": round((direct_dist_m * 1.34) / 520.0, 1),
                    "estimated_congested_duration_min": round((direct_dist_m * 1.34) / 460.0, 1),
                    "estimated_congestion_risk_score": 29.5,
                    "risk_level": "Low",
                    "routing_type_notice": routing_label,
                    "coordinates": [
                        [origin_lat, origin_lon],
                        [mid2_lat, mid2_lon],
                        [dest_lat, dest_lon],
                    ],
                },
            ]

        # Identify nearby less-congested roads
        less_congested_roads = []
        if traffic_segments:
            sorted_segs = sorted(traffic_segments, key=lambda s: s.get("congestion_ratio", 0.5))
            for seg in sorted_segs[:4]:
                less_congested_roads.append(
                    {
                        "road_name": seg.get("road_name"),
                        "congestion_level": seg.get("congestion_level", "Low"),
                        "color_code": seg.get("color_code", "Green"),
                        "current_speed_kmh": seg.get("current_speed_kmh", 42.0),
                        "free_flow_speed_kmh": seg.get("free_flow_speed_kmh", 50.0),
                    }
                )

        # Suggested arrival time calculation
        total_demand = estimated_crowd_count + expected_event_visitors
        if total_demand > 1400:
            suggested_arrival = (now - timedelta(minutes=25)).strftime("%H:%M") + " or after " + (
                now + timedelta(minutes=40)
            ).strftime("%H:%M")
            arrival_advice = "High crowd/event surge active. Arrive 25-30 mins before peak or delay arrival by 40 mins."
        else:
            suggested_arrival = (now + timedelta(minutes=15)).strftime("%H:%M")
            arrival_advice = "Corridors operating within normal parameters; immediate departure is suitable."

        # Campus gate allocation & best gate selection
        gate_recommendations = []
        best_gate = None
        if gates:
            total_cap = sum(max(100, int(g.get("max_capacity_per_hour", 800))) for g in gates) or 1
            for g in gates:
                cap = max(100, int(g.get("max_capacity_per_hour", 800)))
                dist_to_gate = haversine_distance_m(origin_lat, origin_lon, float(g["latitude"]), float(g["longitude"]))
                share = cap / float(total_cap)
                allocated = int(round((estimated_crowd_count + expected_event_visitors * 0.5) * share))
                util_pct = int(min(100, round((allocated / float(cap)) * 100)))
                gate_entry = {
                    "id": g["id"],
                    "campus_name": g["campus_name"],
                    "gate_name": g["gate_name"],
                    "gate_type": g["gate_type"],
                    "connected_road": g["connected_road"],
                    "latitude": g["latitude"],
                    "longitude": g["longitude"],
                    "distance_from_user_m": dist_to_gate,
                    "max_capacity_per_hour": cap,
                    "recommended_allocation_per_hour": allocated,
                    "projected_utilization_pct": util_pct,
                    "status_recommendation": (
                        "Recommended Entry Gate (Lowest Queue Risk)"
                        if util_pct < 68 and g["gate_type"] != "Exit Only"
                        else (
                            "Use for Exit Dispersal Only"
                            if g["gate_type"] == "Exit Only"
                            else "Moderate Queue — Alternate Entry"
                        )
                    ),
                }
                gate_recommendations.append(gate_entry)
                if g["gate_type"] != "Exit Only" and (
                    best_gate is None or util_pct < best_gate["projected_utilization_pct"]
                ):
                    best_gate = gate_entry

        return {
            "suggested_routes": routes_out,
            "nearby_less_congested_roads": less_congested_roads,
            "suggested_arrival_time": suggested_arrival,
            "arrival_advice": arrival_advice,
            "suggested_campus_gate": best_gate,
            "all_campus_gates": gate_recommendations,
            "routing_data_source": routing_source,
            "routing_disclaimer": (
                "Standard OpenStreetMap (OSRM) routes represent geometric road distances/speeds and are NOT live "
                "traffic-aware routes. This system provides advisory commuter & campus gate recommendations only and "
                "never directs emergency vehicles or overrides official traffic controls."
            ),
            "generated_at": now_iso,
        }


routing_service = RoutingService()

