"""
Surrounding Area Recognition & Nearby-Area Analysis Service (OpenStreetMap Overpass API).

Identifies:
- Main roads (primary, trunk, motorway)
- Secondary roads (secondary, tertiary)
- Road intersections & Traffic junctions
- Colleges and schools
- Bus stops
- Railway stations
- Hospitals
- Commercial areas
- Event venues
- Parking areas

Includes a Nearby-Area Analysis Module that identifies critical road segments and potential
congestion points, while explicitly distinguishing roads monitored by a traffic provider
from unmonitored secondary/local segments.
"""
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List
import requests

from config import Config
from database.db import get_db_connection
from services.location_service import haversine_distance_m, validate_radius


class NearbyPlacesService:
    """Queries OpenStreetMap Overpass API and analyzes surrounding roads and POIs."""

    def __init__(self) -> None:
        self.overpass_url = Config.OVERPASS_API_URL
        self.headers = {"User-Agent": Config.NOMINATIM_USER_AGENT}

    @staticmethod
    def _estimate_road_capacity(highway_type: str, lanes: int) -> int:
        per_lane = {
            "motorway": 1900,
            "trunk": 1700,
            "primary": 1400,
            "secondary": 1050,
            "tertiary": 800,
            "residential": 500,
        }.get(highway_type, 850)
        return per_lane * max(1, lanes)

    @staticmethod
    def _categorize_osm_element(tags: Dict[str, str]) -> Dict[str, str]:
        amenity = tags.get("amenity", "")
        highway = tags.get("highway", "")
        railway = tags.get("railway", "")
        leisure = tags.get("leisure", "")
        shop = tags.get("shop", "")
        landuse = tags.get("landuse", "")
        junction = tags.get("junction", "")
        public_transport = tags.get("public_transport", "")

        if highway == "traffic_signals" or junction:
            return {"category": "Traffic junctions", "sub_type": junction or "traffic_signals"}
        if amenity in ("college", "university", "school"):
            return {"category": "Colleges and schools", "sub_type": amenity}
        if highway == "bus_stop" or amenity == "bus_station" or public_transport in ("platform", "station"):
            return {"category": "Bus stops", "sub_type": amenity or highway or public_transport}
        if railway in ("station", "halt", "subway_entrance", "tram_stop"):
            return {"category": "Railway stations", "sub_type": railway}
        if amenity in ("hospital", "clinic", "doctors"):
            return {"category": "Hospitals", "sub_type": amenity}
        if amenity in ("theatre", "arts_centre", "community_centre", "conference_centre", "events_venue") or leisure in (
            "stadium",
            "sports_centre",
        ):
            return {"category": "Event venues", "sub_type": amenity or leisure}
        if amenity == "parking":
            return {"category": "Parking areas", "sub_type": "parking"}
        if shop in ("mall", "department_store", "supermarket") or landuse in ("commercial", "retail"):
            return {"category": "Commercial areas", "sub_type": shop or landuse}
        return {"category": "Other POI", "sub_type": amenity or highway or "poi"}

    def fetch_surrounding_area(
        self,
        latitude: float,
        longitude: float,
        radius_m: int = 1000,
    ) -> Dict[str, Any]:
        """
        Retrieve nearby road network, intersections, and 9 POI categories via Overpass API.
        Falls back to a clearly-labelled coordinate-anchored demonstration topology if Overpass
        is unreachable or times out.
        """
        valid_r = validate_radius(radius_m)
        # Cap Overpass query radius if 10km to avoid public Overpass server timeouts on dense cities
        query_r = min(valid_r, 3500)
        now_iso = datetime.now(timezone.utc).isoformat()

        overpass_query = f"""
        [out:json][timeout:12];
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"](around:{query_r},{latitude},{longitude});
          node["highway"="traffic_signals"](around:{query_r},{latitude},{longitude});
          node["junction"](around:{query_r},{latitude},{longitude});
          nwr["amenity"~"^(college|university|school|hospital|clinic|bus_station|parking|theatre|community_centre|conference_centre|events_venue)$"](around:{query_r},{latitude},{longitude});
          node["highway"="bus_stop"](around:{query_r},{latitude},{longitude});
          nwr["railway"~"^(station|halt|subway_entrance)$"](around:{query_r},{latitude},{longitude});
          nwr["leisure"~"^(stadium|sports_centre)$"](around:{query_r},{latitude},{longitude});
          nwr["shop"~"^(mall|supermarket|department_store)$"](around:{query_r},{latitude},{longitude});
          way["landuse"~"^(commercial|retail)$"](around:{query_r},{latitude},{longitude});
        );
        out body geom 160;
        """

        places: List[Dict[str, Any]] = []
        roads: List[Dict[str, Any]] = []
        intersections: List[Dict[str, Any]] = []
        data_source = "live_openstreetmap_overpass"
        data_label = "Real Geospatial Data (OpenStreetMap Overpass API)"

        try:
            resp = requests.post(
                self.overpass_url,
                data={"data": overpass_query},
                headers=self.headers,
                timeout=10.0,
            )
            if resp.status_code == 200:
                payload = resp.json()
                elements = payload.get("elements", [])
                node_usage_count: Dict[str, Dict[str, Any]] = {}

                for el in elements:
                    el_type = el.get("type")
                    tags = el.get("tags", {})
                    highway = tags.get("highway", "")

                    if el_type == "way" and highway in ("motorway", "trunk", "primary", "secondary", "tertiary"):
                        geom = el.get("geometry", [])
                        if len(geom) >= 2:
                            road_name = (
                                tags.get("name")
                                or tags.get("ref")
                                or f"{highway.capitalize()} Corridor #{el.get('id', '') % 1000}"
                            )
                            road_cat = (
                                "Main roads"
                                if highway in ("motorway", "trunk", "primary")
                                else "Secondary roads"
                            )
                            try:
                                lanes = int(str(tags.get("lanes", "2")).split(";")[0])
                            except ValueError:
                                lanes = 4 if road_cat == "Main roads" else 2
                            try:
                                maxspeed = float(str(tags.get("maxspeed", "50")).split()[0])
                            except ValueError:
                                maxspeed = 60.0 if road_cat == "Main roads" else 40.0

                            coords = [[pt["lat"], pt["lon"]] for pt in geom if "lat" in pt and "lon" in pt]
                            if len(coords) >= 2:
                                start_pt = coords[0]
                                end_pt = coords[-1]
                                mid_pt = coords[len(coords) // 2]
                                dist_m = haversine_distance_m(latitude, longitude, mid_pt[0], mid_pt[1])
                                cap_vph = self._estimate_road_capacity(highway, lanes)
                                # Only primary/trunk/motorway roads are typically monitored continuously by providers
                                provider_monitored = road_cat == "Main roads"
                                congestion_pot = round(
                                    min(0.95, (0.68 if road_cat == "Main roads" else 0.48) + (1.0 / max(1, lanes)) * 0.15),
                                    2,
                                )
                                roads.append(
                                    {
                                        "osm_way_id": str(el.get("id")),
                                        "road_name": road_name,
                                        "road_category": road_cat,
                                        "road_type": highway,
                                        "lanes": lanes,
                                        "maxspeed_kmh": maxspeed,
                                        "capacity_vph": cap_vph,
                                        "provider_monitored": provider_monitored,
                                        "congestion_potential": congestion_pot,
                                        "start_lat": start_pt[0],
                                        "start_lon": start_pt[1],
                                        "end_lat": end_pt[0],
                                        "end_lon": end_pt[1],
                                        "center_lat": mid_pt[0],
                                        "center_lon": mid_pt[1],
                                        "distance_m": dist_m,
                                        "coordinates": coords,
                                    }
                                )
                                # Track shared endpoints to identify road intersections
                                for pt in (start_pt, end_pt):
                                    key = f"{pt[0]:.4f},{pt[1]:.4f}"
                                    if key not in node_usage_count:
                                        node_usage_count[key] = {"lat": pt[0], "lon": pt[1], "roads": set()}
                                    node_usage_count[key]["roads"].add(road_name)
                        continue

                    # Process POIs and junctions
                    lat = el.get("lat")
                    lon = el.get("lon")
                    if (lat is None or lon is None) and "center" in el:
                        lat = el["center"].get("lat")
                        lon = el["center"].get("lon")
                    if (lat is None or lon is None) and "geometry" in el and el["geometry"]:
                        lat = el["geometry"][0].get("lat")
                        lon = el["geometry"][0].get("lon")

                    if lat is None or lon is None:
                        continue

                    cat_info = self._categorize_osm_element(tags)
                    if cat_info["category"] == "Other POI":
                        continue

                    name = (
                        tags.get("name")
                        or tags.get("operator")
                        or f"{cat_info['category'][:-1] if cat_info['category'].endswith('s') else cat_info['category']} ({cat_info['sub_type'].replace('_', ' ').title()})"
                    )
                    dist_m = haversine_distance_m(latitude, longitude, float(lat), float(lon))
                    if dist_m <= valid_r:
                        places.append(
                            {
                                "osm_id": str(el.get("id")),
                                "name": name,
                                "category": cat_info["category"],
                                "sub_type": cat_info["sub_type"],
                                "latitude": round(float(lat), 6),
                                "longitude": round(float(lon), 6),
                                "distance_m": dist_m,
                                "data_source": "live_openstreetmap_overpass",
                            }
                        )

                # Derive road intersections from shared road nodes
                for key, info in node_usage_count.items():
                    if len(info["roads"]) >= 2:
                        r_list = sorted(info["roads"])
                        int_name = f"Intersection: {r_list[0]} & {r_list[1]}"
                        dist_m = haversine_distance_m(latitude, longitude, info["lat"], info["lon"])
                        intersections.append(
                            {
                                "name": int_name,
                                "category": "Road intersections",
                                "sub_type": "intersection",
                                "latitude": round(info["lat"], 6),
                                "longitude": round(info["lon"], 6),
                                "connected_roads": r_list,
                                "distance_m": dist_m,
                                "data_source": "live_openstreetmap_overpass",
                            }
                        )
        except requests.RequestException:
            pass

        # If Overpass was unreachable or returned empty in sandboxed/offline mode, generate
        # coordinate-anchored demonstration topology clearly marked as demonstration_fallback_osm
        if not roads and not places:
            data_source = "demonstration_fallback_osm"
            data_label = "Demonstration Topology (Overpass API Offline / Unreachable)"
            fallback_data = self._build_coordinate_anchored_fallback(latitude, longitude, valid_r)
            roads = fallback_data["roads"]
            places = fallback_data["places"]
            intersections = fallback_data["intersections"]

        # Persist summary snapshot in SQLite
        self._persist_surrounding_snapshot(roads, places, intersections, data_source, now_iso)

        # Run Nearby-Area Analysis Module
        analysis = self.analyze_nearby_area(roads, places, intersections, valid_r)

        return {
            "center": {"latitude": latitude, "longitude": longitude, "radius_m": valid_r},
            "roads": roads[:35],
            "intersections": intersections[:20],
            "places": places[:60],
            "category_counts": analysis["category_counts"],
            "area_analysis": analysis,
            "data_source": data_source,
            "data_label": data_label,
            "provider_monitoring_note": (
                "Only primary arterial corridors are typically monitored by live traffic providers; "
                "secondary/collector roads use capacity-based estimation."
            ),
            "timestamp": now_iso,
        }

    def _build_coordinate_anchored_fallback(
        self, lat: float, lon: float, radius_m: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate realistic road geometry and POI categories anchored around (lat, lon) when offline."""
        scale = min(1.8, max(0.45, radius_m / 1200.0))
        d = 0.0045 * scale

        roads = [
            {
                "osm_way_id": "demo_way_101",
                "road_name": "Primary Urban Arterial Corridor (N-S)",
                "road_category": "Main roads",
                "road_type": "primary",
                "lanes": 4,
                "maxspeed_kmh": 60.0,
                "capacity_vph": 2800,
                "provider_monitored": True,
                "congestion_potential": 0.78,
                "start_lat": round(lat - d * 1.3, 6),
                "start_lon": round(lon - d * 0.2, 6),
                "end_lat": round(lat + d * 1.3, 6),
                "end_lon": round(lon + d * 0.2, 6),
                "center_lat": round(lat, 6),
                "center_lon": round(lon, 6),
                "distance_m": 45.0,
                "coordinates": [
                    [round(lat - d * 1.3, 6), round(lon - d * 0.2, 6)],
                    [round(lat - d * 0.4, 6), round(lon - d * 0.05, 6)],
                    [round(lat, 6), round(lon, 6)],
                    [round(lat + d * 0.5, 6), round(lon + d * 0.08, 6)],
                    [round(lat + d * 1.3, 6), round(lon + d * 0.2, 6)],
                ],
            },
            {
                "osm_way_id": "demo_way_102",
                "road_name": "East-West Institutional Highway",
                "road_category": "Main roads",
                "road_type": "primary",
                "lanes": 4,
                "maxspeed_kmh": 60.0,
                "capacity_vph": 2600,
                "provider_monitored": True,
                "congestion_potential": 0.74,
                "start_lat": round(lat - d * 0.15, 6),
                "start_lon": round(lon - d * 1.4, 6),
                "end_lat": round(lat + d * 0.15, 6),
                "end_lon": round(lon + d * 1.4, 6),
                "center_lat": round(lat, 6),
                "center_lon": round(lon, 6),
                "distance_m": 60.0,
                "coordinates": [
                    [round(lat - d * 0.15, 6), round(lon - d * 1.4, 6)],
                    [round(lat - d * 0.05, 6), round(lon - d * 0.5, 6)],
                    [round(lat, 6), round(lon, 6)],
                    [round(lat + d * 0.08, 6), round(lon + d * 0.6, 6)],
                    [round(lat + d * 0.15, 6), round(lon + d * 1.4, 6)],
                ],
            },
            {
                "osm_way_id": "demo_way_103",
                "road_name": "Campus Ring Collector Road",
                "road_category": "Secondary roads",
                "road_type": "secondary",
                "lanes": 2,
                "maxspeed_kmh": 40.0,
                "capacity_vph": 1400,
                "provider_monitored": False,
                "congestion_potential": 0.64,
                "start_lat": round(lat + d * 0.5, 6),
                "start_lon": round(lon - d * 0.9, 6),
                "end_lat": round(lat + d * 0.55, 6),
                "end_lon": round(lon + d * 0.9, 6),
                "center_lat": round(lat + d * 0.52, 6),
                "center_lon": round(lon, 6),
                "distance_m": round(haversine_distance_m(lat, lon, lat + d * 0.52, lon), 1),
                "coordinates": [
                    [round(lat + d * 0.5, 6), round(lon - d * 0.9, 6)],
                    [round(lat + d * 0.52, 6), round(lon + d * 0.08, 6)],
                    [round(lat + d * 0.55, 6), round(lon + d * 0.9, 6)],
                ],
            },
            {
                "osm_way_id": "demo_way_104",
                "road_name": "South Station Feeder Link Road",
                "road_category": "Secondary roads",
                "road_type": "tertiary",
                "lanes": 2,
                "maxspeed_kmh": 40.0,
                "capacity_vph": 1100,
                "provider_monitored": False,
                "congestion_potential": 0.55,
                "start_lat": round(lat - d * 0.65, 6),
                "start_lon": round(lon - d * 0.85, 6),
                "end_lat": round(lat - d * 0.60, 6),
                "end_lon": round(lon + d * 0.85, 6),
                "center_lat": round(lat - d * 0.62, 6),
                "center_lon": round(lon, 6),
                "distance_m": round(haversine_distance_m(lat, lon, lat - d * 0.62, lon), 1),
                "coordinates": [
                    [round(lat - d * 0.65, 6), round(lon - d * 0.85, 6)],
                    [round(lat - d * 0.62, 6), round(lon - d * 0.05, 6)],
                    [round(lat - d * 0.60, 6), round(lon + d * 0.85, 6)],
                ],
            },
        ]

        intersections = [
            {
                "name": "Intersection: Primary Urban Arterial & East-West Highway",
                "category": "Road intersections",
                "sub_type": "intersection",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "connected_roads": ["Primary Urban Arterial Corridor (N-S)", "East-West Institutional Highway"],
                "distance_m": 15.0,
                "data_source": "demonstration_fallback_osm",
            },
            {
                "name": "Intersection: Primary Arterial & Campus Ring Collector",
                "category": "Road intersections",
                "sub_type": "intersection",
                "latitude": round(lat + d * 0.52, 6),
                "longitude": round(lon + d * 0.08, 6),
                "connected_roads": ["Primary Urban Arterial Corridor (N-S)", "Campus Ring Collector Road"],
                "distance_m": round(haversine_distance_m(lat, lon, lat + d * 0.52, lon + d * 0.08), 1),
                "data_source": "demonstration_fallback_osm",
            },
        ]

        poi_specs = [
            ("Central Signalized Traffic Junction", "Traffic junctions", "traffic_signals", 0.08, 0.05),
            ("North Gate Signal Junction", "Traffic junctions", "junction", 0.52, 0.08),
            ("Institute of Technology & Science Campus", "Colleges and schools", "university", 0.32, 0.28),
            ("Regional Engineering & Arts College", "Colleges and schools", "college", -0.28, 0.35),
            ("Main Campus Arch Bus Stop", "Bus stops", "bus_stop", 0.14, -0.12),
            ("East Corridor Transit Terminal", "Bus stops", "bus_station", 0.08, 0.58),
            ("Suburban Metro & Railway Station", "Railway stations", "station", -0.58, -0.42),
            ("Multi-Specialty General Hospital", "Hospitals", "hospital", -0.35, -0.52),
            ("City Square Commercial & Retail Plaza", "Commercial areas", "commercial", 0.45, -0.48),
            ("University Convocation & Convention Auditorium", "Event venues", "auditorium", 0.25, 0.18),
            ("Multi-Level Visitor Parking Complex P1", "Parking areas", "parking", 0.38, 0.12),
        ]
        places = []
        for idx, (p_name, p_cat, p_sub, d_lat_f, d_lon_f) in enumerate(poi_specs, start=1):
            p_lat = round(lat + d * d_lat_f, 6)
            p_lon = round(lon + d * d_lon_f, 6)
            places.append(
                {
                    "osm_id": f"demo_poi_{idx}",
                    "name": p_name,
                    "category": p_cat,
                    "sub_type": p_sub,
                    "latitude": p_lat,
                    "longitude": p_lon,
                    "distance_m": haversine_distance_m(lat, lon, p_lat, p_lon),
                    "data_source": "demonstration_fallback_osm",
                }
            )

        return {"roads": roads, "places": places, "intersections": intersections}

    def analyze_nearby_area(
        self,
        roads: List[Dict[str, Any]],
        places: List[Dict[str, Any]],
        intersections: List[Dict[str, Any]],
        radius_m: int,
    ) -> Dict[str, Any]:
        """
        Nearby-Area Analysis Module:
        Identifies important road segments, counts all 11 required infrastructure types,
        and pinpoints potential congestion choke points.
        """
        categories = [
            "Main roads",
            "Secondary roads",
            "Road intersections",
            "Traffic junctions",
            "Colleges and schools",
            "Bus stops",
            "Railway stations",
            "Hospitals",
            "Commercial areas",
            "Event venues",
            "Parking areas",
        ]
        counts = {c: 0 for c in categories}
        for r in roads:
            cat = r.get("road_category", "Secondary roads")
            if cat in counts:
                counts[cat] += 1
        counts["Road intersections"] = len(intersections)
        for p in places:
            cat = p.get("category")
            if cat in counts:
                counts[cat] += 1

        # Identify potential congestion points based on proximity of POIs to roads/intersections
        choke_points = []
        for inter in intersections[:6]:
            choke_points.append(
                {
                    "name": inter["name"],
                    "type": "Multi-Corridor Intersection",
                    "latitude": inter["latitude"],
                    "longitude": inter["longitude"],
                    "risk_reason": f"Convergence of {', '.join(inter.get('connected_roads', [])[:2])}",
                    "severity": "High",
                }
            )
        for p in places:
            if p["category"] in ("Colleges and schools", "Event venues", "Railway stations", "Commercial areas"):
                if len(choke_points) < 10:
                    choke_points.append(
                        {
                            "name": f"Access Zone near {p['name']}",
                            "type": p["category"],
                            "latitude": p["latitude"],
                            "longitude": p["longitude"],
                            "risk_reason": f"High pedestrian & vehicle turnover at {p['category'].lower()}.",
                            "severity": "Moderate-High",
                        }
                    )

        monitored_count = sum(1 for r in roads if r.get("provider_monitored"))
        unmonitored_count = max(0, len(roads) - monitored_count)

        return {
            "search_radius_m": radius_m,
            "total_roads_identified": len(roads),
            "provider_monitored_roads": monitored_count,
            "unmonitored_local_roads": unmonitored_count,
            "total_pois_identified": len(places) + len(intersections),
            "category_counts": counts,
            "potential_congestion_points": choke_points,
            "important_road_segments": sorted(
                roads, key=lambda r: r.get("congestion_potential", 0.5), reverse=True
            )[:6],
        }

    @staticmethod
    def _persist_surrounding_snapshot(
        roads: List[Dict[str, Any]],
        places: List[Dict[str, Any]],
        intersections: List[Dict[str, Any]],
        data_source: str,
        now_iso: str,
    ) -> None:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM locations ORDER BY updated_at DESC LIMIT 1")
        loc_row = cursor.fetchone()
        loc_id = loc_row["id"] if loc_row else None

        cursor.execute("DELETE FROM road_segments")
        for r in roads[:25]:
            cursor.execute(
                """
                INSERT INTO road_segments (
                    location_id, osm_way_id, road_name, road_type, lanes,
                    maxspeed_kmh, capacity_vph, is_intersection, start_lat, start_lon,
                    end_lat, end_lon, geometry_json, congestion_potential, data_source, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    loc_id,
                    r.get("osm_way_id"),
                    r.get("road_name"),
                    r.get("road_type", "secondary"),
                    r.get("lanes", 2),
                    r.get("maxspeed_kmh", 50.0),
                    r.get("capacity_vph", 1200),
                    r.get("start_lat", 0.0),
                    r.get("start_lon", 0.0),
                    r.get("end_lat", 0.0),
                    r.get("end_lon", 0.0),
                    json.dumps(r.get("coordinates", [])),
                    r.get("congestion_potential", 0.5),
                    data_source,
                    now_iso,
                ),
            )

        cursor.execute("DELETE FROM nearby_places")
        for p in (places + intersections)[:45]:
            cursor.execute(
                """
                INSERT INTO nearby_places (
                    location_id, osm_id, name, category, sub_type,
                    latitude, longitude, distance_m, importance_score, data_source, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.7, ?, ?)
                """,
                (
                    loc_id,
                    p.get("osm_id", "node"),
                    p.get("name"),
                    p.get("category"),
                    p.get("sub_type", ""),
                    p.get("latitude"),
                    p.get("longitude"),
                    p.get("distance_m", 0.0),
                    data_source,
                    now_iso,
                ),
            )

        conn.commit()
        conn.close()


nearby_places_service = NearbyPlacesService()

