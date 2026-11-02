"""
Automated Test Suite for CrowdCast AI (Section 14 Verification).
Tests:
1. Flask server startup & all HTML pages
2. SQLite database initialization (all 9 required tables)
3. Location API & coordinate validation
4. Manual location search entry
5. Nearby place & road recognition (11 categories)
6. Traffic provider integration & explicit demonstration fallback mode
7. ML Crowd Prediction API (+15m, +30m, +60m trends)
8. Event creation, update, impact analysis, and deletion
9. Crowd-Traffic Fusion Risk Score (0-100) & Gate/Route recommendations
10. Report generation (JSON summary & downloadable CSV)
11. Admin hashed-password authentication & gate registration
"""
import os
import sys
import tempfile
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import create_app
from config import Config


@pytest.fixture()
def client():
    db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    app = create_app({"TESTING": True, "DATABASE_PATH": temp_db_path})
    # Ensure services use the test database path during tests
    old_db_path = Config.DATABASE_PATH
    Config.DATABASE_PATH = temp_db_path
    with app.test_client() as test_client:
        yield test_client
    Config.DATABASE_PATH = old_db_path
    if os.path.exists(temp_db_path):
        try:
            os.remove(temp_db_path)
        except OSError:
            pass


def test_health_and_database_initialization(client):
    """Test 1 & 2: Verify server startup, /api/health, and all 9 required SQLite tables."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["database_connected"] is True
    assert data["ml_model_loaded"] is True

    expected_tables = {
        "users",
        "locations",
        "nearby_places",
        "road_segments",
        "traffic_observations",
        "events",
        "crowd_predictions",
        "congestion_predictions",
        "campus_gates",
    }
    assert expected_tables.issubset(set(data["database_tables"]))


def test_all_website_pages_render(client):
    """Verify all 8 website pages render with HTTP 200."""
    for route in ["/", "/dashboard", "/traffic", "/prediction", "/events", "/congestion", "/reports", "/admin"]:
        resp = client.get(route)
        assert resp.status_code == 200
        assert b"CrowdCast AI" in resp.data


def test_location_detection_and_validation(client):
    """Test 3: Location API with valid coordinates and rejection of invalid coordinates."""
    valid_resp = client.post(
        "/api/location/detect",
        json={"latitude": 13.0108, "longitude": 80.2354, "radius_m": 2000, "source_mode": "browser_geolocation"},
    )
    assert valid_resp.status_code == 200
    loc = valid_resp.get_json()["location"]
    assert loc["latitude"] == 13.0108
    assert loc["longitude"] == 80.2354
    assert loc["radius_m"] == 2000
    assert "city" in loc and "locality" in loc

    # Invalid coordinates (latitude > 90)
    bad_resp = client.post("/api/location/detect", json={"latitude": 125.0, "longitude": 80.0})
    assert bad_resp.status_code == 400

    # Null Island (0, 0) rejection
    null_resp = client.post("/api/location/detect", json={"latitude": 0.0, "longitude": 0.0})
    assert null_resp.status_code == 400


def test_manual_location_search(client):
    """Test 4: Manual location search fallback when GPS permission is denied."""
    resp = client.get("/api/location/search?q=Chennai")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["count"] >= 1
    assert "latitude" in data["results"][0]


def test_nearby_place_and_road_recognition(client):
    """Test 5: Surrounding area recognition across 11 categories and nearby-area analysis."""
    resp = client.get("/api/location/nearby?lat=13.0108&lon=80.2354&radius_m=1000")
    assert resp.status_code == 200
    sur = resp.get_json()["surrounding"]
    assert len(sur["roads"]) >= 1
    assert "area_analysis" in sur
    expected_categories = {
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
    }
    assert expected_categories.issubset(set(sur["category_counts"].keys()))


def test_traffic_integration_and_fallback_mode(client):
    """Test 6 & 7: Traffic flow & incidents API with transparent Demonstration Mode labelling."""
    resp = client.get("/api/traffic/current?lat=13.0108&lon=80.2354&radius_m=1000")
    assert resp.status_code == 200
    traffic = resp.get_json()["traffic"]
    assert "summary" in traffic
    assert traffic["summary"]["color_code"] in ("Green", "Yellow", "Orange", "Red")
    assert "data_source" in traffic
    assert "is_simulated" in traffic
    assert len(traffic["segments"]) >= 1

    inc_resp = client.get("/api/traffic/incidents?lat=13.0108&lon=80.2354&radius_m=2000")
    assert inc_resp.status_code == 200
    inc_data = inc_resp.get_json()["incidents_data"]
    assert "incidents" in inc_data


def test_ml_crowd_prediction_api(client):
    """Test 8: ML Crowd Prediction API (+15m, +30m, +60m horizons & synthetic label)."""
    resp = client.post(
        "/api/prediction/crowd",
        json={
            "latitude": 13.0108,
            "longitude": 80.2354,
            "hour": 9,
            "day_of_week": 1,
            "historical_crowd_count": 700,
            "event_type": "Technical festivals",
            "event_start_time": "09:00",
            "event_end_time": "17:00",
            "expected_visitors": 1800,
            "nearby_traffic_level": 0.58,
            "historical_traffic_conditions": 0.40,
        },
    )
    assert resp.status_code == 200
    pred = resp.get_json()["prediction"]
    assert pred["estimated_crowd_count"] > 0
    assert "next_15_min" in pred["predictions_horizon"]
    assert "next_30_min" in pred["predictions_horizon"]
    assert "next_60_min" in pred["predictions_horizon"]
    assert pred["data_source"] == "ml_prediction_synthetic_trained"
    assert "ensemble_agreement_pct" in pred["confidence_metrics"]


def test_event_crud_and_impact_analysis(client):
    """Test 9: Event creation, listing, updating, Normal vs. Event-Day comparison, and deletion."""
    create_resp = client.post(
        "/api/events",
        json={
            "event_name": "Annual Cultural Fest 2026",
            "event_location": "Open Air Theatre",
            "event_category": "Cultural events",
            "event_date": "2026-11-15",
            "start_time": "10:00",
            "end_time": "20:00",
            "expected_participants": 1600,
            "entry_gates_count": 3,
            "expected_vehicles": 420,
        },
    )
    assert create_resp.status_code == 201
    ev = create_resp.get_json()["event"]
    ev_id = ev["id"]
    assert "impact_analysis" in ev
    assert ev["impact_analysis"]["event_day_predicted_crowd"] >= ev["impact_analysis"]["normal_baseline_crowd"]

    # Update event
    put_resp = client.put(
        f"/api/events/{ev_id}",
        json={
            "event_name": "Annual Cultural Fest 2026 (Updated)",
            "event_location": "Open Air Theatre",
            "event_category": "Cultural events",
            "event_date": "2026-11-15",
            "start_time": "10:00",
            "end_time": "21:00",
            "expected_participants": 2000,
            "entry_gates_count": 4,
            "expected_vehicles": 500,
        },
    )
    assert put_resp.status_code == 200
    assert put_resp.get_json()["event"]["expected_participants"] == 2000

    # Delete event
    del_resp = client.delete(f"/api/events/{ev_id}")
    assert del_resp.status_code == 200


def test_congestion_fusion_risk_score_and_routing(client):
    """Test 10: Crowd-Traffic Fusion Risk Score (0-100) and OSRM route & gate recommendations."""
    resp = client.post(
        "/api/prediction/congestion",
        json={
            "latitude": 13.0108,
            "longitude": 80.2354,
            "traffic_congestion_ratio": 0.65,
            "estimated_crowd_count": 1500,
            "expected_event_visitors": 1200,
            "expected_event_vehicles": 350,
        },
    )
    assert resp.status_code == 200
    fusion = resp.get_json()["fusion"]
    assert 0.0 <= fusion["congestion_risk_score"] <= 100.0
    assert fusion["color_code"] in ("Green", "Yellow", "Orange", "Red")
    assert len(fusion["main_contributing_factors"]) == 4
    assert len(fusion["recommended_actions"]) >= 1
    assert "routing_and_gates" in fusion
    assert len(fusion["routing_and_gates"]["suggested_routes"]) >= 1


def test_reports_summary_and_csv_export(client):
    """Test 11: Reports summary JSON and downloadable CSV endpoint."""
    sum_resp = client.get("/api/reports/summary")
    assert sum_resp.status_code == 200
    sum_data = sum_resp.get_json()
    assert "location_analysis_report" in sum_data
    assert "traffic_summary" in sum_data
    assert "crowd_prediction_report" in sum_data
    assert "event_impact_report" in sum_data

    csv_resp = client.get("/api/reports/csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.content_type
    assert b"CROWDCAST AI" in csv_resp.data


def test_admin_authentication_and_gate_registration(client):
    """Test 12: Admin hashed password login and campus gate registration."""
    bad_login = client.post("/api/admin/login", json={"username": "admin", "password": "wrong_password"})
    assert bad_login.status_code == 401

    ok_login = client.post(
        "/api/admin/login", json={"username": Config.ADMIN_USERNAME, "password": Config.ADMIN_DEFAULT_PASSWORD}
    )
    assert ok_login.status_code == 200
    assert ok_login.get_json()["authenticated"] is True

    gate_resp = client.post(
        "/api/admin/gates",
        json={
            "campus_name": "Main University Campus",
            "gate_name": "Gate 6 - Innovation Annex",
            "gate_type": "Entry/Exit",
            "latitude": 13.0119,
            "longitude": 80.2371,
            "max_capacity_per_hour": 950,
            "vehicle_lanes": 2,
            "connected_road": "East Link Road",
        },
    )
    assert gate_resp.status_code == 201

