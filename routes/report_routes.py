"""
Reports & CSV Export Routes (Section 9G & 11).
Implements:
- GET /api/reports/summary
- GET /api/reports/csv
"""
import csv
import io
from datetime import datetime, timezone
from flask import Blueprint, Response, jsonify

from database.db import get_db_connection
from routes.event_routes import compute_event_impact_comparison
from services.traffic_service import traffic_service

report_bp = Blueprint("report_bp", __name__)


@report_bp.route("/api/reports/summary", methods=["GET"])
def get_reports_summary():
    """
    Compile comprehensive analytical summary:
    1. Location analysis report
    2. Traffic summary
    3. Crowd prediction report
    4. Event impact report
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM locations ORDER BY updated_at DESC LIMIT 5")
    locations = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT category, COUNT(*) as count FROM nearby_places GROUP BY category")
    poi_summary = {r["category"]: r["count"] for r in cursor.fetchall()}

    cursor.execute("SELECT * FROM road_segments ORDER BY congestion_potential DESC LIMIT 10")
    road_segments = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM traffic_observations ORDER BY id DESC LIMIT 15")
    traffic_obs = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM crowd_predictions ORDER BY id DESC LIMIT 15")
    crowd_preds = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM congestion_predictions ORDER BY id DESC LIMIT 15")
    congestion_preds = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM events ORDER BY event_date DESC, start_time ASC")
    events = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM campus_gates ORDER BY id ASC")
    gates = [dict(r) for r in cursor.fetchall()]

    conn.close()

    enriched_events = []
    for ev in events:
        ev_copy = dict(ev)
        ev_copy["impact_analysis"] = compute_event_impact_comparison(ev_copy)
        enriched_events.append(ev_copy)

    avg_speed = (
        round(sum(float(t["current_speed_kmh"]) for t in traffic_obs) / len(traffic_obs), 1)
        if traffic_obs
        else 36.5
    )
    avg_delay = (
        round(sum(float(t["delay_seconds"]) for t in traffic_obs) / len(traffic_obs), 1)
        if traffic_obs
        else 42.0
    )

    return (
        jsonify(
            {
                "status": "ok",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "provider_status": traffic_service.get_provider_status(),
                "location_analysis_report": {
                    "active_locations": locations,
                    "poi_category_breakdown": poi_summary,
                    "monitored_road_segments": road_segments,
                    "campus_gates": gates,
                },
                "traffic_summary": {
                    "recent_observations_count": len(traffic_obs),
                    "average_observed_speed_kmh": avg_speed,
                    "average_delay_seconds": avg_delay,
                    "observations": traffic_obs,
                },
                "crowd_prediction_report": {
                    "predictions_logged": len(crowd_preds),
                    "recent_crowd_predictions": crowd_preds,
                    "recent_fusion_predictions": congestion_preds,
                },
                "event_impact_report": {
                    "scheduled_events_count": len(enriched_events),
                    "total_expected_participants": sum(int(e["expected_participants"]) for e in enriched_events),
                    "total_expected_vehicles": sum(int(e["expected_vehicles"]) for e in enriched_events),
                    "events": enriched_events,
                },
            }
        ),
        200,
    )


@report_bp.route("/api/reports/csv", methods=["GET"])
def download_csv_report():
    """Generate and stream a downloadable CSV report covering Location, Traffic, Crowd, and Events."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM locations ORDER BY updated_at DESC LIMIT 5")
    locations = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM traffic_observations ORDER BY id DESC LIMIT 20")
    traffic_obs = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM crowd_predictions ORDER BY id DESC LIMIT 20")
    crowd_preds = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM congestion_predictions ORDER BY id DESC LIMIT 20")
    congestion_preds = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT * FROM events ORDER BY event_date DESC")
    events = [dict(r) for r in cursor.fetchall()]
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["CROWDCAST AI - LOCATION-AWARE INTELLIGENT CROWD & TRAFFIC REPORT"])
    writer.writerow(["Generated At (UTC)", datetime.now(timezone.utc).isoformat()])
    writer.writerow(
        [
            "Data Integrity Note",
            "Every record explicitly lists its data_source (live_tomtom_api, live_openstreetmap_overpass, ml_prediction_synthetic_trained, or demonstration_sample_traffic).",
        ]
    )
    writer.writerow([])

    writer.writerow(["=== 1. DETECTED & REGISTERED LOCATIONS ==="])
    writer.writerow(["ID", "Label", "City", "Locality", "Area", "Latitude", "Longitude", "Radius (m)", "Data Source", "Updated At"])
    for loc in locations:
        writer.writerow(
            [
                loc.get("id"),
                loc.get("label"),
                loc.get("city"),
                loc.get("locality"),
                loc.get("area_name"),
                loc.get("latitude"),
                loc.get("longitude"),
                loc.get("radius_m"),
                loc.get("data_source"),
                loc.get("updated_at"),
            ]
        )
    writer.writerow([])

    writer.writerow(["=== 2. TRAFFIC OBSERVATIONS ==="])
    writer.writerow(
        [
            "ID",
            "Road Name",
            "Current Speed (km/h)",
            "Free Flow Speed (km/h)",
            "Delay (s)",
            "Congestion Level",
            "Color Indicator",
            "Is Simulated",
            "Data Source",
            "Observed At",
        ]
    )
    for t in traffic_obs:
        writer.writerow(
            [
                t.get("id"),
                t.get("road_name"),
                t.get("current_speed_kmh"),
                t.get("free_flow_speed_kmh"),
                t.get("delay_seconds"),
                t.get("congestion_level"),
                t.get("color_code"),
                "Yes (Demo)" if t.get("is_simulated") else "No (Live)",
                t.get("data_source"),
                t.get("observed_at"),
            ]
        )
    writer.writerow([])

    writer.writerow(["=== 3. ML CROWD PREDICTIONS ==="])
    writer.writerow(
        [
            "ID",
            "Location",
            "Estimated Crowd",
            "Crowd Level",
            "+15m Forecast",
            "+30m Forecast",
            "+60m Forecast",
            "Peak Window",
            "Expected Congestion",
            "Ensemble Agreement (%)",
            "Data Source",
            "Created At",
        ]
    )
    for cp in crowd_preds:
        writer.writerow(
            [
                cp.get("id"),
                cp.get("location_label"),
                cp.get("estimated_crowd_count"),
                cp.get("crowd_level"),
                cp.get("pred_15m"),
                cp.get("pred_30m"),
                cp.get("pred_60m"),
                cp.get("predicted_peak_time"),
                cp.get("expected_congestion_level"),
                cp.get("confidence_score"),
                cp.get("data_source"),
                cp.get("created_at"),
            ]
        )
    writer.writerow([])

    writer.writerow(["=== 4. CROWD-TRAFFIC FUSION RISK SCORES ==="])
    writer.writerow(
        [
            "ID",
            "Location",
            "Risk Score (0-100)",
            "Risk Level",
            "Color",
            "Traffic Subscore",
            "Crowd Subscore",
            "Historical Subscore",
            "Event Subscore",
            "Expected Peak Period",
            "Data Source",
            "Created At",
        ]
    )
    for fp in congestion_preds:
        writer.writerow(
            [
                fp.get("id"),
                fp.get("location_label"),
                fp.get("risk_score"),
                fp.get("risk_level"),
                fp.get("color_code"),
                fp.get("traffic_component"),
                fp.get("crowd_component"),
                fp.get("historical_component"),
                fp.get("event_component"),
                fp.get("expected_peak_period"),
                fp.get("data_source"),
                fp.get("created_at"),
            ]
        )
    writer.writerow([])

    writer.writerow(["=== 5. SCHEDULED EVENTS & EXPECTED IMPACT ==="])
    writer.writerow(
        [
            "ID",
            "Event Name",
            "Location",
            "Category",
            "Date",
            "Start Time",
            "End Time",
            "Expected Participants",
            "Entry Gates",
            "Expected Vehicles",
            "Data Source",
        ]
    )
    for ev in events:
        writer.writerow(
            [
                ev.get("id"),
                ev.get("event_name"),
                ev.get("event_location"),
                ev.get("event_category"),
                ev.get("event_date"),
                ev.get("start_time"),
                ev.get("end_time"),
                ev.get("expected_participants"),
                ev.get("entry_gates_count"),
                ev.get("expected_vehicles"),
                ev.get("data_source"),
            ]
        )

    csv_bytes = output.getvalue()
    filename = f"crowdcast_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

