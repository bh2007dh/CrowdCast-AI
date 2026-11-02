"""Database initialization and helper functions for CrowdCast AI."""
import os
import sqlite3
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash
from config import Config


def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """Create and return a SQLite connection with Row factory enabled."""
    target_path = db_path or Config.DATABASE_PATH
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    conn = sqlite3.connect(target_path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = None) -> None:
    """Initialize all 9 required database tables and seed baseline demonstration data."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. users (Admin authentication with securely hashed passwords)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'admin',
            created_at TEXT NOT NULL
        );
        """
    )

    # 2. locations (Area-level reference contexts; avoids storing continuous raw GPS history)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            city TEXT,
            locality TEXT,
            area_name TEXT,
            formatted_address TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            radius_m INTEGER NOT NULL DEFAULT 1000,
            is_campus INTEGER NOT NULL DEFAULT 0,
            data_source TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    # 3. nearby_places (POIs, schools, colleges, transit hubs, hospitals, parking, venues)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS nearby_places (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id INTEGER,
            osm_id TEXT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            sub_type TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            distance_m REAL,
            importance_score REAL DEFAULT 0.5,
            data_source TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (location_id) REFERENCES locations (id) ON DELETE CASCADE
        );
        """
    )

    # 4. road_segments (Identified main/secondary roads, intersections, junctions)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS road_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id INTEGER,
            osm_way_id TEXT,
            road_name TEXT NOT NULL,
            road_type TEXT NOT NULL,
            lanes INTEGER DEFAULT 2,
            maxspeed_kmh REAL DEFAULT 50.0,
            capacity_vph INTEGER DEFAULT 1200,
            is_intersection INTEGER DEFAULT 0,
            start_lat REAL NOT NULL,
            start_lon REAL NOT NULL,
            end_lat REAL NOT NULL,
            end_lon REAL NOT NULL,
            geometry_json TEXT,
            congestion_potential REAL DEFAULT 0.5,
            data_source TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (location_id) REFERENCES locations (id) ON DELETE CASCADE
        );
        """
    )

    # 5. traffic_observations (Traffic flow snapshots with explicit data_source: live vs demonstration)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS traffic_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            road_segment_id INTEGER,
            road_name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            current_speed_kmh REAL NOT NULL,
            free_flow_speed_kmh REAL NOT NULL,
            delay_seconds REAL NOT NULL DEFAULT 0.0,
            congestion_ratio REAL NOT NULL,
            congestion_level TEXT NOT NULL,
            color_code TEXT NOT NULL,
            confidence REAL DEFAULT 0.85,
            provider_name TEXT NOT NULL,
            data_source TEXT NOT NULL,
            is_simulated INTEGER NOT NULL DEFAULT 0,
            observed_at TEXT NOT NULL,
            FOREIGN KEY (road_segment_id) REFERENCES road_segments (id) ON DELETE SET NULL
        );
        """
    )

    # 6. events (Campus & public events affecting crowd and traffic)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            event_location TEXT NOT NULL,
            event_category TEXT NOT NULL,
            event_date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            expected_participants INTEGER NOT NULL DEFAULT 200,
            entry_gates_count INTEGER NOT NULL DEFAULT 2,
            expected_vehicles INTEGER NOT NULL DEFAULT 60,
            latitude REAL,
            longitude REAL,
            notes TEXT,
            data_source TEXT NOT NULL DEFAULT 'user_input',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    # 7. crowd_predictions (ML predictions with horizon trends and data_source labels)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS crowd_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_label TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            estimated_crowd_count INTEGER NOT NULL,
            crowd_level TEXT NOT NULL,
            pred_15m INTEGER NOT NULL,
            pred_30m INTEGER NOT NULL,
            pred_60m INTEGER NOT NULL,
            predicted_peak_time TEXT NOT NULL,
            expected_congestion_level TEXT NOT NULL,
            confidence_score REAL,
            model_name TEXT NOT NULL,
            data_source TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )

    # 8. congestion_predictions (Crowd-Traffic Fusion Engine risk score outputs)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS congestion_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_label TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            color_code TEXT NOT NULL,
            traffic_component REAL NOT NULL,
            crowd_component REAL NOT NULL,
            historical_component REAL NOT NULL,
            event_component REAL NOT NULL,
            expected_peak_period TEXT NOT NULL,
            contributing_factors_json TEXT NOT NULL,
            recommended_actions_json TEXT NOT NULL,
            weights_used_json TEXT NOT NULL,
            data_source TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )

    # 9. campus_gates (Admin-registered campus entry/exit gates for smart allocation)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS campus_gates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campus_name TEXT NOT NULL,
            gate_name TEXT NOT NULL,
            gate_type TEXT NOT NULL DEFAULT 'Entry/Exit',
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            max_capacity_per_hour INTEGER NOT NULL DEFAULT 800,
            vehicle_lanes INTEGER NOT NULL DEFAULT 2,
            connected_road TEXT,
            status TEXT NOT NULL DEFAULT 'Open',
            data_source TEXT NOT NULL DEFAULT 'admin_registered',
            created_at TEXT NOT NULL
        );
        """
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    # Seed default admin user if not present
    cursor.execute("SELECT id FROM users WHERE username = ?", (Config.ADMIN_USERNAME,))
    if not cursor.fetchone():
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, role, created_at)
            VALUES (?, ?, 'admin', ?)
            """,
            (
                Config.ADMIN_USERNAME,
                generate_password_hash(Config.ADMIN_DEFAULT_PASSWORD),
                now_iso,
            ),
        )

    # Seed default demonstration campus location, gates, and sample events if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM campus_gates")
    if cursor.fetchone()["cnt"] == 0:
        sample_gates = [
            (
                "Main University Campus",
                "Gate 1 - North Main Arch",
                "Entry/Exit",
                13.0108,
                80.2354,
                1200,
                3,
                "Sardar Patel Arterial Road",
                "Open",
                "demonstration_seed",
                now_iso,
            ),
            (
                "Main University Campus",
                "Gate 2 - East Tech Park Access",
                "Entry Only",
                13.0125,
                80.2389,
                900,
                2,
                "Gandhi Mandapam Link Road",
                "Open",
                "demonstration_seed",
                now_iso,
            ),
            (
                "Main University Campus",
                "Gate 3 - West Hostel & Staff Gate",
                "Entry/Exit",
                13.0089,
                80.2318,
                650,
                2,
                "Kotturpuram Bridge Approach",
                "Open",
                "demonstration_seed",
                now_iso,
            ),
            (
                "Main University Campus",
                "Gate 4 - South Auditorium Exit",
                "Exit Only",
                13.0068,
                80.2361,
                750,
                2,
                "Adyar Canal Service Road",
                "Open",
                "demonstration_seed",
                now_iso,
            ),
        ]
        cursor.executemany(
            """
            INSERT INTO campus_gates (
                campus_name, gate_name, gate_type, latitude, longitude,
                max_capacity_per_hour, vehicle_lanes, connected_road, status, data_source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            sample_gates,
        )

    cursor.execute("SELECT COUNT(*) as cnt FROM events")
    if cursor.fetchone()["cnt"] == 0:
        today_str = datetime.now().strftime("%Y-%m-%d")
        sample_events = [
            (
                "TechnoSphere Annual AI & Robotics Symposium",
                "Main Auditorium & Innovation Complex",
                "College symposiums",
                today_str,
                "09:00",
                "17:30",
                1450,
                3,
                380,
                13.0108,
                80.2354,
                "Keynote sessions at 09:30 and 14:00; high external student influx expected.",
                "demonstration_seed",
                now_iso,
                now_iso,
            ),
            (
                "Inter-University Athletics & Football Championship",
                "University Sports Complex & Stadium",
                "Sports events",
                today_str,
                "15:30",
                "20:00",
                2200,
                2,
                520,
                13.0122,
                80.2375,
                "Evening peak overlap with regular commuter traffic along East Corridor.",
                "demonstration_seed",
                now_iso,
                now_iso,
            ),
            (
                "Undergraduate Admissions & Counseling Drive",
                "Administrative Block Hall B",
                "College admissions",
                today_str,
                "08:30",
                "16:00",
                850,
                2,
                310,
                13.0095,
                80.2338,
                "High parent vehicle arrivals and parking turnover near Gate 1 & Gate 3.",
                "demonstration_seed",
                now_iso,
                now_iso,
            ),
        ]
        cursor.executemany(
            """
            INSERT INTO events (
                event_name, event_location, event_category, event_date,
                start_time, end_time, expected_participants, entry_gates_count,
                expected_vehicles, latitude, longitude, notes, data_source, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            sample_events,
        )

    cursor.execute("SELECT COUNT(*) as cnt FROM locations WHERE is_campus = 1")
    if cursor.fetchone()["cnt"] == 0:
        cursor.execute(
            """
            INSERT INTO locations (
                label, city, locality, area_name, formatted_address,
                latitude, longitude, radius_m, is_campus, data_source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 'demonstration_seed', ?)
            """,
            (
                "Main University Campus Hub",
                "Chennai",
                "Guindy / Adyar Education Corridor",
                "Anna University & IIT Madras Institutional Zone",
                "Sardar Patel Road, Guindy Institutional Area, Chennai, Tamil Nadu 600025",
                13.0108,
                80.2354,
                2000,
                now_iso,
            ),
        )

    conn.commit()
    conn.close()

